"""Database for the NutriCoco field app: shops and their (changeable) agreements, field staff, GPS pings,
store visits found by geofencing, and route statistics (time, km, fuel). SQLite, built into Python.

Field staff deliver to shops and check the fridge in the same visit. Their phone sends a GPS ping every
few seconds while the app is open. Each ping is checked against the shops' geofences:
  - inside a shop's circle and no open visit       -> check-in (event "check_in")
  - open visit and clearly outside (radius x 1.5)  -> check-out (event "check_out")
Pings with poor accuracy (> gps_max_accuracy_m) never change a visit and are not counted in the distance,
so GPS noise inside a building does not create false check-outs or kilometres.

A shop's agreement (cups, shelf, zone) can change at any time. Changes are stored as new rows with a date,
never overwritten, so an audit is always compared with the agreement that was valid on that day.
"""
import math
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS shops (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, area TEXT, channel TEXT DEFAULT 'retail',
  lat REAL NOT NULL, lon REAL NOT NULL, radius_m REAL DEFAULT 80, active INTEGER DEFAULT 1, sample INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS agreements (
  id INTEGER PRIMARY KEY, shop_id INTEGER NOT NULL REFERENCES shops(id), cups INTEGER NOT NULL,
  shelf INTEGER, zone TEXT, daily_fee_tk REAL, note TEXT, valid_from TEXT NOT NULL, created_by TEXT);
CREATE TABLE IF NOT EXISTS staff (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, phone TEXT, vehicle TEXT DEFAULT 'motorcycle', active INTEGER DEFAULT 1);
CREATE TABLE IF NOT EXISTS pings (
  id INTEGER PRIMARY KEY, staff_id INTEGER NOT NULL REFERENCES staff(id), t TEXT NOT NULL,
  lat REAL NOT NULL, lon REAL NOT NULL, acc REAL, ok INTEGER DEFAULT 1);
CREATE INDEX IF NOT EXISTS pings_staff_t ON pings(staff_id, t);
CREATE TABLE IF NOT EXISTS visits (
  id INTEGER PRIMARY KEY, staff_id INTEGER NOT NULL REFERENCES staff(id), shop_id INTEGER NOT NULL REFERENCES shops(id),
  check_in TEXT NOT NULL, check_out TEXT, in_lat REAL, in_lon REAL, in_dist_m REAL, how TEXT DEFAULT 'geofence');
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY, t TEXT NOT NULL, staff_id INTEGER, shop_id INTEGER, visit_id INTEGER, kind TEXT NOT NULL);
"""


def iso(dt):
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse(t):
    return datetime.fromisoformat(t)


def dist_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres (haversine)."""
    k = math.pi / 180
    h = (math.sin((lat2 - lat1) * k / 2) ** 2
         + math.cos(lat1 * k) * math.cos(lat2 * k) * math.sin((lon2 - lon1) * k / 2) ** 2)
    return 2 * 6371000 * math.asin(math.sqrt(h))


class DB:
    def __init__(self, path, cfg):
        """cfg: the 'tracking' section of configs/config.yaml."""
        self.cfg = cfg
        self.con = sqlite3.connect(str(path), check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA foreign_keys = ON")
        self.con.executescript(SCHEMA)
        self.lock = threading.Lock()

    def q(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.con.execute(sql, args).fetchall()]

    def x(self, sql, args=()):
        with self.lock, self.con:
            return self.con.execute(sql, args).lastrowid

    # ---------- shops and agreements ----------
    def add_shop(self, name, lat, lon, area="", channel="retail", radius_m=None, sample=False):
        return self.x("INSERT INTO shops (name, area, channel, lat, lon, radius_m, sample) VALUES (?,?,?,?,?,?,?)",
                      (name, area, channel, lat, lon, radius_m or self.cfg["geofence_radius_m"], int(sample)))

    def set_agreement(self, shop_id, cups, shelf=None, zone=None, daily_fee_tk=None, note="", by="", valid_from=None):
        if int(cups) < 0:
            raise ValueError("cups must be 0 or more")
        if not self.q("SELECT id FROM shops WHERE id=?", (shop_id,)):
            raise KeyError(f"unknown shop {shop_id}")
        return self.x("INSERT INTO agreements (shop_id, cups, shelf, zone, daily_fee_tk, note, valid_from, created_by) "
                      "VALUES (?,?,?,?,?,?,?,?)",
                      (shop_id, int(cups), shelf, zone, daily_fee_tk, note, valid_from or iso(datetime.now(timezone.utc)), by))

    def agreement(self, shop_id, at=None):
        """The agreement valid at time `at` (default now), or None."""
        rows = self.q("SELECT * FROM agreements WHERE shop_id=? AND valid_from<=? ORDER BY valid_from DESC, id DESC LIMIT 1",
                      (shop_id, at or iso(datetime.now(timezone.utc))))
        return rows[0] if rows else None

    def shops(self):
        out = self.q("SELECT * FROM shops WHERE active=1 ORDER BY name")
        for s in out:
            s["agreement"] = self.agreement(s["id"])
        return out

    # ---------- staff and GPS ----------
    def add_staff(self, name, phone="", vehicle="motorcycle"):
        return self.x("INSERT INTO staff (name, phone, vehicle) VALUES (?,?,?)", (name, phone, vehicle))

    def staff(self):
        return self.q("SELECT * FROM staff WHERE active=1 ORDER BY name")

    def open_visit(self, staff_id):
        rows = self.q("SELECT * FROM visits WHERE staff_id=? AND check_out IS NULL ORDER BY id DESC LIMIT 1", (staff_id,))
        return rows[0] if rows else None

    def event(self, t, kind, staff_id, shop_id=None, visit_id=None):
        self.x("INSERT INTO events (t, staff_id, shop_id, visit_id, kind) VALUES (?,?,?,?,?)",
               (t, staff_id, shop_id, visit_id, kind))
        return {"kind": kind, "t": t, "shop_id": shop_id, "visit_id": visit_id}

    def ping(self, staff_id, lat, lon, acc=None, t=None):
        """Store one GPS position and apply the geofence. Returns the events it caused (check_in / check_out)."""
        if not self.q("SELECT id FROM staff WHERE id=?", (staff_id,)):
            raise KeyError(f"unknown staff {staff_id}")
        t = t or iso(datetime.now(timezone.utc))
        # a fix is used only if it is accurate enough and not a jump faster than the vehicle can go from the
        # last used fix; rejected fixes are stored (ok=0) but never start/end a visit or count as distance
        prev = self.q("SELECT * FROM pings WHERE staff_id=? AND ok=1 AND t<=? ORDER BY t DESC, id DESC LIMIT 1", (staff_id, t))
        ok = acc is None or acc <= self.cfg["gps_max_accuracy_m"]
        if ok and prev:
            hours = max((parse(t) - parse(prev[0]["t"])).total_seconds(), 1) / 3600
            ok = dist_m(prev[0]["lat"], prev[0]["lon"], lat, lon) / 1000 / hours <= self.cfg["max_speed_kmh"]
        self.x("INSERT INTO pings (staff_id, t, lat, lon, acc, ok) VALUES (?,?,?,?,?,?)", (staff_id, t, lat, lon, acc, int(ok)))
        if not ok:
            return []
        events, v = [], self.open_visit(staff_id)
        if v:
            shop = self.q("SELECT * FROM shops WHERE id=?", (v["shop_id"],))[0]
            if dist_m(lat, lon, shop["lat"], shop["lon"]) <= shop["radius_m"] * self.cfg["checkout_factor"]:
                return events
            self.x("UPDATE visits SET check_out=? WHERE id=?", (t, v["id"]))
            events.append(self.event(t, "check_out", staff_id, v["shop_id"], v["id"]))
        inside = sorted((dist_m(lat, lon, s["lat"], s["lon"]), s["id"]) for s in self.q("SELECT * FROM shops WHERE active=1")
                        if dist_m(lat, lon, s["lat"], s["lon"]) <= s["radius_m"])
        if inside:  # (after a check-out the old shop is > radius away, so this is a new shop)
            d, shop_id = inside[0]
            vid = self.x("INSERT INTO visits (staff_id, shop_id, check_in, in_lat, in_lon, in_dist_m) VALUES (?,?,?,?,?,?)",
                         (staff_id, shop_id, t, lat, lon, round(d, 1)))
            events.append(self.event(t, "check_in", staff_id, shop_id, vid))
        return events

    def latest_positions(self, max_age_sec):
        since = iso(datetime.now(timezone.utc) - timedelta(seconds=max_age_sec))
        return self.q("SELECT s.id AS staff_id, s.name, p.t, p.lat, p.lon, p.acc FROM staff s JOIN pings p ON p.id = "
                      "(SELECT id FROM pings WHERE staff_id=s.id ORDER BY t DESC, id DESC LIMIT 1) WHERE p.t>=?", (since,))

    # ---------- reports ----------
    def visits(self, day=None, staff_id=None):
        sql = ("SELECT v.*, s.name AS shop, st.name AS staff FROM visits v JOIN shops s ON s.id=v.shop_id "
               "JOIN staff st ON st.id=v.staff_id WHERE 1=1")
        args = []
        if day:
            sql += " AND substr(v.check_in,1,10)=?"
            args.append(day)
        if staff_id:
            sql += " AND v.staff_id=?"
            args.append(staff_id)
        rows = self.q(sql + " ORDER BY v.check_in", args)
        for r in rows:
            r["minutes"] = round((parse(r["check_out"]) - parse(r["check_in"])).total_seconds() / 60, 1) if r["check_out"] else None
        return rows

    def route(self, staff_id, day):
        """One staff member's day: start/end, km, fuel cost, visits, time in shops."""
        pings = self.q("SELECT * FROM pings WHERE staff_id=? AND substr(t,1,10)=? ORDER BY t, id", (staff_id, day))
        km, prev = 0.0, None
        for p in (p for p in pings if p["ok"]):
            if prev:
                d = dist_m(prev["lat"], prev["lon"], p["lat"], p["lon"])
                if d < self.cfg["min_step_m"]:  # standing still: GPS jitter, not distance travelled
                    continue
                km += d / 1000
            prev = p
        visits = self.visits(day, staff_id)
        done = [v["minutes"] for v in visits if v["minutes"] is not None]
        fuel = self.cfg["fuel"]
        return {
            "staff_id": staff_id, "day": day, "pings": len(pings),
            "start": pings[0]["t"] if pings else None, "end": pings[-1]["t"] if pings else None,
            "route_minutes": round((parse(pings[-1]["t"]) - parse(pings[0]["t"])).total_seconds() / 60, 1) if pings else 0,
            "km": round(km, 2),
            "fuel_litres": round(km / fuel["km_per_litre"], 3),
            "fuel_cost_tk": round(km / fuel["km_per_litre"] * fuel["price_per_litre_tk"], 1),
            "visits": len(visits), "shops": len({v["shop_id"] for v in visits}),
            "avg_minutes_in_shop": round(sum(done) / len(done), 1) if done else None,
            "last_check_out": max((v["check_out"] for v in visits if v["check_out"]), default=None),
        }

    def events_since(self, after_id=0):
        return self.q("SELECT e.*, s.name AS shop, st.name AS staff FROM events e LEFT JOIN shops s ON s.id=e.shop_id "
                      "LEFT JOIN staff st ON st.id=e.staff_id WHERE e.id>? ORDER BY e.id", (after_id,))
