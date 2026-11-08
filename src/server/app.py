"""Sprint 6: back end for the mobile app. Runs on the demo PC; phones connect over Wi-Fi.

One small server (Python's built-in http.server, nothing to install) that:
  - serves the dashboard (app/mockup/site/) and the evidence images of every audit (data/reports/),
  - receives a shelf video from a phone, saves it in data/uploads/ and runs the real audit pipeline
    (src/report/audit.py: keyframes, detector, integrity checks) in a background worker, one video at a time,
  - stores field staff GPS pings, turns them into shop visits (geofence check-in/out) and daily routes
    (time, km, fuel) and keeps each shop's changeable agreement (src/server/db.py, SQLite),
  - answers "Ask about this audit" questions through the local AI (src/assistant/ask.py), so phones never
    talk to Ollama directly.

API (JSON unless noted):
  POST /api/audits?shop=..&rep=..&lat=..&lon=..&acc=..&captured=..   body = the video file (raw bytes)
       -> {"id", "status": "queued"}
  GET  /api/audits            -> uploads, newest first: id, status, shop, rep, received, verdict
  GET  /api/audits/<id>       -> upload metadata + status + the audit report when done
  POST /api/ask               {"id", "question"} -> {"answer", "seconds", "model"}
  GET  /api/shops             -> shops with their current agreement
  POST /api/shops             {"name", "lat", "lon", "area"?, "channel"?, "radius_m"?} -> {"id"}
  POST /api/shops/<id>/agreement  {"cups", "shelf"?, "zone"?, "daily_fee_tk"?, "note"?, "by"?} -> {"id"}
  GET  /api/staff             POST /api/staff {"name", "phone"?} -> {"id"}
  POST /api/locations         {"staff_id", "lat", "lon", "acc"?} -> {"ok", "events": [check_in / check_out]}
  GET  /api/locations         -> each staff member's last position (newer than server.location_ttl_sec)
  GET  /api/visits?day=YYYY-MM-DD&staff_id=..   GET /api/routes?day=..   GET /api/events?after=<id>

Usage (project root, venv active; Ollama running for /api/ask):
  python -m src.server.app              # http://localhost:8000 on this PC, http://<PC IPv4>:8000 on the Wi-Fi
  python -m src.server.app --seed-demo  # first run: add NutriCoco's sample shops and staff (marked sample)
"""
import argparse
import json
import os
import queue
import re
import secrets
import threading
import time
import urllib.error
from datetime import datetime, timezone
from http.cookies import SimpleCookie
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml

from src.server.business import Business
from src.server.db import DB, dist_m

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"
SITE = ROOT / "app" / "mockup" / "site"
FIELD = ROOT / "app" / "field"
OFFICE = ROOT / "app" / "office"
ORDER = ROOT / "app" / "order"
SESSION_HOURS = 12
MAX_FAILS, LOCK_SEC = 5, 300  # 5 wrong PINs from one address -> wait 5 minutes
UPLOADS = ROOT / "data" / "uploads"
REPORTS = ROOT / "data" / "reports"
VIDEO_TYPES = {"video/mp4": ".mp4", "video/quicktime": ".mov", "video/webm": ".webm", "video/3gpp": ".3gp",
               "application/octet-stream": ".mp4"}


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def client_time(t):
    """A time sent by the phone (a GPS point recorded while offline) as UTC ISO text, or None for 'now'.
    Only the last 24 hours are accepted, and not more than 2 minutes in the future (phone clock drift)."""
    if not t:
        return None
    dt = datetime.fromisoformat(str(t).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("time needs a time zone")
    age = (datetime.now(timezone.utc) - dt).total_seconds()
    if not -120 <= age <= 86400:
        raise ValueError("time is too old or in the future")
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def slug(text, n=30):
    return re.sub(r"[^A-Za-z0-9]+", "-", text or "").strip("-")[:n] or "shop"


def num(qs, key):
    """A float query parameter, or None when missing or not a number."""
    try:
        return float(qs[key][0])
    except (KeyError, ValueError, IndexError):
        return None


class Store:
    """Uploads and their audit reports on disk, the database (shops, staff, GPS, visits) and the audit worker."""

    def __init__(self, cfg, uploads=UPLOADS, reports=REPORTS, run_audit=None, db_path=None):
        self.cfg, self.uploads, self.reports = cfg, Path(uploads), Path(reports)
        self.uploads.mkdir(parents=True, exist_ok=True)
        db_path = Path(db_path or ROOT / cfg["server"]["database"])
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = DB(db_path, cfg["tracking"])
        self.biz = Business(self.db, cfg["business"])
        self.run_audit = run_audit or self._run_pipeline
        self.jobs = queue.Queue()
        self.lock = threading.Lock()
        self.sessions, self.fails = {}, {}  # in memory: a server restart logs everyone out
        threading.Thread(target=self._worker, daemon=True).start()

    def login(self, pin, client):
        """(who, token) for a valid PIN, (None, None) for a wrong one; PermissionError while locked out."""
        now = time.time()
        with self.lock:
            recent = [t for t in self.fails.get(client, []) if now - t < LOCK_SEC]
            self.fails[client] = recent
            if len(recent) >= MAX_FAILS:
                raise PermissionError(f"too many wrong PINs: try again in {int(LOCK_SEC - (now - recent[0])) + 1} s")
        who = self.db.check_pin(pin)
        with self.lock:
            if who is None:
                self.fails[client].append(now)
                return None, None
            self.fails.pop(client, None)
            token = secrets.token_urlsafe(32)
            self.sessions[token] = {**who, "expires": now + SESSION_HOURS * 3600}
        return who, token

    def session(self, token):
        with self.lock:
            s = self.sessions.get(token)
            if s and s["expires"] < time.time():
                del self.sessions[token]
                s = None
        return {k: v for k, v in s.items() if k != "expires"} if s else None

    def logout(self, token):
        with self.lock:
            self.sessions.pop(token, None)

    def meta_path(self, uid):
        return self.uploads / f"{uid}.json"

    def read_meta(self, uid):
        p = self.meta_path(uid)
        with self.lock:  # Windows cannot replace a file while it is open, so reads and writes take turns
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def write_meta(self, meta):
        # write then rename, so a reader never sees a half-written file
        with self.lock:
            tmp = self.uploads / f"{meta['id']}.tmp"
            tmp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
            os.replace(tmp, self.meta_path(meta["id"]))

    def save_upload(self, stream, length, content_type, qs, who=None):
        """Copy the request body to data/uploads/ in 1 MB pieces (never the whole video in memory)."""
        ext = VIDEO_TYPES.get(content_type.split(";")[0].strip().lower())
        if not ext:
            raise ValueError(f"not a video: {content_type}")
        shop = qs.get("shop", [""])[0]
        uid = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3] + "-" + slug(shop)
        video = self.uploads / (uid + ext)
        left = length
        with open(video, "wb") as f:
            while left > 0:
                chunk = stream.read(min(1 << 20, left))
                if not chunk:
                    break
                f.write(chunk)
                left -= len(chunk)
        if left > 0:
            video.unlink(missing_ok=True)
            raise ValueError("upload ended early")
        meta = {"id": uid, "video": video.name, "bytes": length, "status": "queued", "received_utc": now_utc(),
                "shop": shop, "rep": (who or {}).get("name") or qs.get("rep", [""])[0],
                "staff_id": (who or {}).get("staff_id"), "shop_id": int(num(qs, "shop_id")) if num(qs, "shop_id") else None,
                "captured_utc": qs.get("captured", [""])[0],
                "source": "in_app" if qs.get("source", [""])[0] == "in_app" else "file",
                "gps": {"lat": num(qs, "lat"), "lon": num(qs, "lon"), "accuracy_m": num(qs, "acc")}}
        self.write_meta(meta)
        self.jobs.put(uid)
        return meta

    def _run_pipeline(self, video):
        from src.report.audit import audit
        return audit(video, ROOT / self.cfg["detection"]["weights"], self.cfg, self.cfg["server"]["device"],
                     self.reports)[0]

    def _worker(self):
        while True:
            uid = self.jobs.get()
            meta = self.read_meta(uid)
            meta["status"] = "processing"
            self.write_meta(meta)
            try:
                report = self.run_audit(self.uploads / meta["video"])
                # the capture details from the phone go into the report, next to the pipeline's own checks
                report["capture"] = {k: meta.get(k) for k in ("shop", "shop_id", "rep", "captured_utc", "received_utc",
                                                              "source", "gps")}
                self.location_check(report, meta)
                self.age_check(report, meta)
                flags = report["integrity"]["flags"]
                report["integrity"]["verdict"] = ("re-audit" if any(f["severity"] == "critical" for f in flags)
                                                  else "review" if flags else "ok")
                out = self.reports / Path(meta["video"]).stem / "audit.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(report, indent=1), encoding="utf-8")
                meta.update(status="done", verdict=report["integrity"]["verdict"],
                            processing_sec=report.get("processing_sec"))
            except Exception as e:  # a bad file must not stop the worker
                meta.update(status="failed", error=f"{type(e).__name__}: {e}")
            self.write_meta(meta)

    def location_check(self, report, meta):
        """Was the video filmed at the shop it is for? Adds the distance and, when far or unknown, a flag."""
        t, shop, gps = self.cfg["tracking"], self.db.shop(meta.get("shop_id")) if meta.get("shop_id") else None, meta["gps"]
        if not shop:
            return
        flags = report["integrity"]["flags"]
        if gps.get("lat") is None:
            flags.append({"check": "location", "severity": "warning", "detail": "the video was sent without a GPS position"})
        else:
            d = round(dist_m(gps["lat"], gps["lon"], shop["lat"], shop["lon"]))
            report["capture"]["distance_to_shop_m"] = d
            if d > t["video_far_m"]:
                flags.append({"check": "location", "severity": "critical",
                              "detail": f"filmed {d / 1000:.1f} km from {shop['name']}: probably another place"})
            elif d > t["video_max_distance_m"]:
                flags.append({"check": "location", "severity": "warning", "detail": f"filmed {d} m from {shop['name']}"})

    def age_check(self, report, meta):
        """A video chosen from the phone's files may be old: warn when it was recorded long before it was sent.
        (Videos recorded live in the app are sent right away, or queued with their recording time.)"""
        try:
            captured = datetime.fromisoformat(meta["captured_utc"].replace("Z", "+00:00"))
            hours = (datetime.fromisoformat(meta["received_utc"]) - captured).total_seconds() / 3600
        except (KeyError, ValueError, AttributeError, TypeError):
            return
        report["capture"]["hours_before_upload"] = round(hours, 1)
        if hours > self.cfg["tracking"]["video_max_age_hours"]:
            report["integrity"]["flags"].append({"check": "recording_time", "severity": "warning",
                                                 "detail": f"recorded {hours:.0f} h before it was sent"})

    def report(self, uid):
        meta = self.read_meta(uid)
        if not meta or meta["status"] != "done":
            return None
        return json.loads((self.reports / Path(meta["video"]).stem / "audit.json").read_text(encoding="utf-8"))

    def list(self):
        with self.lock:
            metas = [json.loads(p.read_text(encoding="utf-8")) for p in self.uploads.glob("*.json")]
        return sorted(metas, key=lambda m: m["received_utc"] + m["id"], reverse=True)

    def seed_demo(self):
        """NutriCoco's sample shops (Mirpur 10/11, positions approximate) and staff, only into an empty database."""
        if self.db.shops():
            return False
        shops = [("Shwapno, Mirpur 10", "Mirpur 10", "supermarket", 23.8069, 90.3687),
                 ("Al-Madina Store", "Mirpur 11", "retail", 23.8212, 90.3661),
                 ("Rahman General Store", "Mirpur 10", "retail", 23.8051, 90.3712),
                 ("Bhai Bhai Store", "Mirpur 11", "retail", 23.8170, 90.3630),
                 ("Shwapno, Mirpur 11", "Mirpur 11", "supermarket", 23.8189, 90.3655),
                 ("Mirpur Dairy Corner", "Mirpur 10", "retail", 23.8090, 90.3664)]
        for name, area, channel, lat, lon in shops:
            sid = self.db.add_shop(name, lat, lon, area, channel, sample=True)
            # placeholder agreement until NutriCoco sets the real one (cups and position are editable)
            self.db.set_agreement(sid, 4, shelf=2, zone="front", daily_fee_tk=135, note="sample", by="seed",
                                  valid_from="2026-11-01T00:00:00+00:00")
        pins = {"NutriCoco office": self.db.new_pin("office")}
        for name in ("Rahim Uddin", "Jamal Hossain", "Tania Akter", "Sumon Mia"):
            pins[name] = self.db.new_pin("staff", self.db.add_staff(name))
        self.seed_business(pins)
        return pins

    def seed_business(self, pins):
        """Sample products, a Pandamart outlet and a reseller with a PIN (prices are placeholders, marked sample)."""
        if self.biz.products(all_=True):
            return
        self.biz.add_product("Coco Fresh Coconut Pudding 150 g", "cup", 40, 50, 45, sample=True)
        self.biz.add_product("Coco Fresh carton (24 cups)", "box", 900, None, 1050, sample=True)
        self.db.add_shop("Pandamart Mirpur (sample)", 23.8102, 90.3700, "Mirpur 10", "pandamart", sample=True)
        reseller = self.db.add_shop("Mirpur Wholesale Traders (sample)", 23.8150, 90.3620, "Mirpur 11", "reseller", sample=True)
        pins["Mirpur Wholesale Traders (reseller)"] = self.db.new_pin("reseller", shop_id=reseller)


def make_handler(store, site=SITE):
    class Handler(SimpleHTTPRequestHandler):
        extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".webmanifest": "application/manifest+json",
                          ".js": "text/javascript"}

        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(site), **kw)

        def log_message(self, fmt, *args):  # one short line per API call, no noise for static files
            if self.path.startswith("/api/"):
                print(f"{self.log_date_time_string()} {self.command} {self.path.split('?')[0]} {args[1] if len(args) > 1 else ''}")

        def translate_path(self, path):
            # /reports/... -> data/reports/ (evidence, login needed), /field/... -> the phone app, else the dashboard.
            # SimpleHTTPRequestHandler drops ".." parts, so no path can leave its folder.
            if path.startswith("/track/"):  # every share link shows the same page; its script reads the token
                return str(OFFICE / "track.html")
            for prefix, folder in (("/reports/", store.reports), ("/field/", FIELD), ("/office/", OFFICE), ("/order/", ORDER)):
                if path.startswith(prefix):
                    base, self.directory = self.directory, str(folder)
                    try:
                        return super().translate_path(path[len(prefix) - 1:])
                    finally:
                        self.directory = base
            return super().translate_path(path)

        def session(self):
            cookie = SimpleCookie(self.headers.get("Cookie", ""))
            return store.session(cookie["ss"].value) if "ss" in cookie else None

        def client(self):  # behind the Cloudflare tunnel every request comes from 127.0.0.1
            return self.headers.get("CF-Connecting-IP") or self.client_address[0]

        def allowed(self, method, path, who):
            """Office: everything. Staff: their own GPS, uploads, audits, questions and deliveries, the shop list,
            products, open orders/POs and their own motorcycle stock. Resellers: products and their own orders."""
            if who is None:
                return False
            if who["role"] == "office":
                return True
            if who["role"] == "reseller":
                return (method, path) in {("GET", "/api/products"), ("GET", "/api/orders"), ("POST", "/api/orders")}
            return ((method, path) in {("GET", "/api/shops"), ("POST", "/api/locations"), ("POST", "/api/audits"),
                                       ("GET", "/api/audits"), ("POST", "/api/ask"), ("GET", "/api/products"),
                                       ("GET", "/api/deliveries"), ("POST", "/api/deliveries"), ("GET", "/api/orders"),
                                       ("GET", "/api/pos"), ("GET", "/api/my-stock")}
                    or (method == "GET" and (path.startswith("/api/audits/") or path.startswith("/reports/"))))

        def deny(self, who):
            return self.send_json({"error": "log in first" if who is None else "not allowed for your role"},
                                  401 if who is None else 403)

        def auth_routes(self, method, path):
            """/api/login, /api/logout, /api/me. Returns True when the request was handled."""
            if (method, path) == ("POST", "/api/login"):
                try:
                    who, token = store.login(str(self.read_json().get("pin", "")), self.client())
                except PermissionError as e:
                    self.send_json({"error": str(e)}, 429)
                    return True
                if who is None:
                    self.send_json({"error": "wrong PIN"}, 401)
                    return True
                secure = "; Secure" if self.headers.get("X-Forwarded-Proto") == "https" else ""
                body = json.dumps(who).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Set-Cookie", f"ss={token}; Path=/; HttpOnly; SameSite=Strict; "
                                               f"Max-Age={SESSION_HOURS * 3600}{secure}")
                self.end_headers()
                self.wfile.write(body)
                return True
            if (method, path) == ("POST", "/api/logout"):
                cookie = SimpleCookie(self.headers.get("Cookie", ""))
                if "ss" in cookie:
                    store.logout(cookie["ss"].value)
                self.send_json({"ok": True})
                return True
            if (method, path) == ("GET", "/api/me"):
                who = self.session()
                if who and who["role"] == "staff":
                    visit = store.db.open_visit(who["staff_id"])
                    who = {**who, "visit_shop_id": visit["shop_id"] if visit else None}
                self.send_json(who if who else {"error": "log in first"}, 200 if who else 401)
                return True
            return False

        def send_json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def read_json(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n > 100_000:
                raise ValueError("request too large")
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            url = urlparse(self.path)
            if self.auth_routes("GET", url.path):
                return
            if url.path in ("/office", "/field", "/order"):  # a typed address without the final slash
                self.send_response(301)
                self.send_header("Location", url.path + "/")
                self.end_headers()
                return
            if url.path.startswith("/api/track/"):  # public: a share link needs no login, only a valid token
                t = store.db.tracked(url.path.rsplit("/", 1)[1])
                return self.send_json(t if t else {"error": "this link has expired or is not valid"}, 200 if t else 404)
            if url.path.startswith("/api/") or url.path.startswith("/reports/"):
                who = self.session()
                if not self.allowed("GET", url.path, who):
                    return self.deny(who)
                if url.path.startswith("/reports/"):
                    return super().do_GET()
            own = who["staff_id"] if url.path.startswith("/api/") and who["role"] == "staff" else None
            if url.path == "/api/audits":
                return self.send_json([m for m in store.list() if own is None or m.get("staff_id") == own])
            if url.path.startswith("/api/audits/"):
                meta = store.read_meta(url.path.rsplit("/", 1)[1])
                if not meta or (own is not None and meta.get("staff_id") != own):
                    return self.send_json({"error": "unknown audit"}, 404)
                return self.send_json({**meta, "report": store.report(meta["id"])})
            qs = {k: v[0] for k, v in parse_qs(url.query).items()}
            db = store.db
            if url.path == "/api/locations":
                return self.send_json(db.latest_positions(store.cfg["server"]["location_ttl_sec"]))
            if url.path == "/api/shops":
                return self.send_json(db.shops())
            if url.path == "/api/staff":
                return self.send_json(db.staff())
            if url.path == "/api/visits":
                return self.send_json(db.visits(qs.get("day"), qs.get("staff_id")))
            if url.path == "/api/routes":
                day = qs.get("day") or db.today()
                return self.send_json([{**db.route(s["id"], day), "name": s["name"]} for s in db.staff()])
            if url.path == "/api/events":
                return self.send_json(db.events_since(int(qs.get("after") or 0)))
            m = re.fullmatch(r"/api/shops/(\d+)/agreements", url.path)
            if m:
                return self.send_json(db.agreements(int(m.group(1))))
            if url.path == "/api/settings":
                return self.send_json({"fuel": db.fuel()})
            if url.path == "/api/shares":
                return self.send_json(db.shares())
            biz, role = store.biz, who["role"] if url.path.startswith("/api/") else None
            if url.path == "/api/products":
                return self.send_json(biz.products(all_=role == "office" and qs.get("all") == "1"))
            if url.path == "/api/dispatches":
                return self.send_json(biz.dispatches(qs.get("day")))
            if url.path == "/api/deliveries":
                staff = who["staff_id"] if role == "staff" else (int(qs["staff_id"]) if qs.get("staff_id") else None)
                return self.send_json(biz.deliveries(qs.get("day"), int(qs["shop_id"]) if qs.get("shop_id") else None, staff))
            if url.path == "/api/pos":
                return self.send_json(biz.pos("open" if role == "staff" else qs.get("status")))
            if url.path == "/api/orders":
                if role == "reseller":
                    return self.send_json(biz.orders(who["shop_id"]))
                rows = biz.orders(status=qs.get("status"))
                return self.send_json([o for o in rows if o["status"] in ("confirmed", "dispatched")] if role == "staff" else rows)
            if url.path == "/api/stock":
                return self.send_json(biz.stock(qs.get("day")))
            if url.path == "/api/my-stock":
                mine = [r for r in biz.stock()["riders"] if r["staff_id"] == who["staff_id"]]
                return self.send_json(mine[0] if mine else {"items": []})
            if url.path == "/api/restock":
                return self.send_json(biz.restock())
            if url.path.startswith("/api/"):
                return self.send_json({"error": "not found"}, 404)
            return super().do_GET()

        def do_POST(self):
            url = urlparse(self.path)
            try:
                if self.auth_routes("POST", url.path):
                    return
                who = self.session()
                if not self.allowed("POST", url.path, who):
                    return self.deny(who)
                if url.path == "/api/audits":
                    length = int(self.headers.get("Content-Length") or 0)
                    if length <= 0:
                        return self.send_json({"error": "empty upload"}, 400)
                    if length > store.cfg["server"]["max_upload_mb"] * 1_000_000:
                        return self.send_json({"error": "video too large"}, 413)
                    meta = store.save_upload(self.rfile, length, self.headers.get("Content-Type", ""),
                                             parse_qs(url.query), who)
                    return self.send_json({"id": meta["id"], "status": meta["status"]}, 202)
                db = store.db
                if url.path == "/api/locations":
                    d = self.read_json()
                    lat, lon = float(d["lat"]), float(d["lon"])
                    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                        raise ValueError("bad coordinates")
                    acc = float(d["acc"]) if d.get("acc") is not None else None
                    # staff always ping as themselves; the office may send a ping for anyone (testing)
                    if who["role"] != "staff" and "staff_id" not in d:
                        raise ValueError("tracking is for riders: log in with a rider's PIN")
                    staff_id = who["staff_id"] if who["role"] == "staff" else int(d["staff_id"])
                    events = db.ping(staff_id, lat, lon, acc, client_time(d.get("t")))
                    visit = db.open_visit(staff_id)  # the phone also learns a visit that started before it opened
                    return self.send_json({"ok": True, "events": events, "visit_shop_id": visit["shop_id"] if visit else None})
                if url.path == "/api/shops":
                    d = self.read_json()
                    sid = db.add_shop(str(d["name"])[:100], float(d["lat"]), float(d["lon"]), str(d.get("area", ""))[:60],
                                      str(d.get("channel", "retail"))[:30], float(d["radius_m"]) if d.get("radius_m") else None)
                    return self.send_json({"id": sid}, 201)
                m = re.fullmatch(r"/api/shops/(\d+)/agreement", url.path)
                if m:
                    d = self.read_json()
                    aid = db.set_agreement(int(m.group(1)), int(d["cups"]), d.get("shelf"), d.get("zone"),
                                           d.get("daily_fee_tk"), str(d.get("note", ""))[:300], str(d.get("by", ""))[:60])
                    return self.send_json({"id": aid}, 201)
                if url.path == "/api/staff":
                    d = self.read_json()
                    return self.send_json({"id": db.add_staff(str(d["name"])[:60], str(d.get("phone", ""))[:20])}, 201)
                if url.path == "/api/settings":
                    d = self.read_json().get("fuel", {})
                    return self.send_json({"fuel": db.set_fuel(d.get("km_per_litre"), d.get("price_per_litre_tk"))})
                if url.path == "/api/shares":
                    d = self.read_json()
                    return self.send_json(db.new_share(int(d["staff_id"]), float(d.get("hours", 10))), 201)
                if url.path == "/api/shares/end":
                    db.end_share(str(self.read_json().get("token", "")))
                    return self.send_json({"ok": True})
                biz, by = store.biz, who["name"]
                text = lambda d, k, n=300: str(d.get(k) or "")[:n]  # noqa: E731
                if url.path == "/api/products":
                    d = self.read_json()
                    prices = {k: float(d[k]) if d.get(k) not in (None, "") else None for k in ("price_wholesale", "price_retail", "price_pandamart")}
                    return self.send_json({"id": biz.add_product(text(d, "name", 100), text(d, "unit", 20) or "cup", **prices)}, 201)
                m = re.fullmatch(r"/api/products/(\d+)", url.path)
                if m:
                    d = self.read_json()
                    fields = {k: (float(v) if k.startswith("price") and v not in (None, "") else None if k.startswith("price") else v)
                              for k, v in d.items() if k in ("name", "unit", "price_wholesale", "price_retail", "price_pandamart", "active")}
                    biz.update_product(int(m.group(1)), **fields)
                    return self.send_json({"ok": True})
                if url.path == "/api/production":
                    d = self.read_json()
                    return self.send_json({"id": biz.add_production(int(d["product_id"]), int(d["qty"]), text(d, "note"), by)}, 201)
                if url.path == "/api/dispatches":
                    d = self.read_json()
                    return self.send_json({"id": biz.add_dispatch(int(d["staff_id"]), d.get("items"), text(d, "note"), by)}, 201)
                if url.path == "/api/deliveries":
                    d = self.read_json()
                    staff = who["staff_id"] if who["role"] == "staff" else (int(d["staff_id"]) if d.get("staff_id") else None)
                    did = biz.add_delivery(int(d["shop_id"]), d.get("items"), staff, int(d["po_id"]) if d.get("po_id") else None,
                                           int(d["order_id"]) if d.get("order_id") else None, text(d, "note"),
                                           client_time(d.get("t")))  # a delivery saved offline keeps its time
                    return self.send_json({"id": did}, 201)
                if url.path == "/api/pos":
                    d = self.read_json()
                    return self.send_json({"id": biz.add_po(text(d, "number", 40), int(d["shop_id"]), d.get("items"), d.get("po_date") or None,
                                                            d.get("due_date") or None, text(d, "note"), by)}, 201)
                m = re.fullmatch(r"/api/pos/(\d+)/status", url.path)
                if m:
                    biz.close_po(int(m.group(1)), str(self.read_json().get("status", "closed")))
                    return self.send_json({"ok": True})
                if url.path == "/api/orders":
                    d = self.read_json()
                    shop = who["shop_id"] if who["role"] == "reseller" else int(d["shop_id"])
                    return self.send_json({"id": biz.add_order(shop, d.get("items"), d.get("wanted_date") or None, text(d, "note"), by)}, 201)
                m = re.fullmatch(r"/api/orders/(\d+)/status", url.path)
                if m:
                    biz.set_order_status(int(m.group(1)), str(self.read_json().get("status")))
                    return self.send_json({"ok": True})
                m = re.fullmatch(r"/api/customers/(\d+)/pin", url.path)
                if m:  # a new reseller PIN, shown once to the office
                    return self.send_json({"pin": db.new_pin("reseller", shop_id=int(m.group(1)))}, 201)
                if url.path == "/api/ask":
                    d = self.read_json()
                    meta = store.read_meta(str(d.get("id", "")))
                    mine = meta and (who["role"] == "office" or meta.get("staff_id") == who["staff_id"])
                    report = store.report(meta["id"]) if mine else None
                    if report is None:
                        return self.send_json({"error": "audit not found or not finished"}, 404)
                    from src.assistant.ask import ask
                    cfg = store.cfg["assistant"]
                    try:
                        answer, sec = ask(str(d.get("question", ""))[:300], report, cfg)
                    except (urllib.error.URLError, TimeoutError, OSError):
                        return self.send_json({"error": "the AI assistant is not running on the PC"}, 503)
                    return self.send_json({"answer": answer, "seconds": sec, "model": cfg["model"]})
                return self.send_json({"error": "not found"}, 404)
            except (ValueError, KeyError, TypeError) as e:
                return self.send_json({"error": str(e)}, 400)

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--port", type=int, help="default: server.port in the config")
    ap.add_argument("--seed-demo", action="store_true", help="add sample shops, staff and PINs to an empty database")
    ap.add_argument("--new-pin", metavar="office|STAFF_ID", help="print a new PIN for the office or one staff member, then exit")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    port = args.port or cfg["server"]["port"]
    if args.new_pin:
        db = Store(cfg, run_audit=lambda v: None).db
        who = "office" if args.new_pin == "office" else int(args.new_pin)
        print("New PIN:", db.new_pin("office") if who == "office" else db.new_pin("staff", who))
        return
    # load torch + the detector code once, in the main thread, before any upload arrives (a first import inside
    # the worker thread was blocked by Windows Application Control on the demo PC, and it is faster this way)
    print("Loading the detector...")
    import src.report.audit  # noqa: F401
    import ultralytics  # noqa: F401
    store = Store(cfg)
    if args.seed_demo:
        pins = store.seed_demo()
        if pins:
            text = "ShelfSense demo PINs (keep private; renew with --new-pin)\n" + "".join(f"{k}: {v}\n" for k, v in pins.items())
            (ROOT / "data" / "demo_pins.txt").write_text(text, encoding="utf-8")
            print("Added sample shops and staff. " + text + "Saved in data/demo_pins.txt")
        else:
            print("Database already has shops: nothing added.")
    new = {}
    if store.db.shops() and not store.biz.products(all_=True):  # a demo database from before the business records
        store.seed_business(new)
        with open(ROOT / "data" / "demo_pins.txt", "a", encoding="utf-8") as f:
            f.writelines(f"{k}: {v}\n" for k, v in new.items())
        print("Added sample products, a Pandamart outlet and a reseller; reseller PIN saved in data/demo_pins.txt")
    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(store))
    print(f"ShelfSense server: http://localhost:{port}  (phones on the same Wi-Fi: http://<this PC's IPv4>:{port})")
    print(f"Uploads -> {UPLOADS}, reports -> {REPORTS}. Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
