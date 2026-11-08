"""Demo data for showing the app: one rider's last hours, played through the real server API.

The rider loads the motorcycle at the factory, rides through the sample Mirpur shops (a GPS point every 30 s at
about 20 km/h, 8-12 minutes in each shop), is checked in and out automatically, delivers cups (a few taken back),
delivers a Pandamart PO a little short, fills a reseller order and films two shelves (real test videos). Times
end now, so the Live and Today screens show it. Everything goes into the sample (demo) database only.

Usage (project root, venv active, server running, office PIN from data/demo_pins.txt):
  python scripts/demo_day.py --pin <office PIN>
"""
import argparse
import http.cookiejar
import json
import math
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ["Shwapno, Mirpur 10", "Rahman General Store", "Mirpur Dairy Corner", "Pandamart Mirpur (sample)",
         "Mirpur Wholesale Traders (sample)", "Al-Madina Store", "Shwapno, Mirpur 11"]
FACTORY = (23.8040, 90.3640)  # sample start point
VIDEOS = {"Shwapno, Mirpur 10": "Test_Videos/VID-20261018-WA0012.mp4", "Al-Madina Store": "Test_Videos/VID-20261018-WA0030.mp4"}


class Client:
    def __init__(self, base, pin):
        self.base = base.rstrip("/")
        self.op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        self.post("/api/login", {"pin": pin})

    def call(self, path, data=None, ctype="application/json"):
        req = urllib.request.Request(self.base + path, data=data, headers={"Content-Type": ctype} if data is not None else {})
        with self.op.open(req, timeout=60) as r:
            return json.loads(r.read())

    def post(self, path, obj):
        return self.call(path, json.dumps(obj).encode())


def step_points(a, b, start, speed_kmh=20, every_s=30):
    """GPS points from a to b at a steady speed; returns (points, arrival time)."""
    d = 2 * 6371000 * math.asin(math.sqrt(math.sin(math.radians(b[0] - a[0]) / 2) ** 2 + math.cos(math.radians(a[0]))
                                          * math.cos(math.radians(b[0])) * math.sin(math.radians(b[1] - a[1]) / 2) ** 2))
    secs = max(d / (speed_kmh / 3.6), every_s)
    n = max(int(secs // every_s), 1)
    pts = [((a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n), start + timedelta(seconds=secs * k / n)) for k in range(1, n + 1)]
    return pts, start + timedelta(seconds=secs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pin", required=True, help="office PIN")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--rider", default="Jamal Hossain")
    ap.add_argument("--hours", type=float, default=2.0, help="how long before now the day starts")
    args = ap.parse_args()
    c = Client(args.url, args.pin)
    shops = {s["name"]: s for s in c.call("/api/shops")}
    staff = next(s for s in c.call("/api/staff") if s["name"] == args.rider)
    products = c.call("/api/products")
    cup, box = products[0]["id"], products[1]["id"] if len(products) > 1 else products[0]["id"]
    route = [shops[n] for n in ROUTE if n in shops]

    # factory: production and loading the motorcycle
    c.post("/api/production", {"product_id": cup, "qty": 600, "note": "demo"})
    c.post("/api/dispatches", {"staff_id": staff["id"], "items": [{"product_id": cup, "qty": 180}, {"product_id": box, "qty": 4}], "note": "demo"})
    panda = shops.get("Pandamart Mirpur (sample)")
    po = c.post("/api/pos", {"number": "PM-DEMO-" + datetime.now().strftime("%H%M"), "shop_id": panda["id"],
                             "items": [{"product_id": cup, "qty": 60}, {"product_id": box, "qty": 2}], "note": "demo"}) if panda else None
    reseller = shops.get("Mirpur Wholesale Traders (sample)")
    order = c.post("/api/orders", {"shop_id": reseller["id"], "items": [{"product_id": box, "qty": 2}], "note": "demo"}) if reseller else None
    if order:
        c.post(f"/api/orders/{order['id']}/status", {"status": "dispatched"})

    t = datetime.now(timezone.utc) - timedelta(hours=args.hours)
    here, pings, events = FACTORY, 0, []

    def ping(ll, when):
        nonlocal pings
        if when > datetime.now(timezone.utc):  # the day is played up to now only
            return
        r = c.post("/api/locations", {"staff_id": staff["id"], "lat": ll[0], "lon": ll[1], "acc": 8, "t": when.isoformat()})
        pings += 1
        names = {s["id"]: s["name"] for s in shops.values()}  # passing close to another shop also checks in there
        events.extend(f"{e['kind']} {names.get(e['shop_id'], e['shop_id'])}" for e in r["events"])

    ping(here, t)
    for i, shop in enumerate(route):
        pts, t = step_points(here, (shop["lat"], shop["lon"]), t)
        for ll, when in pts:
            ping(ll, when)
        stay = 8 + (i * 3) % 5  # 8-12 minutes in the shop
        for k in range(1, stay * 2 + 1):  # standing in the shop: tiny GPS jitter
            ping((shop["lat"] + 0.00002 * math.sin(k), shop["lon"] + 0.00002 * math.cos(k)), t + timedelta(seconds=30 * k))
        when = min(t + timedelta(minutes=3), datetime.now(timezone.utc))
        if shop["channel"] == "pandamart" and po:
            items = [{"product_id": cup, "delivered": 54}, {"product_id": box, "delivered": 2}]  # 6 cups short
            c.post("/api/deliveries", {"staff_id": staff["id"], "shop_id": shop["id"], "po_id": po["id"], "items": items, "t": when.isoformat()})
        elif shop["channel"] == "reseller" and order:
            c.post("/api/deliveries", {"staff_id": staff["id"], "shop_id": shop["id"], "order_id": order["id"],
                                       "items": [{"product_id": box, "delivered": 2}], "t": when.isoformat()})
        else:
            items = [{"product_id": cup, "delivered": 16 + 4 * (i % 3)}]
            if i % 3 == 1:
                items[0].update(returned=2, return_reason="expired")
            c.post("/api/deliveries", {"staff_id": staff["id"], "shop_id": shop["id"], "items": items, "t": when.isoformat()})
        video = VIDEOS.get(shop["name"])
        if video and (ROOT / video).exists():
            q = urllib.parse.urlencode({"shop": shop["name"], "shop_id": shop["id"], "lat": shop["lat"], "lon": shop["lon"], "acc": 8,
                                        "captured": (t + timedelta(minutes=5)).isoformat(), "source": "in_app"})
            c.call("/api/audits?" + q, (ROOT / video).read_bytes(), "video/mp4")
        t += timedelta(minutes=stay)
        here = (shop["lat"], shop["lon"])
    pts, t = step_points(here, FACTORY, t)  # back to the factory: the last visit ends
    for ll, when in pts:
        if when < datetime.now(timezone.utc):
            ping(ll, when)
    route_today = next(r for r in c.call("/api/routes") if r["staff_id"] == staff["id"])
    print(f"{pings} GPS points; events: {', '.join(events)}")
    print(f"route: {route_today['km']} km, fuel {route_today['fuel_cost_tk']} Tk, {route_today['visits']} visits, "
          f"avg {route_today['avg_minutes_in_shop']} min per shop")


if __name__ == "__main__":
    main()
