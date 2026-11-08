"""Tests for src/server/app.py with a fake audit (no detector, no Ollama needed).

Run from the project root:  python -m pytest -v
"""
import http.cookiejar
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from src.server import app

CFG = {"server": {"max_upload_mb": 1, "location_ttl_sec": 600, "device": "cpu"},
       "tracking": {"utc_offset_hours": 6, "geofence_radius_m": 80, "checkout_factor": 1.5, "gps_max_accuracy_m": 50, "min_step_m": 10,
                    "max_speed_kmh": 90, "video_max_distance_m": 200, "video_far_m": 1000, "video_max_age_hours": 6,
                    "fuel": {"km_per_litre": 40, "price_per_litre_tk": 130}},
       "business": {"restock_days": 2},
       "assistant": {"url": "http://localhost:1", "model": "m", "num_ctx": 512, "timeout_sec": 1}}


def fake_audit(video):
    if video.read_bytes() == b"bad":
        raise RuntimeError("cannot read video")
    return {"video": {"file": video.name}, "integrity": {"verdict": "ok", "flags": []}, "summary": {"keyframes": 3},
            "kpis": {}, "keyframes": [], "processing_sec": 0.1}


@pytest.fixture
def server(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("dashboard")
    (tmp_path / "reports" / "old").mkdir(parents=True)
    (tmp_path / "reports" / "old" / "e.jpg").write_bytes(b"jpg")
    store = app.Store(CFG, tmp_path / "uploads", tmp_path / "reports", run_audit=fake_audit, db_path=tmp_path / "t.db")
    PINS.clear()
    PINS.update(store.seed_demo())
    srv = ThreadingHTTPServer(("127.0.0.1", 0), app.make_handler(store, site))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    OPENER["office"] = login(base, PINS["NutriCoco office"])[1]  # call() uses the office login by default
    yield base, tmp_path
    srv.shutdown()


PINS, OPENER = {}, {}


def login(base, pin):
    """(status, opener that keeps the session cookie)."""
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    code, _ = call(f"{base}/api/login", json.dumps({"pin": pin}).encode(), op=op)
    return code, op


def call(url, data=None, ctype="application/json", op=None):
    op = op or OPENER.get("office") or urllib.request.build_opener()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": ctype} if data is not None else {})
    try:
        with op.open(req, timeout=5) as r:
            body = r.read()
            return r.status, json.loads(body) if r.headers.get_content_type() == "application/json" else body
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def wait_done(base, uid):
    for _ in range(50):
        _, meta = call(f"{base}/api/audits/{uid}")
        if meta["status"] in ("done", "failed"):
            return meta
        time.sleep(0.05)
    raise AssertionError("audit did not finish")


def test_upload_runs_audit_and_keeps_capture_details(server):
    base, tmp = server
    code, r = call(f"{base}/api/audits?shop=Shwapno%20Mirpur%2010&rep=Rahim&lat=23.8&lon=90.36&acc=12", b"video", "video/mp4")
    assert code == 202 and r["status"] == "queued"
    meta = wait_done(base, r["id"])
    assert meta["status"] == "done" and meta["verdict"] == "ok" and meta["shop"] == "Shwapno Mirpur 10"
    assert meta["report"]["capture"]["gps"] == {"lat": 23.8, "lon": 90.36, "accuracy_m": 12.0}
    assert (tmp / "uploads" / meta["video"]).read_bytes() == b"video"
    _, listing = call(f"{base}/api/audits")
    assert [m["id"] for m in listing] == [r["id"]]


def test_bad_video_fails_without_stopping_the_worker(server):
    base, _ = server
    _, bad = call(f"{base}/api/audits?shop=a", b"bad", "video/mp4")
    _, good = call(f"{base}/api/audits?shop=b", b"good", "video/mp4")
    assert wait_done(base, bad["id"])["status"] == "failed"
    assert wait_done(base, good["id"])["status"] == "done"


def test_upload_rejects_non_video_and_too_large(server):
    base, _ = server
    assert call(f"{base}/api/audits", b"x", "text/plain")[0] == 400
    try:  # the server answers 413 and closes before the body is sent, so the client may see a reset instead
        assert call(f"{base}/api/audits", b"x" * 1_000_001, "video/mp4")[0] == 413
    except ConnectionError:
        pass
    assert not list((base_tmp := server[1] / "uploads").glob("*.mp4")), base_tmp


def post(url, obj, op=None):
    return call(url, json.dumps(obj).encode(), op=op)


def test_login_required_and_roles(server):
    base, _ = server
    anon = urllib.request.build_opener()
    assert call(f"{base}/api/shops", op=anon)[0] == 401
    assert call(f"{base}/reports/old/e.jpg", op=anon)[0] == 401
    assert call(f"{base}/index.html", op=anon)[0] == 200          # the public dashboard mockup needs no login
    assert login(base, "000000")[0] == 401 or PINS["NutriCoco office"] == "000000"
    code, rahim = login(base, PINS["Rahim Uddin"])
    assert code == 200 and call(f"{base}/api/me", op=rahim)[1]["name"] == "Rahim Uddin"
    assert call(f"{base}/api/shops", op=rahim)[0] == 200          # staff may read shops...
    assert call(f"{base}/api/routes", op=rahim)[0] == 403         # ...but not office reports
    assert post(f"{base}/api/shops/1/agreement", {"cups": 9}, op=rahim)[0] == 403
    # staff always ping as themselves, whatever staff_id they send
    _, r = post(f"{base}/api/locations", {"staff_id": 999, "lat": 23.8069, "lon": 90.3687, "acc": 5}, op=rahim)
    assert r["events"][0]["kind"] == "check_in"
    _, locs = call(f"{base}/api/locations")
    assert [p["name"] for p in locs] == ["Rahim Uddin"]
    # staff see only their own uploads
    _, up = call(f"{base}/api/audits?shop=a", b"video", "video/mp4", op=rahim)
    wait_done(base, up["id"])
    _, office_up = call(f"{base}/api/audits?shop=b", b"video", "video/mp4")
    assert [m["id"] for m in call(f"{base}/api/audits", op=rahim)[1]] == [up["id"]]
    assert call(f"{base}/api/audits/{office_up['id']}", op=rahim)[0] == 404
    assert call(f"{base}/api/audits/{up['id']}", op=rahim)[1]["rep"] == "Rahim Uddin"
    post(f"{base}/api/logout", {}, op=rahim)
    assert call(f"{base}/api/me", op=rahim)[0] == 401


def test_video_location_check(server):
    base, _ = server
    _, shops = call(f"{base}/api/shops")
    s = shops[0]
    cases = {"near": (s["lat"] + 0.0005, s["lon"]), "far": (s["lat"] + 0.003, s["lon"]), "elsewhere": (s["lat"] + 0.05, s["lon"]), "none": None}
    results = {}
    for name, ll in cases.items():
        q = f"shop={name}&shop_id={s['id']}" + (f"&lat={ll[0]}&lon={ll[1]}&acc=10" if ll else "")
        _, up = call(f"{base}/api/audits?{q}", b"video", "video/mp4")
        meta = wait_done(base, up["id"])
        results[name] = (meta["report"]["integrity"]["verdict"], meta["report"]["capture"].get("distance_to_shop_m"))
    assert results["near"][0] == "ok" and 50 < results["near"][1] < 60          # 56 m: fine
    assert results["far"][0] == "review" and 300 < results["far"][1] < 350       # 334 m: warning
    assert results["elsewhere"][0] == "re-audit" and results["elsewhere"][1] > 5000
    assert results["none"] == ("review", None)                                   # no GPS sent


def test_share_link_is_public_but_only_with_a_valid_token(server):
    base, _ = server
    _, staff = call(f"{base}/api/staff")
    code, s = post(f"{base}/api/shares", {"staff_id": staff[0]["id"], "hours": 2})
    assert code == 201
    anon = urllib.request.build_opener()
    code, shown = call(f"{base}/api/track/{s['token']}", op=anon)
    assert code == 200 and shown["name"] == staff[0]["name"] and shown["position"] is None
    assert call(f"{base}/api/track/not-a-token", op=anon)[0] == 404
    assert call(f"{base}/api/shares", op=anon)[0] == 401                         # listing links needs the office
    page = urllib.request.urlopen(f"{base}/track/{s['token']}", timeout=5)
    assert page.status == 200 and b"/api/track/" in page.read()
    post(f"{base}/api/shares/end", {"token": s["token"]})
    assert call(f"{base}/api/track/{s['token']}", op=anon)[0] == 404


def test_settings_and_agreement_history(server):
    base, _ = server
    assert call(f"{base}/api/settings")[1]["fuel"]["km_per_litre"] == 40
    assert post(f"{base}/api/settings", {"fuel": {"km_per_litre": 50}})[1]["fuel"]["km_per_litre"] == 50.0
    assert post(f"{base}/api/settings", {"fuel": {"km_per_litre": -1}})[0] == 400
    post(f"{base}/api/shops/1/agreement", {"cups": 6, "by": "office"})
    hist = call(f"{base}/api/shops/1/agreements")[1]
    assert [a["cups"] for a in hist] == [6, 4]
    _, rahim = login(base, PINS["Rahim Uddin"])
    assert call(f"{base}/api/settings", op=rahim)[0] == 403
    code, body = post(f"{base}/api/locations", {"lat": 23.8, "lon": 90.36})   # the office has no route to track
    assert code == 400 and "rider" in body["error"]


def test_offline_points_and_old_videos(server):
    base, _ = server
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    _, rahim = login(base, PINS["Rahim Uddin"])
    _, shops = call(f"{base}/api/shops", op=rahim)
    s = shops[0]
    # a GPS point recorded 20 min ago while offline, sent now: the visit starts at the recorded time
    then = (now - timedelta(minutes=20)).isoformat(timespec="seconds")
    _, r = post(f"{base}/api/locations", {"lat": s["lat"], "lon": s["lon"], "acc": 5, "t": then}, op=rahim)
    assert r["events"][0]["kind"] == "check_in" and r["events"][0]["t"] == then
    old = (now - timedelta(days=2)).isoformat()
    assert post(f"{base}/api/locations", {"lat": s["lat"], "lon": s["lon"], "t": old}, op=rahim)[0] == 400
    assert post(f"{base}/api/locations", {"lat": s["lat"], "lon": s["lon"], "t": "2026-11-09T10:00:00"}, op=rahim)[0] == 400
    # a video recorded 10 h before it was sent gets a warning; one recorded in the app just now does not
    near = f"shop_id={s['id']}&lat={s['lat']}&lon={s['lon']}&acc=5"
    q_old = urllib.parse.urlencode({"captured": (now - timedelta(hours=10)).isoformat()})
    _, up = call(f"{base}/api/audits?{near}&source=file&{q_old}", b"video", "video/mp4", op=rahim)
    meta = wait_done(base, up["id"])
    assert meta["report"]["integrity"]["verdict"] == "review" and meta["source"] == "file"
    assert [f["check"] for f in meta["report"]["integrity"]["flags"]] == ["recording_time"]
    q_new = urllib.parse.urlencode({"captured": now.isoformat()})
    _, up = call(f"{base}/api/audits?{near}&source=in_app&{q_new}", b"video", "video/webm", op=rahim)
    meta = wait_done(base, up["id"])
    assert meta["report"]["integrity"]["verdict"] == "ok" and meta["report"]["capture"]["source"] == "in_app"
    assert meta["video"].endswith(".webm")


def test_business_roles(server):
    base, _ = server
    _, products = call(f"{base}/api/products")
    cup = products[0]["id"]
    _, shops = call(f"{base}/api/shops")
    ids = {s["name"]: s["id"] for s in shops}
    # the reseller orders for itself, whatever shop_id it sends, at the wholesale price
    _, res = login(base, PINS["Mirpur Wholesale Traders (reseller)"])
    assert call(f"{base}/api/shops", op=res)[0] == 403 and call(f"{base}/api/stock", op=res)[0] == 403
    code, o = post(f"{base}/api/orders", {"shop_id": ids["Al-Madina Store"], "items": [{"product_id": cup, "qty": 100}]}, op=res)
    assert code == 201
    mine = call(f"{base}/api/orders", op=res)[1]
    assert [(x["shop"], x["total_tk"]) for x in mine] == [("Mirpur Wholesale Traders (sample)", 4000.0)]
    assert post(f"{base}/api/orders/{o['id']}/status", {"status": "confirmed"}, op=res)[0] == 403
    assert post(f"{base}/api/orders/{o['id']}/status", {"status": "confirmed"})[0] == 200
    # the office dispatches to Rahim; Rahim delivers (recorded as Rahim) and sees his motorcycle stock
    _, staff = call(f"{base}/api/staff")
    rahim_id = next(s["id"] for s in staff if s["name"] == "Rahim Uddin")
    assert post(f"{base}/api/dispatches", {"staff_id": rahim_id, "items": [{"product_id": cup, "qty": 150}]})[0] == 201
    _, rider = login(base, PINS["Rahim Uddin"])
    assert [x["id"] for x in call(f"{base}/api/orders", op=rider)[1]] == [o["id"]]          # confirmed: to be delivered
    code, _ = post(f"{base}/api/deliveries", {"shop_id": ids["Mirpur Wholesale Traders (sample)"], "order_id": o["id"], "staff_id": 999,
                                              "items": [{"product_id": cup, "delivered": 100}]}, op=rider)
    assert code == 201
    assert call(f"{base}/api/deliveries", op=rider)[1][0]["staff"] == "Rahim Uddin"
    assert call(f"{base}/api/my-stock", op=rider)[1]["items"][0]["on_bike"] == 50
    assert call(f"{base}/api/orders", op=res)[1][0]["status"] == "delivered"
    assert post(f"{base}/api/dispatches", {"staff_id": rahim_id, "items": []}, op=rider)[0] == 403
    assert post(f"{base}/api/deliveries", {"shop_id": ids["Al-Madina Store"], "items": []}, op=rider)[0] == 400
    # a new reseller PIN from the office
    code, r = post(f"{base}/api/customers/{ids['Mirpur Wholesale Traders (sample)']}/pin", {})
    assert code == 201 and login(base, r["pin"])[0] == 200


def test_wrong_pins_lock_out(server):
    base, _ = server
    wrong = next(p for p in (f"{i:06d}" for i in range(100)) if p not in PINS.values())
    for _ in range(5):
        assert login(base, wrong)[0] == 401
    assert login(base, PINS["Rahim Uddin"])[0] == 429              # locked even with the right PIN


def test_gps_pings_make_visits_and_routes(server):
    base, _ = server
    _, staff = call(f"{base}/api/staff")
    _, shops = call(f"{base}/api/shops")
    rahim, shop = staff[[s["name"] for s in staff].index("Rahim Uddin")]["id"], shops[0]
    code, r = post(f"{base}/api/locations", {"staff_id": rahim, "lat": shop["lat"], "lon": shop["lon"], "acc": 8})
    assert code == 200 and [e["kind"] for e in r["events"]] == ["check_in"]
    assert post(f"{base}/api/locations", {"staff_id": rahim, "lat": 999, "lon": 0})[0] == 400
    assert post(f"{base}/api/locations", {"staff_id": 999, "lat": 23.8, "lon": 90.3})[0] == 400
    _, locs = call(f"{base}/api/locations")
    assert [(p["name"], p["lat"]) for p in locs] == [("Rahim Uddin", shop["lat"])]
    _, visits = call(f"{base}/api/visits")
    assert [(v["staff"], v["shop"], v["check_out"]) for v in visits] == [("Rahim Uddin", shop["name"], None)]
    _, routes = call(f"{base}/api/routes")
    assert {r["name"]: r["visits"] for r in routes}["Rahim Uddin"] == 1
    _, events = call(f"{base}/api/events?after=0")
    assert events[0]["kind"] == "check_in"


def test_shops_and_changeable_agreement(server):
    base, _ = server
    _, shops = call(f"{base}/api/shops")
    assert len(shops) == 8 and shops[0]["agreement"]["cups"] == 4
    code, r = post(f"{base}/api/shops", {"name": "New Shop", "lat": 23.81, "lon": 90.37, "area": "Mirpur 10"})
    assert code == 201
    assert post(f"{base}/api/shops/{r['id']}/agreement", {"cups": 8, "shelf": 1, "zone": "front", "by": "NutriCoco"})[0] == 201
    assert post(f"{base}/api/shops/{r['id']}/agreement", {"cups": -2})[0] == 400
    assert post(f"{base}/api/shops/9999/agreement", {"cups": 2})[0] == 400
    _, shops = call(f"{base}/api/shops")
    new = next(s for s in shops if s["name"] == "New Shop")
    assert (new["agreement"]["cups"], new["agreement"]["shelf"], new["agreement"]["created_by"]) == (8, 1, "NutriCoco")


def test_static_files_and_no_path_escape(server):
    base, _ = server
    assert call(f"{base}/index.html") == (200, b"dashboard")
    assert call(f"{base}/reports/old/e.jpg") == (200, b"jpg")
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(f"{base}/reports/../uploads/x.json", timeout=5)


def test_ask_needs_finished_audit_and_reports_ai_offline(server):
    base, _ = server
    assert call(f"{base}/api/ask", json.dumps({"id": "nope", "question": "q"}).encode())[0] == 404
    _, r = call(f"{base}/api/audits?shop=a", b"video", "video/mp4")
    wait_done(base, r["id"])
    code, body = call(f"{base}/api/ask", json.dumps({"id": r["id"], "question": "How many keyframes?"}).encode())
    assert code == 503 and "not running" in body["error"]
