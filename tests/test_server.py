"""Tests for src/server/app.py with a fake audit (no detector, no Ollama needed).

Run from the project root:  python -m pytest -v
"""
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from src.server import app

CFG = {"server": {"max_upload_mb": 1, "location_ttl_sec": 600, "device": "cpu"},
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
    store = app.Store(CFG, tmp_path / "uploads", tmp_path / "reports", run_audit=fake_audit)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), app.make_handler(store, site))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", tmp_path
    srv.shutdown()


def call(url, data=None, ctype="application/json"):
    req = urllib.request.Request(url, data=data, headers={"Content-Type": ctype} if data is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
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


def test_locations(server):
    base, _ = server
    assert call(f"{base}/api/locations", json.dumps({"rep": "Rahim", "lat": 23.81, "lon": 90.37, "acc": 9}).encode())[0] == 200
    assert call(f"{base}/api/locations", json.dumps({"rep": "x", "lat": 999, "lon": 0}).encode())[0] == 400
    _, locs = call(f"{base}/api/locations")
    assert [(p["rep"], p["lat"]) for p in locs] == [("Rahim", 23.81)]


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
