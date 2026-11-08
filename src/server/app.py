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
import threading
import urllib.error
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml

from src.server.db import DB

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"
SITE = ROOT / "app" / "mockup" / "site"
UPLOADS = ROOT / "data" / "uploads"
REPORTS = ROOT / "data" / "reports"
VIDEO_TYPES = {"video/mp4": ".mp4", "video/quicktime": ".mov", "video/webm": ".webm", "video/3gpp": ".3gp",
               "application/octet-stream": ".mp4"}


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
        self.run_audit = run_audit or self._run_pipeline
        self.jobs = queue.Queue()
        self.lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()

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

    def save_upload(self, stream, length, content_type, qs):
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
                "shop": shop, "rep": qs.get("rep", [""])[0], "captured_utc": qs.get("captured", [""])[0],
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
                report["capture"] = {k: meta[k] for k in ("shop", "rep", "captured_utc", "received_utc", "gps")}
                out = self.reports / Path(meta["video"]).stem / "audit.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(report, indent=1), encoding="utf-8")
                meta.update(status="done", verdict=report["integrity"]["verdict"],
                            processing_sec=report.get("processing_sec"))
            except Exception as e:  # a bad file must not stop the worker
                meta.update(status="failed", error=f"{type(e).__name__}: {e}")
            self.write_meta(meta)

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
        for name in ("Rahim Uddin", "Jamal Hossain", "Tania Akter", "Sumon Mia"):
            self.db.add_staff(name)
        return True


def make_handler(store, site=SITE):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(site), **kw)

        def log_message(self, fmt, *args):  # one short line per API call, no noise for static files
            if self.path.startswith("/api/"):
                print(f"{self.log_date_time_string()} {self.command} {self.path.split('?')[0]} {args[1] if len(args) > 1 else ''}")

        def translate_path(self, path):
            # /reports/... -> data/reports/... (evidence images); everything else -> the dashboard site.
            # SimpleHTTPRequestHandler drops ".." parts, so neither can leave its folder.
            if path.startswith("/reports/"):
                base, self.directory = self.directory, str(store.reports)
                try:
                    return super().translate_path(path[len("/reports"):])
                finally:
                    self.directory = base
            return super().translate_path(path)

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
            if url.path == "/api/audits":
                return self.send_json(store.list())
            if url.path.startswith("/api/audits/"):
                meta = store.read_meta(url.path.rsplit("/", 1)[1])
                if not meta:
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
                day = qs.get("day") or now_utc()[:10]
                return self.send_json([{**db.route(s["id"], day), "name": s["name"]} for s in db.staff()])
            if url.path == "/api/events":
                return self.send_json(db.events_since(int(qs.get("after") or 0)))
            if url.path.startswith("/api/"):
                return self.send_json({"error": "not found"}, 404)
            return super().do_GET()

        def do_POST(self):
            url = urlparse(self.path)
            try:
                if url.path == "/api/audits":
                    length = int(self.headers.get("Content-Length") or 0)
                    if length <= 0:
                        return self.send_json({"error": "empty upload"}, 400)
                    if length > store.cfg["server"]["max_upload_mb"] * 1_000_000:
                        return self.send_json({"error": "video too large"}, 413)
                    meta = store.save_upload(self.rfile, length, self.headers.get("Content-Type", ""),
                                             parse_qs(url.query))
                    return self.send_json({"id": meta["id"], "status": meta["status"]}, 202)
                db = store.db
                if url.path == "/api/locations":
                    d = self.read_json()
                    lat, lon = float(d["lat"]), float(d["lon"])
                    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                        raise ValueError("bad coordinates")
                    acc = float(d["acc"]) if d.get("acc") is not None else None
                    return self.send_json({"ok": True, "events": db.ping(int(d["staff_id"]), lat, lon, acc)})
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
                if url.path == "/api/ask":
                    d = self.read_json()
                    report = store.report(str(d.get("id", "")))
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
    ap.add_argument("--seed-demo", action="store_true", help="add sample shops and staff to an empty database")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    port = args.port or cfg["server"]["port"]
    # load torch + the detector code once, in the main thread, before any upload arrives (a first import inside
    # the worker thread was blocked by Windows Application Control on the demo PC, and it is faster this way)
    print("Loading the detector...")
    import src.report.audit  # noqa: F401
    import ultralytics  # noqa: F401
    store = Store(cfg)
    if args.seed_demo:
        print("Added sample shops and staff." if store.seed_demo() else "Database already has shops: nothing added.")
    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(store))
    print(f"ShelfSense server: http://localhost:{port}  (phones on the same Wi-Fi: http://<this PC's IPv4>:{port})")
    print(f"Uploads -> {UPLOADS}, reports -> {REPORTS}. Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
