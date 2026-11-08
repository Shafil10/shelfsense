"""Sprint 6: back end for the mobile app. Runs on the demo PC; phones connect over Wi-Fi.

One small server (Python's built-in http.server, nothing to install) that:
  - serves the dashboard (app/mockup/site/) and the evidence images of every audit (data/reports/),
  - receives a shelf video from a phone, saves it in data/uploads/ and runs the real audit pipeline
    (src/report/audit.py: keyframes, detector, integrity checks) in a background worker, one video at a time,
  - keeps the merchandisers' last GPS positions in memory for the live map,
  - answers "Ask about this audit" questions through the local AI (src/assistant/ask.py), so phones never
    talk to Ollama directly.

API (JSON unless noted):
  POST /api/audits?shop=..&rep=..&lat=..&lon=..&acc=..&captured=..   body = the video file (raw bytes)
       -> {"id", "status": "queued"}
  GET  /api/audits            -> uploads, newest first: id, status, shop, rep, received, verdict
  GET  /api/audits/<id>       -> upload metadata + status + the audit report when done
  POST /api/locations         {"rep", "lat", "lon", "acc"} -> {"ok": true}
  GET  /api/locations         -> positions newer than server.location_ttl_sec
  POST /api/ask               {"id", "question"} -> {"answer", "seconds", "model"}

Usage (project root, venv active; Ollama running for /api/ask):
  python -m src.server.app              # http://localhost:8000 on this PC, http://<PC IPv4>:8000 on the Wi-Fi
"""
import argparse
import json
import os
import queue
import re
import threading
import time
import urllib.error
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml

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
    """Uploads and their audit reports on disk, GPS positions in memory, and the audit worker."""

    def __init__(self, cfg, uploads=UPLOADS, reports=REPORTS, run_audit=None):
        self.cfg, self.uploads, self.reports = cfg, Path(uploads), Path(reports)
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.run_audit = run_audit or self._run_pipeline
        self.jobs = queue.Queue()
        self.lock = threading.Lock()
        self.locations = {}
        threading.Thread(target=self._worker, daemon=True).start()

    def meta_path(self, uid):
        return self.uploads / f"{uid}.json"

    def read_meta(self, uid):
        p = self.meta_path(uid)
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
        metas = [json.loads(p.read_text(encoding="utf-8")) for p in self.uploads.glob("*.json")]
        return sorted(metas, key=lambda m: m["received_utc"] + m["id"], reverse=True)

    def set_location(self, rep, lat, lon, acc):
        with self.lock:
            self.locations[rep] = {"rep": rep, "lat": lat, "lon": lon, "accuracy_m": acc, "t": time.time()}

    def get_locations(self):
        ttl = self.cfg["server"]["location_ttl_sec"]
        with self.lock:
            return [dict(v, age_sec=round(time.time() - v["t"])) for v in self.locations.values()
                    if time.time() - v["t"] <= ttl]


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
            if url.path == "/api/locations":
                return self.send_json(store.get_locations())
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
                if url.path == "/api/locations":
                    d = self.read_json()
                    lat, lon = float(d["lat"]), float(d["lon"])
                    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                        raise ValueError("bad coordinates")
                    store.set_location(str(d.get("rep", "unknown"))[:60], lat, lon, d.get("acc"))
                    return self.send_json({"ok": True})
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
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    port = args.port or cfg["server"]["port"]
    # load torch + the detector code once, in the main thread, before any upload arrives (a first import inside
    # the worker thread was blocked by Windows Application Control on the demo PC, and it is faster this way)
    print("Loading the detector...")
    import src.report.audit  # noqa: F401
    import ultralytics  # noqa: F401
    store = Store(cfg)
    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(store))
    print(f"ShelfSense server: http://localhost:{port}  (phones on the same Wi-Fi: http://<this PC's IPv4>:{port})")
    print(f"Uploads -> {UPLOADS}, reports -> {REPORTS}. Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
