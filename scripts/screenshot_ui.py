"""Screenshot a ShelfSense page in headless Edge, optionally after logging in with a PIN and clicking buttons.
For checking the UI and for report/demo figures. Edge's profile stays on E: (E:\\tools\\edge-test\\profile).

Usage (project root, venv active, server running):
  python scripts/screenshot_ui.py --url http://localhost:8000/office/ --pin <office PIN> --click "[data-tab=today]" --out E:/tools/edge-test/today.png
"""
import argparse
import base64
import json
import subprocess
import time
import urllib.request
from pathlib import Path

import websocket

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


class Page:
    def __init__(self, ws_url):
        self.ws, self.n, self.errors = websocket.create_connection(ws_url, timeout=30), 0, []

    def send(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("method") == "Runtime.exceptionThrown":  # JavaScript errors on the page
                self.errors.append(msg["params"]["exceptionDetails"].get("exception", {}).get("description", "error"))
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result", {})

    def js(self, expr):
        r = self.send("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"].get("exception", {}).get("description", r["exceptionDetails"]))
        return r.get("result", {}).get("value")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--pin")
    ap.add_argument("--click", nargs="*", default=[], help="CSS selectors clicked in order (after login)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--mobile", action="store_true", help="phone emulation (390 px wide)")
    ap.add_argument("--wait", type=float, default=2.5, help="seconds to wait after each step")
    ap.add_argument("--fake-camera", action="store_true", help="Edge's test-pattern camera, allowed without asking")
    ap.add_argument("--geo", help="fake GPS position 'lat,lon' (location allowed without asking)")
    ap.add_argument("--no-shot", action="store_true", help="only run the steps (e.g. a recording test)")
    args = ap.parse_args()
    port = 9333
    extra = ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"] if args.fake_camera else []
    edge = subprocess.Popen([EDGE, "--headless=new", "--disable-gpu", f"--remote-debugging-port={port}",
                             r"--user-data-dir=E:\tools\edge-test\profile-cdp", r"--crash-dumps-dir=E:\tools\edge-test\dumps",
                             "--no-first-run", *extra, "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=1))
                break
            except OSError:
                time.sleep(0.2)
        page = Page(next(t for t in targets if t["type"] == "page")["webSocketDebuggerUrl"])
        page.send("Page.enable")
        page.send("Runtime.enable")
        w, h = (390, 844) if args.mobile else (args.width, args.height)
        page.send("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=2 if args.mobile else 1, mobile=args.mobile)
        if args.geo:
            lat, lon = map(float, args.geo.split(","))
            page.send("Browser.grantPermissions", permissions=["geolocation", "notifications"])
            page.send("Emulation.setGeolocationOverride", latitude=lat, longitude=lon, accuracy=10)
        page.send("Page.navigate", url=args.url)
        time.sleep(args.wait)
        if args.pin:
            # no PIN box = still logged in from an earlier run (the session cookie stays in the test profile)
            page.js(f"(() => {{ const el = document.getElementById('pin'); if (!el) return false; "
                    f"el.value = '{args.pin}'; el.form.requestSubmit(); return true; }})()")
            time.sleep(args.wait)
        for sel in args.click:
            found = page.js(f"(() => {{ const el = document.querySelector({json.dumps(sel)}); if (el) el.click(); return !!el; }})()")
            if not found:
                print("not found:", sel)
            time.sleep(args.wait)
        if not args.no_shot:
            shot = page.send("Page.captureScreenshot", format="png", captureBeyondViewport=not args.mobile)
            Path(args.out).write_bytes(base64.b64decode(shot["data"]))
        print("saved" if not args.no_shot else "done", args.out, "| JavaScript errors:", page.errors or "none")
    finally:
        edge.terminate()


if __name__ == "__main__":
    main()
