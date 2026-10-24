"""Build the standalone dashboard page served by start_dashboard.bat.

app/mockup/index.html is written without <!doctype>/<html>/<head>/<body> tags. This wraps it into a
complete page at app/mockup/site/index.html.

Run from the project root:  python scripts/build_mockup_site.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "app" / "mockup" / "index.html"
OUT = ROOT / "app" / "mockup" / "site" / "index.html"

RESET = ("<style>html{color-scheme:light}body{margin:0}img{max-width:100%}[hidden]{display:none!important}"
         ":root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}</style>\n")


def main():
    src = SRC.read_text(encoding="utf-8")
    i = src.index('<div class="demo"')  # everything before the first visible element goes in <head>
    head = src[:i].replace('<meta name="claude:viewport-fit-sides" content="cover">\n', "")
    page = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
            + RESET + head + "</head>\n<body>\n" + src[i:] + "\n</body>\n</html>\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(page):,} chars)")


if __name__ == "__main__":
    main()
