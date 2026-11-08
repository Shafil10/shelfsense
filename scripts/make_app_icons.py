"""Draw the ShelfSense app icons (home-screen icons for Android and iPhone) into app/field/icons/.

The design follows the dashboard logo: a rounded tile with three shelf bars and a ring (the "checked" mark).
Run from the project root:  python scripts/make_app_icons.py
"""
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "app" / "field" / "icons"
BG, C1, C2, FG = (5, 10, 19), (46, 230, 255), (157, 123, 255), (2, 18, 26)


def gradient(size):
    """Diagonal cyan -> purple gradient."""
    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            t = (x + y) / (2 * (size - 1))
            px[x, y] = tuple(round(a + (b - a) * t) for a, b in zip(C1, C2))
    return img


def icon(size, maskable=False):
    """maskable: the logo sits inside the central 80% safe zone and the background fills the square."""
    img = Image.new("RGB", (size, size), BG)
    pad = round(size * (0.2 if maskable else 0.08))
    tile = size - 2 * pad
    mask = Image.new("L", (tile, tile), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, tile - 1, tile - 1), radius=round(tile * 0.22), fill=255)
    img.paste(gradient(tile), (pad, pad), mask)
    d = ImageDraw.Draw(img)
    u = tile / 32  # the dashboard logo is drawn on a 32-unit grid
    for i, (w, alpha) in enumerate(((16, 255), (16, 190), (10, 130))):
        y = pad + (7 + 6 * i) * u
        colour = tuple(round(c * alpha / 255 + g * (1 - alpha / 255)) for c, g in zip(FG, C1))
        d.rounded_rectangle((pad + 8 * u, y, pad + (8 + w) * u, y + 2.5 * u), radius=u, fill=colour)
    cx, cy, r = pad + 22.5 * u, pad + 22.5 * u, 4.5 * u
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=FG, width=max(2, round(2 * u)))
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, size, maskable in (("icon-192.png", 192, False), ("icon-512.png", 512, False),
                                 ("maskable-512.png", 512, True), ("apple-touch-icon.png", 180, True)):
        icon(size, maskable).save(OUT / name, optimize=True)
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
