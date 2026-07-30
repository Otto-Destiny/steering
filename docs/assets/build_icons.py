"""Derive the interface icons from the STEERING banner.

Run this only when the banner changes; the generated files are committed, so a
normal checkout needs neither this script nor Pillow.

    uv run --with pillow python docs/assets/build_icons.py

The symbol is cropped out of the banner rather than redrawn, so the icon is the
established mark and not a second treatment of it. Small sizes are cropped a
little tighter: at 16 px the hexagon frame is thinner than a pixel and turns to
haze, and giving the S and the needle more of the tile is what keeps the mark
legible. That is optical scaling, which is how icon sets have always worked.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[2]
BANNER = ROOT / "docs" / "assets" / "steering-hero.png"
STATIC = ROOT / "src" / "steering" / "web" / "static"

#: The mark's measured extent in the banner: the hexagon frame, plus the compass
#: needle that overhangs it to the right.
MARK = (330, 169, 710, 637)
#: The banner's ground, sampled clear of every drawn element.
GROUND = (0, 7, 8)


def tile(padding: float) -> Image.Image:
    """The mark centred on a square of ground, with room around it."""

    banner = Image.open(BANNER).convert("RGB")
    left, top, right, bottom = MARK
    width, height = right - left, bottom - top
    side = int(max(width, height) * padding)
    square = Image.new("RGB", (side, side), GROUND)
    square.paste(banner.crop(MARK), ((side - width) // 2, (side - height) // 2))
    return square


def scaled(source: Image.Image, size: int, *, contrast: float = 1.0) -> Image.Image:
    small = source.resize((size, size), Image.LANCZOS)
    return ImageEnhance.Contrast(small).enhance(contrast) if contrast != 1.0 else small


def main() -> None:
    STATIC.mkdir(parents=True, exist_ok=True)
    close, roomy = tile(1.04), tile(1.16)

    # One .ico carrying the three sizes a browser actually asks for.
    scaled(close, 48).save(
        STATIC / "favicon.ico",
        sizes=[(16, 16), (32, 32), (48, 48)],
        append_images=[scaled(close, 32, contrast=1.2), scaled(close, 16, contrast=1.35)],
    )
    # iOS rounds the corners of this one, so it is given the most room.
    scaled(tile(1.22), 180).save(STATIC / "apple-touch-icon.png")
    scaled(roomy, 512).save(STATIC / "icon-512.png")
    # Shown beside the wordmark; 128 stays crisp on a high-density screen.
    scaled(close, 128).save(STATIC / "steering-mark.png")

    for name in ("favicon.ico", "apple-touch-icon.png", "icon-512.png", "steering-mark.png"):
        print(f"{name:24} {(STATIC / name).stat().st_size:>7,} bytes")


if __name__ == "__main__":
    main()
