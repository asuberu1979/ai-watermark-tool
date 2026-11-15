#!/usr/bin/env python3
"""aisign - tiled AI-readable signature watermark tool (prototype).

Lays a repeated credit text over an image by shifting local brightness
(+/- strength) under a text mask: lightens on dark areas, darkens on
bright areas, so the text stays readable on any background while staying
subtle for humans. AI vision models (Grok / Gemini class) can read the
text and attribute the work, even after platform recompression.

Usage:
  python3 aisign.py input.jpg --text "ORIGINAL BY @your_name" \
      --sub "not the reposter's work" --strength 20 --output signed.png

Strength guide (0-255 brightness levels, measured on a 1280px-wide image):
  30+  clearly visible to humans on smooth areas; proven readable by Grok on X
  20   faint; readable by AI in tests so far (field test recommended)
  10-15 near-invisible to humans; AI readability is model-dependent
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_B_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
]
FONT_R_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
]


def pick_font(cands, size):
    for p in cands:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def build_mask(w, h, text, sub, scale_ref=1280):
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    scale = w / scale_ref
    f1 = pick_font(FONT_B_CANDIDATES, int(40 * scale))
    f2 = pick_font(FONT_R_CANDIDATES, int(30 * scale))
    # tile width from measured text width so tiles never overlap badly
    tw = d.textlength(text, font=f1)
    tile_w = int(tw + 90 * scale)
    step_y = int(300 * scale)
    row = 0
    for y in range(int(60 * scale), h, step_y):
        offset = -int(120 * scale) + (row % 2) * int(tile_w // 2)
        for x in range(offset, w, tile_w):
            d.text((x, y), text, font=f1, fill=255)
            if sub:
                d.text((x, y + int(52 * scale)), sub, font=f2, fill=255)
        row += 1
    return mask


def embed(img, mask, strength):
    arr = np.asarray(img.convert("RGB")).astype(np.int16)
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    m = np.asarray(mask).astype(np.float32) / 255.0
    direction = np.where(lum < 140, 1.0, -1.0)
    shift = (direction * strength * m)[..., None]
    return Image.fromarray(np.clip(arr + shift, 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser(description="Tiled AI-readable signature watermark")
    ap.add_argument("input")
    ap.add_argument("--text", default="ORIGINAL BY @homurasaki_eris")
    ap.add_argument("--sub", default="not the reposter's work")
    ap.add_argument("--strength", type=int, default=20)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    img = Image.open(args.input).convert("RGB")
    mask = build_mask(img.width, img.height, args.text, args.sub)
    out = embed(img, mask, args.strength)
    dest = args.output or os.path.splitext(args.input)[0] + f"_signed_S{args.strength}.png"
    out.save(dest)
    print(f"saved: {dest} (strength={args.strength}, text={args.text!r})")


if __name__ == "__main__":
    main()
