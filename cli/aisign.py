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
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
]
FONT_R_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
]


def pick_font(cands, size):
    for p in cands:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def build_mask(w, h, text, sub, scale_ref=1280, font_scale=1.0, placement="tile"):
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    scale = w / scale_ref
    f1 = pick_font(FONT_B_CANDIDATES, int(40 * scale * font_scale))
    f2 = pick_font(FONT_R_CANDIDATES, int(30 * scale * font_scale))
    gap = int(52 * scale * font_scale)
    lines = [(text, f1)] + ([(sub, f2)] if sub else [])
    if placement == "five":
        # four corners + centre: survives edge trims and centre crops better than a single spot
        margin = int(36 * scale)
        main_h = int(40 * scale * font_scale)
        spots = [("l", "t"), ("r", "t"), ("l", "b"), ("r", "b"), ("c", "c")]
        for ax, ay in spots:
            if ay == "t":
                y = margin
            elif ay == "b":
                y = h - margin - gap * (len(lines) - 1) - main_h
            else:
                y = h // 2 - (gap * (len(lines) - 1)) // 2 - main_h // 2
            for t, f in lines:
                tw = d.textlength(t, font=f)
                x = margin if ax == "l" else (w - margin - tw if ax == "r" else (w - tw) // 2)
                d.text((x, y), t, font=f, fill=255)
                y += gap
        return mask
    if placement in ("br", "bl"):
        # single spot near a bottom corner: tidy, but a crop removes it entirely
        margin = int(36 * scale)
        y = h - margin - gap * (len(lines) - 1) - int(40 * scale * font_scale)
        for t, f in lines:
            tw = d.textlength(t, font=f)
            x = w - margin - tw if placement == "br" else margin
            d.text((x, y), t, font=f, fill=255)
            y += gap
        return mask
    # tile width from measured text width so tiles never overlap badly
    tw = d.textlength(text, font=f1)
    tile_w = int(tw + 90 * scale)
    step_y = int(300 * scale * font_scale)
    row = 0
    for y in range(int(60 * scale), h, step_y):
        offset = -int(120 * scale) + (row % 2) * int(tile_w // 2)
        for x in range(offset, w, tile_w):
            d.text((x, y), text, font=f1, fill=255)
            if sub:
                d.text((x, y + gap), sub, font=f2, fill=255)
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
    ap.add_argument("--font-scale", type=float, default=1.0,
                    help="font size multiplier (e.g. 1.5 or 2.0); larger glyphs survive recompression better at low strength")
    ap.add_argument("--placement", choices=["tile", "five", "br", "bl"], default="tile",
                    help="tile = full tiling (default, crop-resistant); five = four corners + centre; br/bl = single spot at a bottom corner (tidy but crop-fragile)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    img = Image.open(args.input).convert("RGB")
    mask = build_mask(img.width, img.height, args.text, args.sub, font_scale=args.font_scale, placement=args.placement)
    out = embed(img, mask, args.strength)
    dest = args.output or os.path.splitext(args.input)[0] + f"_signed_S{args.strength}.png"
    out.save(dest)
    print(f"saved: {dest} (strength={args.strength}, font_scale={args.font_scale}, text={args.text!r})")


if __name__ == "__main__":
    main()
