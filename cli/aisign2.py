#!/usr/bin/env python3
"""aisign2 - 'invisible but AI-readable' signature watermark, R&D prototype.

Goal: maximise human invisibility at equal AI readability, versus the v1
tile signature (uniform brightness shift under a binary text mask).

Techniques under test:
  A. Perceptual masking: the shift strength is scaled by local image
     activity (local luminance std). Flat areas (sky, skin) where the
     human eye catches faint text get a small fraction of the strength;
     textured areas (hair, knit, foliage) where contrast masking hides
     the shift get full or boosted strength.
  B. Soft mask edges: the text mask is gaussian-blurred (~1.2px) so the
     shift has no hard outline; the eye catches edges, VLMs read area
     contrast.
  C. Larger glyphs (x1.5) at lower strength: survives the downscale a
     vision model applies, per the 2026-11-15 field result (x2.0 + S10
     passed on X).
  D. (variant only) Blue-channel bias: part of the shift is pushed into
     the B channel, where human luminance sensitivity is lowest.
     Risk: JPEG 4:2:0 chroma subsampling may smear it.

Measurement:
  - flat-area max/mean |delta| inside the mask  -> human visibility proxy
  - survival: mean |delta| inside the mask after JPEG q85 and after
    JPEG q85 + 75% resize roundtrips          -> AI readability proxy
"""
import io
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

LINES = [
    ("ORIGINAL BY @homurasaki_eris", True),
    ("Homurasaki Eris", False),
    ("not the reposter's work", False),
]


def build_mask(w, h, lines=LINES, font_scale=1.5):
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    s = (w / 1280) * font_scale
    fonts = [
        ImageFont.truetype(FONT_B if bold else FONT_R, int((40 if i == 0 else 30) * s))
        for i, (_, bold) in enumerate(lines)
    ]
    tw = max(d.textlength(t, font=f) for (t, _), f in zip(lines, fonts))
    tile_w = int(tw + 90 * (w / 1280))
    step_y = int(320 * (w / 1280) * font_scale)
    line_dy = [0, int(52 * s), int(96 * s)]
    row = 0
    for y in range(int(40 * (w / 1280)), h, step_y):
        offset = -int(100 * (w / 1280)) + (row % 2) * (tile_w // 2)
        for x in range(offset, w, tile_w):
            for (t, _), f, dy in zip(lines, fonts, line_dy):
                d.text((x, y + dy), t, font=f, fill=255)
        row += 1
    return mask


def activity_gain(lum, floor=0.35, boost=1.15, knee=18.0):
    """Per-pixel strength multiplier from local luminance std."""
    win = 21
    mean = ndimage.uniform_filter(lum, size=win)
    var = ndimage.uniform_filter(lum * lum, size=win) - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))
    gain = floor + (boost - floor) * np.clip(std / knee, 0.0, 1.0)
    # smooth the gain so strength changes are gradual, not patchy
    gain = ndimage.gaussian_filter(gain, sigma=6.0)
    return gain, std


def embed_v1(img, mask, strength):
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    m = np.asarray(mask).astype(np.float32) / 255.0
    direction = np.where(lum < 140, 1.0, -1.0)
    shift = (direction * strength * m)[..., None]
    return Image.fromarray(np.clip(arr + shift, 0, 255).astype(np.uint8))


def embed_v2(img, mask, strength, blue_bias=0.0, soft_sigma=1.2):
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    m = np.asarray(mask).astype(np.float32) / 255.0
    if soft_sigma:
        m = ndimage.gaussian_filter(m, sigma=soft_sigma)
    gain, _ = activity_gain(lum)
    direction = np.where(lum < 140, 1.0, -1.0)
    base = direction * strength * m * gain
    shift = np.stack([base, base, base * (1.0 + blue_bias)], axis=-1)
    return Image.fromarray(np.clip(arr + shift, 0, 255).astype(np.uint8))


def jpeg_roundtrip(img, quality, scale=1.0):
    im = img
    if scale != 1.0:
        im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    out = Image.open(buf).convert("RGB")
    if scale != 1.0:
        out = out.resize(img.size, Image.LANCZOS)
    return out


def metrics(wm, clean, mask):
    a = np.asarray(wm).astype(np.float32)
    b = np.asarray(clean).astype(np.float32)
    diff = np.abs(a - b).mean(axis=2)
    lum = 0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2]
    _, std = activity_gain(lum)
    m = np.asarray(mask) > 80
    flat = m & (std < 4)
    tex = m & (std >= 12)
    surv85, _ = _survival(wm, clean, mask, quality=85)
    surv85r, _ = _survival(wm, clean, mask, quality=85, scale=0.75)
    return {
        "flat_mean": float(diff[flat].mean()) if flat.any() else 0.0,
        "flat_p99": float(np.percentile(diff[flat], 99)) if flat.any() else 0.0,
        "tex_mean": float(diff[tex].mean()) if tex.any() else 0.0,
        "surv85": surv85,
        "surv85r": surv85r,
    }


def _survival(wm, clean, mask, **kw):
    wm_c = np.asarray(jpeg_roundtrip(wm, **kw)).astype(np.float32)
    cl_c = np.asarray(jpeg_roundtrip(clean, **kw)).astype(np.float32)
    diff = np.abs(wm_c - cl_c).mean(axis=2)
    m = np.asarray(mask) > 80
    bg = np.asarray(mask) == 0
    return float(diff[m].mean()), float(diff[bg].mean())


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "/home/hatch/workspace/instagram/eris-x-2026-11-18.jpg"
    img = Image.open(src).convert("RGB")
    print(f"source: {src} size={img.size}")
    mask15 = build_mask(img.width, img.height, font_scale=1.5)
    mask10 = build_mask(img.width, img.height, font_scale=1.0)
    mask15.save(os.path.join(HERE, "mask_v2_preview.png"))

    variants = [
        ("V0 v1 S10 x1.0 (control)", embed_v1(img, mask10, 10), mask10),
        ("V1 v2 S14 adaptive soft", embed_v2(img, mask15, 14), mask15),
        ("V2 v2 S18 adaptive soft", embed_v2(img, mask15, 18), mask15),
        ("V3 v2 S18 adaptive+blue", embed_v2(img, mask15, 18, blue_bias=0.6), mask15),
    ]
    hdr = f"{'variant':28} | {'flat mean':>9} | {'flat p99':>8} | {'tex mean':>8} | {'surv q85':>8} | {'surv q85+75%':>12}"
    print(hdr)
    for name, wm, mk in variants:
        tag = name.split()[0]
        wm.save(os.path.join(HERE, f"test_{tag}.png"))
        jpeg_roundtrip(wm, 85).save(os.path.join(HERE, f"test_{tag}_xsim.jpg"), quality=92)
        mt = metrics(wm, img, mk)
        print(f"{name:28} | {mt['flat_mean']:9.2f} | {mt['flat_p99']:8.2f} | "
              f"{mt['tex_mean']:8.2f} | {mt['surv85']:8.2f} | {mt['surv85r']:12.2f}")


def cli(argv):
    """Standard CLI (V2 adopted as the standard on 2026-11-18).

    Usage:
      python3 aisign2.py input.jpg [--strength 18] [--output out.png]
    Defaults are the field-verified V2 settings: adaptive perceptual
    masking, soft mask edges, x1.5 glyphs, 3-line signature
    (handle line / symbol-free name line / sub line).
    """
    import argparse

    ap = argparse.ArgumentParser(description="AI-readable signature watermark (v2 standard)")
    ap.add_argument("input")
    ap.add_argument("--text", default="ORIGINAL BY @homurasaki_eris")
    ap.add_argument("--name", default="Homurasaki Eris")
    ap.add_argument("--sub", default="not the reposter's work")
    ap.add_argument("--strength", type=int, default=18)
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    img = Image.open(args.input).convert("RGB")
    lines = [(args.text, True), (args.name, False), (args.sub, False)]
    mask = build_mask(img.width, img.height, lines=lines, font_scale=1.5)
    out = embed_v2(img, mask, args.strength)
    dest = args.output or os.path.splitext(args.input)[0] + f"_signed_v2_S{args.strength}.png"
    out.save(dest)
    print(f"saved: {dest} (v2 standard, strength={args.strength})")


if __name__ == "__main__":
    if "--exp" in sys.argv:
        sys.argv.remove("--exp")
        main()
    else:
        cli(sys.argv[1:])
