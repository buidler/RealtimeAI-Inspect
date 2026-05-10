from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

root = Path(r"d:\zjy\code\RealtimeAI-Inspect")
base = root / "RtDETRv2" / "output" / "rtdetrv2_fiber_4"
train = root / "RtDETRv2" / "dataset" / "data" / "train" / "images"
d_light = base / "hqsam_results_v2_tier_light"
d_loose = base / "hqsam_results_v2_loose_extreme"
out = base / "compare_light_vs_loose_extreme"
out.mkdir(parents=True, exist_ok=True)

exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
imgs = sorted([p for p in train.rglob("*") if p.suffix.lower() in exts])

fixed = 0
err = 0
for p in imgs:
    op = out / f"{p.stem}_hqsam_contours.jpg"
    if op.exists():
        continue
    try:
        f1 = d_light / f"{p.stem}_hqsam_contours.jpg"
        f2 = d_loose / f"{p.stem}_hqsam_contours.jpg"
        im1 = Image.open(f1).convert("RGB") if f1.exists() else Image.open(p).convert("RGB")
        im2 = Image.open(f2).convert("RGB") if f2.exists() else Image.open(p).convert("RGB")
        if not f2.exists():
            d = ImageDraw.Draw(im2)
            d.rectangle((0, 0, im2.width, 30), fill=(20, 20, 20))
            d.text((8, 8), "loose_extreme: missing", fill=(255, 80, 80))
        h = max(im1.height, im2.height)
        if im1.height != h:
            im1 = ImageOps.pad(im1, (im1.width, h), color=(0, 0, 0))
        if im2.height != h:
            im2 = ImageOps.pad(im2, (im2.width, h), color=(0, 0, 0))
        gap, top = 10, 36
        canvas = Image.new("RGB", (im1.width + im2.width + gap, h + top), (18, 18, 18))
        canvas.paste(im1, (0, top))
        canvas.paste(im2, (im1.width + gap, top))
        dr = ImageDraw.Draw(canvas)
        dr.text((8, 10), "light", fill=(255, 255, 255))
        dr.text((im1.width + gap + 8, 10), "loose_extreme", fill=(255, 255, 255))
        canvas.save(op, quality=95)
        fixed += 1
    except Exception:
        err += 1

print(f"fixed={fixed} err={err} total={len(list(out.glob('*_hqsam_contours.jpg')))}")
