from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

photo_dir = Path("data/scene1")
depth_dir = Path("outputs/scene1")
vggt = np.load("outputs/scene1_vggt_cameras.npz")
out_path = Path("results/2026-10-20_scene1/depth_compare.jpg")
views = [3, 12, 21]

size = 518
gap = 10
top = 46
left = 70
font = ImageFont.load_default(size=26)


def vggt_crop(image):
    w, h = image.size
    resized_h = round(h * size / w / 14) * 14
    y0 = (resized_h - size) // 2 * h / resized_h
    y1 = y0 + size * h / resized_h
    return image.crop((0, round(y0), w, round(y1))).resize((size, size), Image.LANCZOS)


def inverse_depth(depth):
    inv = 1.0 / np.clip(depth, 1e-6, None)
    lo, hi = np.percentile(inv, [1, 99])
    inv = np.clip((inv - lo) / (hi - lo), 0, 1)
    return Image.fromarray((inv * 255).astype(np.uint8))


titles = ["Photo", "Depth Anything V2 Small", "VGGT-1B"]
width = left + 3 * size + 2 * gap
height = top + len(views) * size + (len(views) - 1) * gap
figure = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(figure)

for col, title in enumerate(titles):
    x = left + col * (size + gap) + size / 2
    draw.text((x, top / 2), title, fill=(31, 78, 121), font=font, anchor="mm")

for row, view in enumerate(views):
    name = f"img_{view:02d}"
    y = top + row * (size + gap)
    photo = vggt_crop(Image.open(photo_dir / f"{name}.jpg").convert("RGB"))
    depth_anything = vggt_crop(Image.open(depth_dir / f"{name}_depth.png").convert("RGB"))
    vggt_depth = inverse_depth(vggt["depth"][view - 1, :, :, 0]).convert("RGB")
    for col, tile in enumerate([photo, depth_anything, vggt_depth]):
        figure.paste(tile, (left + col * (size + gap), y))
    draw.text((left / 2, y + size / 2), f"{view:02d}", fill=(64, 64, 64), font=font, anchor="mm")

figure = figure.resize((1200, round(height * 1200 / width)), Image.LANCZOS)
figure.save(out_path, quality=85)
print(f"Saved: {out_path}")
