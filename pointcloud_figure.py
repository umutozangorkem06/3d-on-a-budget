from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

photo_dir = Path("data/scene1")
vggt = np.load("outputs/scene1_vggt_cameras.npz")
out_path = Path("results/2026-10-20_scene1/pointclouds.jpg")
thresholds = [5.0, 2.0]

size = 518
width, height = 900, 680
azimuth = np.radians(35)
elevation = np.radians(32)
distance = 1.9
lift = -0.1
background = 32
focal = 1300
gap = 10
top = 46
font = ImageFont.load_default(size=26)


def vggt_crop(image):
    w, h = image.size
    resized_h = round(h * size / w / 14) * 14
    y0 = (resized_h - size) // 2 * h / resized_h
    y1 = y0 + size * h / resized_h
    return image.crop((0, round(y0), w, round(y1))).resize((size, size), Image.LANCZOS)


extrinsic = vggt["extrinsic"]
intrinsic = vggt["intrinsic"]
depth = vggt["depth"][..., 0]
conf = vggt["depth_conf"]
n = depth.shape[0]

v, u = np.mgrid[0:size, 0:size]
points, colors, scores = [], [], []
for i in range(n):
    k = intrinsic[i]
    z = depth[i]
    cam = np.stack([(u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z], axis=-1).reshape(-1, 3)
    r, t = extrinsic[i, :, :3], extrinsic[i, :, 3]
    points.append((cam - t) @ r)
    photo = vggt_crop(Image.open(photo_dir / f"img_{i + 1:02d}.jpg").convert("RGB"))
    colors.append(np.asarray(photo).reshape(-1, 3))
    scores.append(conf[i].reshape(-1))
points = np.concatenate(points)
colors = np.concatenate(colors)
scores = np.concatenate(scores)

centres = -np.einsum("nji,nj->ni", extrinsic[:, :, :3], extrinsic[:, :, 3])
_, _, vt = np.linalg.svd(centres - centres.mean(0))
up = vt[2]
target = np.median(points[scores >= 5.0], axis=0)
if np.dot(centres.mean(0) - target, up) < 0:
    up = -up
a = centres[0] - target
a = a - np.dot(a, up) * up
a = a / np.linalg.norm(a)
b = np.cross(up, a)
target = target + lift * up
direction = np.cos(elevation) * (np.cos(azimuth) * a + np.sin(azimuth) * b) + np.sin(elevation) * up
eye = target + distance * direction
forward = (target - eye) / np.linalg.norm(target - eye)
right = np.cross(forward, up)
right = right / np.linalg.norm(right)
down = np.cross(forward, right)
view = np.stack([right, down, forward])


def render(mask):
    p = (points[mask] - eye) @ view.T
    c = colors[mask]
    keep = p[:, 2] > 0.05
    p, c = p[keep], c[keep]
    x = np.round(focal * p[:, 0] / p[:, 2] + width / 2).astype(int)
    y = np.round(focal * p[:, 1] / p[:, 2] + height / 2).astype(int)
    order = np.argsort(-p[:, 2])
    x, y, c = x[order], y[order], c[order]
    image = np.full((height, width, 3), background, np.uint8)
    for dx in (0, 1):
        for dy in (0, 1):
            xs, ys = x + dx, y + dy
            inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            image[ys[inside], xs[inside]] = c[inside]
    return Image.fromarray(image)


figure = Image.new("RGB", (len(thresholds) * width + (len(thresholds) - 1) * gap, top + height), "white")
draw = ImageDraw.Draw(figure)
for col, threshold in enumerate(thresholds):
    mask = scores >= threshold
    x = col * (width + gap)
    figure.paste(render(mask), (x, top))
    title = f"Threshold {threshold:.1f}: {mask.sum():,} points"
    draw.text((x + width / 2, top / 2), title, fill=(31, 78, 121), font=font, anchor="mm")

figure = figure.resize((1400, round(figure.height * 1400 / figure.width)), Image.LANCZOS)
figure.save(out_path, quality=85)
print(f"Saved: {out_path}")
