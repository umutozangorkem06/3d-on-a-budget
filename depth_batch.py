import time
from pathlib import Path
from PIL import Image
from transformers import pipeline

in_dir = Path("data/scene1")
out_dir = Path("outputs/scene1")
out_dir.mkdir(parents=True, exist_ok=True)

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

times = []
for path in sorted(in_dir.glob("*.jpg")):
    image = Image.open(path)
    start = time.perf_counter()
    depth = pipe(image)["depth"]
    elapsed = time.perf_counter() - start
    times.append(elapsed)
    depth.save(out_dir / f"{path.stem}_depth.png")
    print(f"{path.name}: {elapsed:.2f} s")

warm = times[1:]
print(f"Images: {len(times)}")
print(f"Mean time per image (excluding first): {sum(warm) / len(warm):.2f} s")