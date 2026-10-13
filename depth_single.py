from PIL import Image
from transformers import pipeline

image = Image.open("data/scene1/img_01.jpg")

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
depth = pipe(image)["depth"]
depth.save("outputs/img_01_depth.png")
print("saved: outputs/img_01_depth.png")