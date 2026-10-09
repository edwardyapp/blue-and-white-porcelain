import os
from PIL import Image
from tqdm import tqdm

IMAGE_ROOT = "images"

bad_images = []

for root, _, files in os.walk(IMAGE_ROOT):
    for f in files:
        if not f.lower().endswith(".jpg"):
            continue

        path = os.path.join(root, f)

        try:
            with Image.open(path) as img:
                img.convert("RGB")   # FORCE full decode
        except Exception as e:
            bad_images.append((path, str(e)))

print(f"\nFound {len(bad_images)} corrupted images:\n")

for path, err in bad_images:
    print(path)
    print(f"  → {err}\n")
