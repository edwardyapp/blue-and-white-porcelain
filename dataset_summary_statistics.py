import os
from PIL import Image
from collections import defaultdict

IMAGE_ROOT = "images_clean_4"

sizes = []

for root, _, files in os.walk(IMAGE_ROOT):
    for fname in files:
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(root, fname)
            try:
                with Image.open(path) as img:
                    w, h = img.size
                    sizes.append((w, h))
            except Exception as e:
                print(f"⚠️ Could not read {path}: {e}")

if not sizes:
    print("No images found.")
    exit()

widths = [w for w, h in sizes]
heights = [h for w, h in sizes]

print(f"Total images: {len(sizes)}")
print(f"Width  range: {min(widths)} – {max(widths)}")
print(f"Height range: {min(heights)} – {max(heights)}")
print(f"Smallest image (pixels): {min(w*h for w,h in sizes)}")
print(f"Largest  image (pixels): {max(w*h for w,h in sizes)}")