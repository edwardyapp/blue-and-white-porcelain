import cv2
import numpy as np
import os
from tqdm import tqdm

# =========================
# CONFIG
# =========================
IMAGE_ROOT = "images_clean_4"
OUTPUT_ROOT = "images_clean_cropped_4"
ENERGY_THRESH = 0.03   # fraction of max energy
PADDING = 10

os.makedirs(OUTPUT_ROOT, exist_ok=True)

def ceramic_crop(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Energy profiles
    row_energy = edges.sum(axis=1)
    col_energy = edges.sum(axis=0)

    # Normalize
    row_energy = row_energy / row_energy.max()
    col_energy = col_energy / col_energy.max()

    # Rows with object
    rows = np.where(row_energy > ENERGY_THRESH)[0]
    cols = np.where(col_energy > ENERGY_THRESH)[0]

    if len(rows) == 0 or len(cols) == 0:
        return img  # fallback

    y1, y2 = rows[0], rows[-1]
    x1, x2 = cols[0], cols[-1]

    # Enforce symmetry around center
    cx = w // 2
    half_w = max(cx - x1, x2 - cx)
    x1 = max(0, cx - half_w)
    x2 = min(w, cx + half_w)

    # Padding
    x1 = max(0, x1 - PADDING)
    y1 = max(0, y1 - PADDING)
    x2 = min(w, x2 + PADDING)
    y2 = min(h, y2 + PADDING)

    return img[y1:y2, x1:x2]

# =========================
# MAIN
# =========================
for obj in tqdm(os.listdir(IMAGE_ROOT), desc="Objects"):
    in_dir = os.path.join(IMAGE_ROOT, obj)
    if not os.path.isdir(in_dir):
        continue

    out_dir = os.path.join(OUTPUT_ROOT, obj)
    os.makedirs(out_dir, exist_ok=True)

    for f in os.listdir(in_dir):
        if not f.lower().endswith((".jpg", ".png")):
            continue

        img = cv2.imread(os.path.join(in_dir, f))
        if img is None:
            continue

        cropped = ceramic_crop(img)
        cv2.imwrite(os.path.join(out_dir, f), cropped)

print("✅ Ceramic-aware cropping complete.")
