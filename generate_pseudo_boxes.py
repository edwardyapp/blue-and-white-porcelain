import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

IMAGE_ROOT = "images"
OUTPUT_CSV = "pseudo_boxes.csv"

EDGE_THRESH = 0.02   # edge energy threshold
MIN_AREA = 0.05      # min box area (fraction of image)
PADDING = 8

records = []

def get_pseudo_box(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 80, 160)

    row_energy = edges.sum(axis=1) / edges.sum()
    col_energy = edges.sum(axis=0) / edges.sum()

    rows = np.where(row_energy > EDGE_THRESH)[0]
    cols = np.where(col_energy > EDGE_THRESH)[0]

    if len(rows) == 0 or len(cols) == 0:
        return None

    y1, y2 = rows[0], rows[-1]
    x1, x2 = cols[0], cols[-1]

    # padding
    x1 = max(0, x1 - PADDING)
    y1 = max(0, y1 - PADDING)
    x2 = min(w, x2 + PADDING)
    y2 = min(h, y2 + PADDING)

    # area sanity check
    if (x2 - x1) * (y2 - y1) < MIN_AREA * w * h:
        return None

    return x1, y1, x2, y2

for obj_id in tqdm(os.listdir(IMAGE_ROOT), desc="Objects"):
    folder = os.path.join(IMAGE_ROOT, obj_id)
    if not os.path.isdir(folder):
        continue

    for fname in os.listdir(folder):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is None:
            continue

        box = get_pseudo_box(img)
        if box is None:
            continue

        records.append({
            "image": path,
            "x1": box[0],
            "y1": box[1],
            "x2": box[2],
            "y2": box[3]
        })

df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Saved {len(df)} pseudo boxes → {OUTPUT_CSV}")
