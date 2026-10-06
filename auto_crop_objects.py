import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# =========================
# CONFIG
# =========================
INPUT_ROOT = "/home/edward-yapp/PycharmProjects/AI-archaeology-paper-latest/images_clean"              # root with object folders
OUTPUT_ROOT = "images_clean_cropped"     # where cropped images go
PADDING_RATIO = 0.08               # extra padding around object
MIN_OBJECT_AREA = 0.01             # min area fraction to be considered object
OVERWRITE = False                  # overwrite originals if True

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# =========================
# AUTO CROP FUNCTION
# =========================
def auto_crop_image(img, padding_ratio=0.05):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu threshold
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Invert if background is white
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return img  # fallback

    # Largest contour = object
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    # Reject tiny detections
    img_area = img.shape[0] * img.shape[1]
    if (w * h) / img_area < MIN_OBJECT_AREA:
        return img

    # Padding
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img.shape[1], x + w + pad_x)
    y2 = min(img.shape[0], y + h + pad_y)

    return img[y1:y2, x1:x2]

# =========================
# MAIN LOOP
# =========================
for object_id in tqdm(os.listdir(INPUT_ROOT), desc="Processing objects"):
    in_folder = os.path.join(INPUT_ROOT, object_id)
    if not os.path.isdir(in_folder):
        continue

    out_folder = in_folder if OVERWRITE else os.path.join(OUTPUT_ROOT, object_id)
    os.makedirs(out_folder, exist_ok=True)

    for fname in os.listdir(in_folder):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        in_path = os.path.join(in_folder, fname)
        out_path = os.path.join(out_folder, fname)

        img = cv2.imread(in_path)
        if img is None:
            continue

        cropped = auto_crop_image(img, PADDING_RATIO)
        cv2.imwrite(out_path, cropped)

print("✅ Auto-cropping complete.")
