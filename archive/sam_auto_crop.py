import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# =========================
# CONFIG
# =========================
IMAGE_ROOT = "images"
OUTPUT_ROOT = "images_sam_cropped"
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PADDING_RATIO = 0.05

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# =========================
# LOAD SAM
# =========================
sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
sam.to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.92,
    min_mask_region_area=5000
)

# =========================
# AUTO CROP
# =========================
def sam_crop(img):
    masks = mask_generator.generate(img)
    if not masks:
        return img

    # Largest mask = object
    mask = max(masks, key=lambda m: m["area"])["segmentation"]

    ys, xs = np.where(mask)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    h, w = img.shape[:2]
    pad_x = int((x2 - x1) * PADDING_RATIO)
    pad_y = int((y2 - y1) * PADDING_RATIO)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    return img[y1:y2, x1:x2]

# =========================
# MAIN LOOP
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

        cropped = sam_crop(img)
        cv2.imwrite(os.path.join(out_dir, f), cropped)

print("✅ SAM cropping complete.")
