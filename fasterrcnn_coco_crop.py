import os
import torch
import torchvision
import cv2
from tqdm import tqdm

# =========================
# CONFIG
# =========================
IMAGE_ROOT = "images"
OUTPUT_ROOT = "images_cropped_rcnn"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONF_THRESH = 0.6
PADDING_RATIO = 0.08

# COCO labels likely to match ceramics
CERAMIC_CLASSES = {
    "vase", "bottle", "cup", "bowl", "potted plant", "wine glass"
}

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# =========================
# LOAD MODEL
# =========================
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.to(DEVICE)
model.eval()

# 🔧 Correct COCO ID → name mapping (non-contiguous IDs!)
COCO_ID_TO_NAME = {
    44: "bottle",
    46: "wine glass",
    47: "cup",
    51: "bowl",
    64: "potted plant",
    86: "vase"
}

# =========================
# MAIN LOOP
# =========================
for obj_id in tqdm(os.listdir(IMAGE_ROOT), desc="Objects"):
    in_dir = os.path.join(IMAGE_ROOT, obj_id)
    if not os.path.isdir(in_dir):
        continue

    out_dir = os.path.join(OUTPUT_ROOT, obj_id)
    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(in_dir):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(in_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        # Fix negative stride issue
        img_rgb = img[:, :, ::-1].copy()

        img_tensor = torch.from_numpy(img_rgb).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            preds = model(img_tensor)[0]

        best_box = None
        best_area = 0

        for box, label, score in zip(
            preds["boxes"], preds["labels"], preds["scores"]
        ):
            if score < CONF_THRESH:
                continue

            label_id = label.item()
            class_name = COCO_ID_TO_NAME.get(label_id)

            if class_name not in CERAMIC_CLASSES:
                continue

            area = (box[2] - box[0]) * (box[3] - box[1])
            if area > best_area:
                best_area = area
                best_box = box.cpu().numpy()

        # Fallback: no detection
        if best_box is None:
            cv2.imwrite(os.path.join(out_dir, fname), img)
            continue

        x1, y1, x2, y2 = best_box.astype(int)

        pad_x = int((x2 - x1) * PADDING_RATIO)
        pad_y = int((y2 - y1) * PADDING_RATIO)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        crop = img[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(out_dir, fname), crop)

print("✅ COCO Faster R-CNN ceramic-aware cropping complete")
