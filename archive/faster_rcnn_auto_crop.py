import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
from tqdm import tqdm

# =========================
# CONFIG
# =========================
INPUT_ROOT = "images"
OUTPUT_ROOT = "images_cropped_rcnn2"
MODEL_PATH = "ceramic_detector.pt"

SCORE_THRESH = 0.6      # confidence threshold
PADDING_RATIO = 0.05    # padding around detected box
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 2  # background + ceramic

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# =========================
# LOAD MODEL
# =========================
model = fasterrcnn_resnet50_fpn(pretrained=False)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features, NUM_CLASSES
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("✅ Ceramic detector loaded")

# =========================
# CROP FUNCTION
# =========================
def detect_and_crop(img):
    img_tensor = F.to_tensor(img).to(DEVICE)

    with torch.no_grad():
        outputs = model([img_tensor])[0]

    if len(outputs["boxes"]) == 0:
        return None

    scores = outputs["scores"]
    best_idx = torch.argmax(scores).item()

    if scores[best_idx] < SCORE_THRESH:
        return None

    x1, y1, x2, y2 = outputs["boxes"][best_idx].cpu().numpy()

    w, h = img.size
    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(bw * PADDING_RATIO)
    pad_y = int(bh * PADDING_RATIO)

    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(w, int(x2 + pad_x))
    y2 = min(h, int(y2 + pad_y))

    return img.crop((x1, y1, x2, y2))

# =========================
# MAIN LOOP
# =========================
for obj_id in tqdm(os.listdir(INPUT_ROOT), desc="Objects"):
    in_dir = os.path.join(INPUT_ROOT, obj_id)
    if not os.path.isdir(in_dir):
        continue

    out_dir = os.path.join(OUTPUT_ROOT, obj_id)
    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(in_dir):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        in_path = os.path.join(in_dir, fname)
        out_path = os.path.join(out_dir, fname)

        img = Image.open(in_path).convert("RGB")
        cropped = detect_and_crop(img)

        # Fallback: save original if detection fails
        if cropped is None:
            cropped = img

        cropped.save(out_path)

print("🏁 Ceramic cropping complete.")
