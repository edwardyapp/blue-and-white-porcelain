import os
import re

# ============================================================
# 1️⃣ DEFINE BASE DIRECTORIES
# ============================================================

experiment_dir = "experiments/resnet50_frozen_all_baseaug_img448_ptV2_softmax_seed42_runs30"

log_dir = os.path.join(experiment_dir, "logs")
model_dir = os.path.join(experiment_dir, "models")

best_f1 = 0
best_file = None
best_seed = None

# =========================
# DYNASTY INFERENCE
# =========================
def infer_dynasty(row):
    period = str(row.get("Period", "")).lower()

    DYNASTY_KEYWORDS = [
        ("northern song", "Song"),
        ("southern song", "Song"),
        ("song dynasty", "Song"),
        ("jin dynasty", "Jin"),
        ("qing", "Qing"),
        ("ming", "Ming"),
        ("yuan", "Yuan"),
        ("tang", "Tang"),
        ("sui", "Sui"),
        ("han", "Han"),
    ]

    for key, dynasty in DYNASTY_KEYWORDS:
        if key in period:
            return dynasty

    # If SKIP_IF_NO_DYNASTY is True, return None to skip this object
    if SKIP_IF_NO_DYNASTY:
        return None

    begin = row.get("Object Begin Date")
    end = row.get("Object End Date")

    year = begin if pd.notna(begin) else end
    if pd.isna(year):
        return None

    year = int(year)
    if year < 220:
        return "Han"
    if 589 <= year < 618:
        return "Sui"
    if 618 <= year < 907:
        return "Tang"
    if 960 <= year < 1279:
        return "Song"
    if 1271 <= year < 1368:
        return "Yuan"
    if 1368 <= year < 1644:
        return "Ming"
    if 1644 <= year <= 1911:
        return "Qing"

    return "Other"

# -------------------------
# Period-level inference
# -------------------------
MING_PERIODS = ["Tianqi", "Chenghua", "Chongzhen", "Jiajing", "Longqing", "Wanli", "Xuande", "Zhengde"]
QING_PERIODS = ["Yongzheng", "Kangxi", "Daoguang", "Guangxu", "Jiaqing", "Qianlong", "Shunzhi"]

def infer_period(row):
    period_text = row.get("Period")

    # ----------------------
    # Case 1: Period text exists
    # ----------------------
    if pd.notna(period_text):
        text = str(period_text).lower()
        for p in MING_PERIODS + QING_PERIODS:
            if p.lower() in text:
                return p

    # ----------------------
    # Case 2: Period text missing → infer from dates
    # ----------------------
    begin = row.get("Object Begin Date")
    end = row.get("Object End Date")

    year = None
    if pd.notna(begin):
        year = int(begin)
    elif pd.notna(end):
        year = int(end)

    if year is not None:
        # Infer Ming period
        MING_PERIOD_YEARS = [
            ("Hongwu", 1368, 1398), ("Jianwen", 1399, 1402), ("Yongle", 1403, 1424),
            ("Hongxi", 1425, 1425), ("Xuande", 1426, 1435), ("Zhengtong", 1436, 1449),
            ("Jingtai", 1450, 1457), ("Chenghua", 1465, 1487), ("Hongzhi", 1488, 1505),
            ("Zhengde", 1506, 1521), ("Jiajing", 1522, 1566), ("Longqing", 1567, 1572),
            ("Wanli", 1573, 1620), ("Taichang", 1620, 1620), ("Tianqi", 1621, 1627),
            ("Chongzhen", 1628, 1644)
        ]
        for name, start, end_ in MING_PERIOD_YEARS:
            if start <= year <= end_:
                return name

        # Infer Qing period
        QING_PERIOD_YEARS = [
            ("Shunzhi", 1644, 1661), ("Kangxi", 1662, 1722), ("Yongzheng", 1723, 1735),
            ("Qianlong", 1736, 1795), ("Jiaqing", 1796, 1820), ("Daoguang", 1821, 1850),
            ("Xianfeng", 1851, 1861), ("Tongzhi", 1862, 1874), ("Guangxu", 1875, 1908),
            ("Xuantong", 1909, 1911)
        ]
        for name, start, end_ in QING_PERIOD_YEARS:
            if start <= year <= end_:
                return name

    return None

# ============================================================
# 2️⃣ FIND BEST SEED BASED ON OBJECT-LEVEL TEST MACRO F1
# ============================================================

for filename in os.listdir(log_dir):
    if filename.startswith("results_seed") and filename.endswith(".txt"):

        filepath = os.path.join(log_dir, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract OBJECT-LEVEL TEST RESULTS section
        test_section_match = re.search(
            r"🧪 OBJECT-LEVEL TEST RESULTS(.*?)weighted avg",
            content,
            re.DOTALL
        )

        if test_section_match:
            test_section = test_section_match.group(1)

            # Extract macro avg F1
            macro_match = re.search(
                r"macro avg\s+\d+\.\d+\s+\d+\.\d+\s+(\d+\.\d+)",
                test_section
            )

            if macro_match:
                macro_f1 = float(macro_match.group(1))

                if macro_f1 > best_f1:
                    best_f1 = macro_f1
                    best_file = filename

                    # Extract seed number from filename
                    seed_match = re.search(r"results_seed(\d+)\.txt", filename)
                    if seed_match:
                        best_seed = seed_match.group(1)

# ============================================================
# 3️⃣ CONSTRUCT CORRESPONDING MODEL PATH
# ============================================================

if best_seed is not None:
    model_filename = f"model_seed{best_seed}.pth"
    best_model_path = os.path.join(model_dir, model_filename)

    print("Best log file:", best_file)
    print("Best OBJECT-LEVEL TEST macro F1:", best_f1)
    print("Best seed:", best_seed)
    print("Corresponding model path:", best_model_path)

    if os.path.exists(best_model_path):
        print("Model file exists. Ready to load.")
    else:
        print("WARNING: Model file not found!")
else:
    print("No valid seed found.")

# ============================================================
# 4️⃣ LOAD MODEL
# ============================================================

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models
from pytorch_grad_cam import GradCAM, AblationCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_resnet50_v2(output_dim=2):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # for param in model.parameters():
    #     param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, output_dim)
    )

    # for p in model.fc.parameters():
    #     p.requires_grad = True

    return model

model = create_resnet50_v2(output_dim=2)

# model.load_state_dict(torch.load(best_model_path, map_location=device))

checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(device)
model.eval()

print(f"Loaded model from seed {best_seed}")

# ============================================================
# 5️⃣ REBUILD df_samples EXACTLY LIKE TRAINING
# ============================================================

import pandas as pd

CSV_FILE = "filtered_chinese_porcelain.csv"
IMAGE_ROOT = "images_clean_4"
image_selection = "all"  # <-- must match experiment
CLASSIFY_BY_PERIOD = False
SKIP_IF_NO_DYNASTY = False

samples = []

df = pd.read_csv(CSV_FILE)

print("Rebuilding dataset exactly as in training...")

for _, row in df.iterrows():
    object_id = str(row.get("Object ID", "")).strip()

    if CLASSIFY_BY_PERIOD:
        label = infer_period(row)
    else:
        label = infer_dynasty(row)

    if not object_id or label is None:
        continue

    folder = os.path.join(IMAGE_ROOT, object_id)
    if not os.path.isdir(folder):
        continue

    medium = row.get("Medium")

    for fname in os.listdir(folder):
        name = fname.lower()

        if not name.endswith(".jpg"):
            continue

        img_path = os.path.join(folder, fname)

        if image_selection == "all":
            samples.append((img_path, label, object_id, medium))

        elif image_selection == "primary":
            if "primary" in name:
                samples.append((img_path, label, object_id, medium))

        elif image_selection == "no_bottom":
            if "bottom" not in name:
                samples.append((img_path, label, object_id, medium))

        elif image_selection == "primary_bottom":
            if ("primary" in name) or ("bottom" in name):
                samples.append((img_path, label, object_id, medium))

if not samples:
    raise RuntimeError("No valid images found.")

df_samples = pd.DataFrame(
    samples,
    columns=["image", "label", "object_id", "medium"]
)

print("Total images:", len(df_samples))

from sklearn.model_selection import train_test_split

base_seed = 42
seed = int(best_seed)

print("🔒 Using OBJECT-AWARE split")

keep_labels = ["Ming", "Qing"]
df_samples = df_samples[df_samples["label"].isin(keep_labels)]

print("\n⚠️ FOCUS MODE ENABLED: Ming vs Qing only")
print("Image counts:")
print(df_samples["label"].value_counts())

unique_objects = df_samples[["object_id", "label"]].drop_duplicates()

train_val_objs, test_objs = train_test_split(
    unique_objects,
    test_size=0.3,
    stratify=unique_objects["label"],
    random_state=base_seed
)

test_ids = set(test_objs["object_id"])

train_objs, val_objs = train_test_split(
    train_val_objs,
    test_size=0.5,
    stratify=train_val_objs["label"],
    random_state=seed
)

train_ids = set(train_objs["object_id"])
val_ids = set(val_objs["object_id"])

test_samples = df_samples[df_samples["object_id"].isin(test_ids)]

print("Test objects:", len(test_ids))
print("Test images:", len(test_samples))

# ===============================
# ❌ Objects the model got wrong
# ===============================
incorrect_objects = {
    "44740", "44752", "47649", "48051", "49189", "50247", "781812",
    "39666", "42190", "40518"
}

# -------------------------------
# Filter test samples to remove these
# -------------------------------
correct_test_samples = test_samples[~test_samples["object_id"].isin(incorrect_objects)]

# ============================================================
# 5️⃣ DEFINE IMAGES FOR VISUALISATION
# ============================================================

# selected_images = []
#
# for label in ["Ming", "Qing"]:
#     class_objects = correct_test_samples[correct_test_samples["label"] == label]["object_id"].drop_duplicates().tolist()
#
#     for obj_id in class_objects[:2]:  # pick first 2 correct objects per class
#         obj_images = correct_test_samples[correct_test_samples["object_id"] == obj_id]
#         first_image = obj_images.iloc[0]
#
#         selected_images.append(
#             (first_image["image"], obj_id)
#         )
#
# print("Selected test images (only correct objects):")
# print(selected_images)

# ============================================================
# 5️⃣ DEFINE ALL CORRECT IMAGES FOR VISUALISATION
# ============================================================

# Take only first N images for quick testing
# TEST_NUM_IMAGES = 1

selected_images = []

# Loop over all correct objects in test set
# for obj_id in correct_test_samples["object_id"].drop_duplicates().tolist():
# for obj_id in correct_test_samples["object_id"].drop_duplicates().tolist()[:TEST_NUM_IMAGES]:
#     obj_images = correct_test_samples[correct_test_samples["object_id"] == obj_id]
    # Add all images for this object
    # for _, row in obj_images.iterrows():
    #     selected_images.append(
    #         (row["image"], obj_id)
    #     )

# print(f"Selected {len(selected_images)} images for visualisation (all correct objects).")

# Specify your four images and their object IDs
# Replace the paths below with the ones you actually want
selected_images = [
    # ("images_clean_4/44746/primary.jpg", "44746"),
    # ("images_clean_4/680974/extra_02.jpg", "680974"),
    # ("images_clean_4/680974/extra_02.jpg", "680974"),
    ("images_clean_4/666569/extra_01.jpg", "666569"),
    ("images_clean_4/680974/extra_02.jpg", "680974"),

    ("images_clean_4/685113/extra_02.jpg", "685113"),
    # ("images_clean_4/42201/extra_03.jpg", "42201"),
    # ("images_clean_4/47376/primary.jpg", "47376")
    # ("images_clean_4/460669/primary.jpg", "460669"),
    ("images_clean_4/50943/primary.jpg", "50943"),
]

print(f"Selected {len(selected_images)} images for visualisation (manual selection).")

# ============================================================
# 6️⃣ CAM TARGET LAYER (Correct for ResNet50)
# ============================================================

target_layers = [model.layer4[-1]]

original_images = []
gradcam_images = []
ablationcam_images = []
scorecam_images = []

# ============================================================
# 7️⃣ GENERATE CAMS
# ============================================================
for img_path, object_id in selected_images:
    print("Processing:", img_path)

    img = Image.open(img_path).convert('RGB')
    img = np.array(img)
    img = cv2.resize(img, (448, 448))
    img_float = np.float32(img) / 255.0

    input_tensor = preprocess_image(
        img_float,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ).to(device)

    # Use model prediction (not ground truth)
    output = model(input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()
    targets = [ClassifierOutputTarget(predicted_class)]

    # -------- GradCAM --------
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        gradcam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

    # -------- AblationCAM --------
    with AblationCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        ablationcam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

    # -------- ScoreCAM --------
    with ScoreCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        scorecam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

    original_images.append(img)
    gradcam_images.append(gradcam_image)
    ablationcam_images.append(ablationcam_image)
    scorecam_images.append(scorecam_image)

# ============================================================
# 8️⃣ GENERATE SCROLLABLE WEBPAGE FOR VISUAL EXPLANATIONS
# ============================================================

# import base64
# from io import BytesIO
#
# # Create output HTML file
# html_file = f"visual_explanations_seed{best_seed}.html"
#
# html_content = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
# <meta charset="UTF-8">
# <title>Visual Explanations - Seed {seed}</title>
# <style>
# body {{ font-family: Arial, sans-serif; margin: 20px; }}
# .container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
# .card {{ border: 1px solid #ccc; padding: 10px; width: 400px; }}
# .card img {{ width: 100%; height: auto; margin-bottom: 5px; }}
# .label {{ font-weight: bold; margin-bottom: 5px; }}
# .correct {{ color: green; font-weight: bold; }}
# .incorrect {{ color: red; font-weight: bold; }}
# </style>
# </head>
# <body>
# <h1>Visual Explanations (Seed {seed})</h1>
# <div class="container">
# """.format(seed=best_seed)
#
# # Add each image as a card
# for i, (img_path, object_id) in enumerate(selected_images):
#     # Ground truth label
#     gt_label = correct_test_samples[correct_test_samples["object_id"] == object_id]["label"].iloc[0]
#
#     # Model prediction
#     img_tensor = preprocess_image(
#         cv2.resize(np.array(Image.open(img_path).convert('RGB')), (448, 448)) / 255.0,
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     ).float().to(device)
#
#     # Add batch dimension if missing
#     if img_tensor.dim() == 3:
#         img_tensor = img_tensor.unsqueeze(0)  # shape: [1, 3, H, W]
#
#     output = model(img_tensor)
#     pred_label = ["Ming", "Qing"][torch.argmax(output, dim=1).item()]
#     correct_flag = "correct" if pred_label == gt_label else "incorrect"
#
#     # Convert images to base64
#     def img_to_base64(img_array):
#         pil_img = Image.fromarray(img_array)
#         buffer = BytesIO()
#         pil_img.save(buffer, format="PNG")
#         return base64.b64encode(buffer.getvalue()).decode("utf-8")
#
#     original_b64 = img_to_base64(original_images[i])
#     gradcam_b64 = img_to_base64(gradcam_images[i])
#     ablation_b64 = img_to_base64(ablationcam_images[i])
#     scorecam_b64 = img_to_base64(scorecam_images[i])
#
#     html_content += f"""
#     <div class="card">
#         <div class="label">Object ID: {object_id} | Ground Truth: {gt_label} | Prediction: <span class="{correct_flag}">{pred_label}</span></div>
#         <div><img src="data:image/png;base64,{original_b64}" alt="Original"></div>
#         <div><img src="data:image/png;base64,{gradcam_b64}" alt="Grad-CAM"></div>
#         <div><img src="data:image/png;base64,{ablation_b64}" alt="Ablation-CAM"></div>
#         <div><img src="data:image/png;base64,{scorecam_b64}" alt="Score-CAM"></div>
#     </div>
#     """
#
# html_content += """
# </div>
# </body>
# </html>
# """
#
# with open(html_file, "w", encoding="utf-8") as f:
#     f.write(html_content)
#
# print(f"Saved visual explanations webpage to {html_file}")
# print(f"Open this file in your browser to scroll through all images.")

# ============================================================
# 8️⃣ PLOT AND SAVE
# ============================================================

fig, axes = plt.subplots(4, 4, figsize=(14, 14))

for i in range(4):

    axes[0, i].imshow(original_images[i])
    axes[0, i].set_title("Original")

    axes[1, i].imshow(gradcam_images[i])
    axes[1, i].set_title("Grad-CAM")

    axes[2, i].imshow(ablationcam_images[i])
    axes[2, i].set_title("Ablation-CAM")

    axes[3, i].imshow(scorecam_images[i])
    axes[3, i].set_title("Score-CAM")

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()

output_name = f"visual_explanations_seed{best_seed}.pdf"
plt.savefig(output_name, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved visual explanations to {output_name}")
