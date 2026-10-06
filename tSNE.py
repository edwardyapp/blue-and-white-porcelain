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

import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm

# ----------------------------
# Image preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------------------
# Feature extractor (ResNet50 backbone)
# ----------------------------
feature_model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove final FC
feature_model.eval()
feature_model = feature_model.to(device)

# ----------------------------
# Extract features
# ----------------------------
features = []
labels = []
pred_correct = []
image_paths = []

for img_path, label, obj_id, _ in tqdm(test_samples[["image", "label", "object_id", "medium"]].values):
    try:
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = feature_model(x)  # (1, 2048, 1, 1)
            feat = feat.flatten(1).cpu().numpy()[0]

        features.append(feat)
        labels.append(label)
        image_paths.append(img_path)

        # Model prediction for correctness
        with torch.no_grad():
            logits = model(x)
            pred_label = "Ming" if torch.argmax(logits, dim=1).item() == 0 else "Qing"
            pred_correct.append(pred_label == label)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

features = np.array(features)
labels = np.array(labels)
pred_correct = np.array(pred_correct)

# ----------------------------
# PCA + t-SNE
# ----------------------------
# pca = PCA(random_state=42)
pca = PCA(n_components=50, random_state=42)
features_pca = pca.fit_transform(features)

tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
embeddings = tsne.fit_transform(features_pca)

def frame_image(img, frame_width, frame_color, correct=True):
    """
    Adds a border around the image.
    Solid border if correct=True
    Dotted border if correct=False
    """
    b = frame_width
    ny, nx = img.shape[0], img.shape[1]

    frame_color = (np.array(frame_color[:3]) * 255).astype(int)

    framed_img = np.zeros((ny + 2*b, nx + 2*b, 3), dtype=np.uint8)

    # Fill border
    for i in range(ny + 2*b):
        for j in range(nx + 2*b):

            border_pixel = (
                i < b or i >= ny + b or
                j < b or j >= nx + b
            )

            if border_pixel:
                if correct:
                    # Solid border
                    framed_img[i, j] = frame_color
                else:
                    # Dotted border pattern
                    if (i + j) % 6 < 3:
                        framed_img[i, j] = frame_color
                    else:
                        framed_img[i, j] = [255, 255, 255]

    # Insert original image
    framed_img[b:-b, b:-b] = img

    return framed_img

def visualize_with_images_and_borders(
        tsne_results,
        image_paths,
        labels,
        pred_correct,
        figsize=(20, 20),
        image_size=(50, 50),
        frame_width=5,
        zoom=0.8,
        save_path="tsne_dynasty_correctness.pdf"
):
    dynasty_colors = {
        "Ming": (0, 0, 1),     # blue
        "Qing": (1, 0.5, 0)    # orange
    }

    plt.figure(figsize=figsize)
    ax = plt.gca()

    for (x, y), image_path, label, correct in zip(
            tsne_results, image_paths, labels, pred_correct):

        try:
            img = Image.open(image_path).convert("RGB")
            img = img.resize(image_size, Image.Resampling.LANCZOS)
            img = np.array(img)

            frame_color = dynasty_colors[label]

            framed_img = frame_image(
                img,
                frame_width,
                frame_color,
                correct=correct
            )

            im = OffsetImage(framed_img, zoom=zoom)
            ab = AnnotationBbox(im, (x, y), frameon=False)
            ax.add_artist(ab)

            ax.update_datalim([(x, y)])

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

    ax.autoscale()

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="blue", lw=4, linestyle="solid", label="Ming - correct"),
        Line2D([0], [0], color="blue", lw=4, linestyle="dotted", label="Ming - wrong"),
        Line2D([0], [0], color="orange", lw=4, linestyle="solid", label="Qing - correct"),
        Line2D([0], [0], color="orange", lw=4, linestyle="dotted", label="Qing - wrong")
    ]

    ax.legend(handles=legend_elements, loc="upper right", fontsize=25)

    plt.axis('off')
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()

# ----------------------------
# t-SNE parameter sweeps
# ----------------------------
perplexities = [30]
iterations_list = [1000]

for perp in perplexities:
    for n_iter in iterations_list:

        print(f"\nRunning t-SNE | Perplexity={perp} | Iter={n_iter}")

        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            max_iter=n_iter,
            init='pca',              # important for stability
            random_state=42,
            learning_rate='auto'
        )

        embeddings = tsne.fit_transform(features_pca)

        save_name = f"tsne_perp{perp}_iter{n_iter}.pdf"

        visualize_with_images_and_borders(
            tsne_results=embeddings,
            image_paths=image_paths,
            labels=labels,
            pred_correct=pred_correct,
            save_path=save_name
        )
