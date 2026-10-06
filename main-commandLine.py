import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from collections import defaultdict, Counter
import numpy as np
import copy
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

import random
import math
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

EARLY_STOPPING_PATIENCE = 20 #20
EARLY_STOPPING_MIN_DELTA = 0.001
MAX_EPOCHS = 100 # large upper bound 100

import torch.multiprocessing as mp

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Porcelain Dynasty Classification Training")

    parser.add_argument(
        "--resnet",
        type=str,
        default="18",
        choices=["18", "50"],
        help="ResNet architecture to use (18 or 50)"
    )

    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze backbone parameters (default: False)"
    )

    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[0, 1],
        help="List of GPU indices to use for multiprocessing (default: [0,1])"
    )

    parser.add_argument(
        "--n_runs",
        type=int,
        default=30,
        help="Number of runs for multi-seed experiments"
    )

    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Base random seed"
    )

    args = parser.parse_args()
    return args

def run_experiment(run_id, gpu_id, base_seed, return_dict, resnet_arch, freeze_backbone):
    run_seed = base_seed + run_id
    log_file = f"results_seed{run_seed}.txt"  # you can include timestamp if you want

    # Redirect stdout to a file (per run)
    import sys
    sys.stdout = open(log_file, 'w')

    print(f"\n🚀 RUN {run_id+1} | seed={run_seed} | GPU={gpu_id}")
    metrics = main(
        seed=run_seed,
        gpu=gpu_id,
        resnet_arch=resnet_arch,
        freeze_backbone=freeze_backbone
    )
    return_dict[run_id] = metrics

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

import re

DYNASTIES = ["han", "sui", "tang", "song", "jin", "yuan", "ming", "qing"]

def extract_dynasties(text):
    return {d for d in DYNASTIES if re.search(rf"\b{d}\b", text)}

def is_ambiguous_period(period_text):
    if pd.isna(period_text):
        return False

    text = period_text.lower()

    # Explicit uncertainty
    if " or " in text:
        return True

    dynasties = extract_dynasties(text)

    # Ambiguous ONLY if more than one dynasty is mentioned
    return len(dynasties) > 1

# =========================
# CONFIG
# =========================
CSV_FILE = "filtered_chinese_porcelain.csv"
IMAGE_ROOT = "images_clean_4"

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
IMAGE_SIZE = 224
RANDOM_SEED = 42

FOCUS_ON_MING_QING = True

REMOVE_AMBIGUOUS_DYNASTIES = False

# NEW FLAG: Skip objects if dynasty is not provided
SKIP_IF_NO_DYNASTY = False

# NEW FLAG
SPLIT_BY_OBJECT = True   # True = current behavior, False = image-level split

# NEW FLAG: keep only objects whose Medium contains "blue"
FILTER_MEDIUM_BLUE = False

USE_ALTERNATIVE_AUGMENTATION = True   # ⬅️ switch ON / OFF here

# NEW FLAG: undersample Qing objects to match Ming objects
UNDERSAMPLE_QING_TO_MING = False

# NEW FLAG: filter new dataset by external object IDs file
FILTER_BY_OBJECT_ID_FILE = False
OBJECT_ID_FILE = "object-IDs.csv"  # one ID per line
# OBJECT_ID_FILE = "unique_object_ids.csv"  # one ID per line

CLASSIFY_BY_PERIOD = False   # True → predict specific periods instead of dynasty

EVALUATE_AT_OBJECT_LEVEL = True   # ⬅️ toggle here

USE_SOFTMAX_VOTING = False   # True → use softmax averaging per object; False → majority vote

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

# =========================
# NEW FLAG: Use old dataset
# =========================
USE_OLD_DATASET = False  # Set True to load old CSV + image folder
OLD_DATASET_CSV = "/home/edward-yapp/PycharmProjects/ai-for-archaeology-main/ai-for-archaeology-main/data/ai_archaeology_latest.csv"
OLD_IMAGE_ROOT = "/home/edward-yapp/PycharmProjects/ai-for-archaeology-main/ai-for-archaeology-main/data/ai_archaeology/The Metropolitan Museum of Art"

OUTPUT_MODEL_TEMPLATE = "porcelain_resnet50_seed{seed}.pth"

# CUDA_VISIBLE_DEVICES=1

allowed_object_ids = None
if FILTER_BY_OBJECT_ID_FILE:
    if not os.path.exists(OBJECT_ID_FILE):
        raise FileNotFoundError(f"Object ID file not found: {OBJECT_ID_FILE}")

    with open(OBJECT_ID_FILE, "r") as f:
        allowed_object_ids = set(line.strip() for line in f if line.strip())

    print(f"⚡ Filtering new dataset by {len(allowed_object_ids)} object IDs from {OBJECT_ID_FILE}")

def medium_contains_blue(medium):
    if pd.isna(medium):
        return False
    return "blue" in str(medium).lower()

# =========================
# Small classifier head on ResNet backbone
# =========================
# class SmallResNetClassifier(nn.Module):
#     def __init__(self, num_classes=2, freeze_backbone=True):
#         super().__init__()
#         # Load pretrained ResNet50
#         self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
#
#         # Freeze backbone if desired
#         if freeze_backbone:
#             for param in self.backbone.parameters():
#                 param.requires_grad = False
#
#         # Remove the original classifier
#         in_features = self.backbone.fc.in_features
#         self.backbone.fc = nn.Identity()  # output feature map
#
#         # Small trainable head
#         # self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # global spatial average
#         self.dropout = nn.Dropout(0.2)
#         self.classifier = nn.Linear(in_features, num_classes)  # small head
#
#     def forward(self, x):
#         x = self.backbone(x)               # shape: (batch, 2048, 1, 1) after avgpool?
#         # x = self.global_avg_pool(x)        # shape: (batch, 2048, 1, 1)
#         x = torch.flatten(x, 1)            # shape: (batch, 2048)
#         x = self.dropout(x)
#         x = self.classifier(x)             # shape: (batch, num_classes)
#         return x

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.02):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)

def aggregate_object_predictions(preds, labels, object_ids):
    """
    Majority vote aggregation per object.
    """
    obj_pred_map = defaultdict(list)
    obj_label_map = {}

    for p, y, oid in zip(preds, labels, object_ids):
        obj_pred_map[oid].append(p)
        obj_label_map[oid] = y  # same label for all images of object

    final_preds = []
    final_labels = []

    for oid in obj_pred_map:
        final_preds.append(Counter(obj_pred_map[oid]).most_common(1)[0][0])
        final_labels.append(obj_label_map[oid])

    return final_preds, final_labels


def aggregate_object_predictions_majority_verbose(
    preds_list, labels, object_ids, filenames, class_names
):
    obj_map = defaultdict(list)
    obj_label_map = {}

    for pred, y, oid, fname in zip(preds_list, labels, object_ids, filenames):
        obj_map[oid].append((pred, fname))
        obj_label_map[oid] = y

    final_preds, final_labels = [], []
    details = {}

    for oid, items in obj_map.items():
        preds = [p for p, _ in items]
        vote_counts = Counter(preds)

        # majority vote (deterministic tie break)
        pred = vote_counts.most_common(1)[0][0]
        true = obj_label_map[oid]

        details[oid] = {
            "true_label": class_names[true],
            "pred_label": class_names[pred],
            "num_images": len(items),
            "vote_counts": {
                class_names[k]: v for k, v in vote_counts.items()
            },
            "files": [
                {
                    "filename": fname,
                    "pred": class_names[p]
                }
                for p, fname in items
            ]
        }

        final_preds.append(pred)
        final_labels.append(true)

    return final_preds, final_labels, details

def aggregate_object_predictions_softmax(probs_list, labels, object_ids):
    """
    Softmax/probabilistic voting per object.
    """
    obj_probs_map = defaultdict(list)
    obj_label_map = {}

    for probs, y, oid in zip(probs_list, labels, object_ids):
        obj_probs_map[oid].append(probs)
        obj_label_map[oid] = y

    final_preds = []
    final_labels = []

    for oid in obj_probs_map:
        avg_probs = np.mean(obj_probs_map[oid], axis=0)
        final_preds.append(np.argmax(avg_probs))
        final_labels.append(obj_label_map[oid])

    return final_preds, final_labels

def aggregate_object_predictions_softmax_verbose(
    probs_list, labels, object_ids, filenames, class_names
):
    obj_map = defaultdict(list)
    obj_label_map = {}

    for probs, y, oid, fname in zip(probs_list, labels, object_ids, filenames):
        obj_map[oid].append((probs, fname))
        obj_label_map[oid] = y

    final_preds, final_labels = [], []
    details = {}

    for oid, items in obj_map.items():
        probs_stack = np.array([p for p, _ in items])
        avg_probs = probs_stack.mean(axis=0)

        pred = int(np.argmax(avg_probs))
        true = obj_label_map[oid]

        details[oid] = {
            "true_label": class_names[true],
            "pred_label": class_names[pred],
            "num_images": len(items),
            "avg_probs": {
                class_names[i]: float(avg_probs[i])
                for i in range(len(avg_probs))
            },
            "files": [
                {
                    "filename": fname,
                    "pred": class_names[int(np.argmax(p))],
                    "probs": {
                        class_names[i]: float(p[i])
                        for i in range(len(p))
                    }
                }
                for p, fname in items
            ]
        }

        final_preds.append(pred)
        final_labels.append(true)

    return final_preds, final_labels, details

# =========================
# DATASET
# =========================
class PorcelainDataset(Dataset):
    def __init__(self, samples, label_encoder, transform=None):
        self.samples = samples
        self.le = label_encoder
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, object_id = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label_id = self.le.transform([label])[0]
        filename = os.path.basename(img_path)

        return image, label_id, object_id, filename   # ⬅️ keep object_id

# =========================
# MAIN
# =========================
def main(seed, gpu, resnet_arch="18", freeze_backbone=False):
    seed_everything(seed)
    torch.backends.cudnn.benchmark = True  # Optimize convs for fixed input size
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------
    # Load CSV
    # -------------------------
    samples = []

    if USE_OLD_DATASET:
        print("⚠️ Using OLD dataset")
        df = pd.read_csv(OLD_DATASET_CSV)
        df = df[df['Museum'] == "The Metropolitan Museum of Art"]
        IMAGE_ROOT_USED = OLD_IMAGE_ROOT

        # Optional filter: Ming / Qing + blue medium if desired
        filtered_df = df[df['Dynasty'].isin(["Ming", "Qing"])]

        filtered_df = filtered_df[filtered_df['Category'].str.contains("blue", case=False, na=False)]

        for _, row in filtered_df.iterrows():
            dynasty = row['Dynasty']
            image_file = row['Image ID']  # assuming CSV has the exact filename
            if "bottom" in image_file.lower():
                continue

            img_path = os.path.join(IMAGE_ROOT_USED, image_file)
            if not os.path.exists(img_path):
                continue

            # Extract numeric object ID from filename (e.g., Sample229_frontview.png → 229)
            match = re.search(r"Sample(\d+)", image_file, re.IGNORECASE)
            if match:
                object_id = match.group(1)
            else:
                object_id = os.path.splitext(image_file)[0]  # fallback

            # ✅ Extract object ID from Met Museum link
            # link = row['Link']
            # match = re.search(r'/search/(\d+)', link)
            # if not match:
            #     continue
            # object_id = match.group(1)

            # ⚡ Apply external object ID filter
            if FILTER_BY_OBJECT_ID_FILE and object_id not in allowed_object_ids:
                continue

            samples.append((img_path, dynasty, object_id))
    else:
        df = pd.read_csv(CSV_FILE)
        IMAGE_ROOT_USED = IMAGE_ROOT

        if SKIP_IF_NO_DYNASTY:
            print(f"\n⚠️ SKIPPING MISSING DYNASTY ENTRIES")

        for _, row in df.iterrows():
            object_id = str(row.get("Object ID", "")).strip()

            if CLASSIFY_BY_PERIOD:
                label = infer_period(row)
            else:
                label = infer_dynasty(row)

            # Skip objects without recognized label
            if not object_id or label is None:
                continue

            # ⚡ Apply external object ID filter
            if FILTER_BY_OBJECT_ID_FILE and object_id not in allowed_object_ids:
                continue

            folder = os.path.join(IMAGE_ROOT_USED, object_id)
            if not os.path.isdir(folder):
                continue

            medium = row.get("Medium")

            # All
            for f in os.listdir(folder):
                if f.lower().endswith(".jpg"):
                    samples.append((os.path.join(folder, f), label, object_id, medium))

            # Primary only
            # primary_path = os.path.join(folder, "primary.jpg")
            #
            # if os.path.exists(primary_path):
            #     samples.append((primary_path, label, object_id))

            # All but bottom
            # for fname in os.listdir(folder):
            #     if not fname.lower().endswith(".jpg"):
            #         continue
            #
            #     if "bottom" in fname.lower():
            #         continue
            #
            #     img_path = os.path.join(folder, fname)
            #     samples.append((img_path, label, object_id))

            # Primary and bottom
            # for fname in os.listdir(folder):
            #     name = fname.lower()
            #
            #     if not name.endswith(".jpg"):
            #         continue
            #
            #     if ("primary" not in name) and ("bottom" not in name):
            #         continue
            #
            #     img_path = os.path.join(folder, fname)
            #     samples.append((img_path, label, object_id))

    if not samples:
        raise RuntimeError("No valid images found.")

    df_samples = pd.DataFrame(samples, columns=["image", "label", "object_id", "medium"])

    # -------------------------
    # Optional filter: Medium contains "blue"
    # -------------------------
    if FILTER_MEDIUM_BLUE:
        print("\n🎨 FILTER ENABLED: Medium contains 'blue'")

        # Map Object ID → Medium
        df_medium = df[["Object ID", "Medium"]].copy()
        df_medium["Object ID"] = df_medium["Object ID"].astype(str)

        blue_object_ids = set(
            df_medium[df_medium["Medium"].apply(medium_contains_blue)]["Object ID"]
        )

        before = len(df_samples)
        df_samples = df_samples[df_samples["object_id"].isin(blue_object_ids)]
        after = len(df_samples)

        print(f"Kept {after} / {before} images from blue-medium objects")

    # -------------------------
    # Optional removal of ambiguous dynasty entries
    # -------------------------
    if REMOVE_AMBIGUOUS_DYNASTIES:
        df_period = df[["Object ID", "Period"]].copy()
        df_period["Object ID"] = df_period["Object ID"].astype(str)

        ambiguous_ids = set(
            df_period[df_period["Period"].apply(is_ambiguous_period)]["Object ID"]
        )

        before = len(df_samples)
        df_samples = df_samples[~df_samples["object_id"].isin(ambiguous_ids)]
        after = len(df_samples)

        print(f"\n⚠️ REMOVED AMBIGUOUS DYNASTY ENTRIES")
        print(f"Removed {before - after} images from ambiguous periods")

    # -------------------------
    # Optional focus: Ming vs Qing only
    # -------------------------
    if not CLASSIFY_BY_PERIOD and FOCUS_ON_MING_QING:
        keep_labels = ["Ming", "Qing"]
        df_samples = df_samples[df_samples["label"].isin(keep_labels)]

        print("\n⚠️ FOCUS MODE ENABLED: Ming vs Qing only")
        print("Image counts:")
        print(df_samples["label"].value_counts())

    # -------------------------
    # Object-level filtering
    # -------------------------
    # -------------------------
    # Object-level filtering
    # -------------------------
    unique_objects = df_samples[["object_id", "label"]].drop_duplicates()

    # Set minimum number of objects per class (for stratified splitting)
    min_objects_per_class = 6 # adjust as needed
    label_counts = unique_objects["label"].value_counts()
    valid_labels = label_counts[label_counts >= min_objects_per_class].index

    unique_objects = unique_objects[unique_objects["label"].isin(valid_labels)]
    df_samples = df_samples[df_samples["label"].isin(valid_labels)]

    print("Final class counts (objects):")
    print(unique_objects["label"].value_counts())

    # -------------------------
    # Optional undersampling: Qing → Ming (OBJECT LEVEL)
    # -------------------------
    if UNDERSAMPLE_QING_TO_MING and FOCUS_ON_MING_QING:
        print("\n⚖️ UNDERSAMPLING ENABLED: Qing → Ming (object-level)")

        # Count objects per class
        obj_counts = unique_objects["label"].value_counts()

        if "Ming" in obj_counts and "Qing" in obj_counts:
            n_ming = obj_counts["Ming"]
            n_qing = obj_counts["Qing"]

            if n_qing > n_ming:
                print(f"Reducing Qing objects from {n_qing} → {n_ming}")

                ming_objs = unique_objects[unique_objects["label"] == "Ming"]
                qing_objs = unique_objects[unique_objects["label"] == "Qing"].sample(
                    n=n_ming,
                    random_state=RANDOM_SEED
                )

                balanced_objects = pd.concat([ming_objs, qing_objs])

                # Keep only selected objects
                keep_ids = set(balanced_objects["object_id"])
                df_samples = df_samples[df_samples["object_id"].isin(keep_ids)]
                unique_objects = balanced_objects

            else:
                print("No undersampling needed (Qing ≤ Ming)")

        else:
            print("⚠️ Cannot undersample — Ming or Qing missing")

        print("Balanced object counts:")
        print(unique_objects["label"].value_counts())

    # -------------------------
    # Train / Val / Test split
    # -------------------------
    if SPLIT_BY_OBJECT:
        print("🔒 Using OBJECT-AWARE split")

        unique_objects = df_samples[["object_id", "label"]].drop_duplicates()

        # train_objs, temp_objs = train_test_split(
        #     unique_objects,
        #     test_size=0.3,
        #     stratify=unique_objects["label"],
        #     random_state=RANDOM_SEED
        # )

        # Stratified split for test set
        train_val_objs, test_objs = train_test_split(
            unique_objects,
            test_size=0.3,  # 20% test
            stratify=unique_objects["label"],
            random_state=RANDOM_SEED  # fixed for all runs
        )

        # train_val_ids = set(train_val_objs["object_id"])
        test_ids = set(test_objs["object_id"])

        train_objs, val_objs = train_test_split(
            train_val_objs,
            test_size=0.5,
            stratify=train_val_objs["label"],
            random_state=seed
        )

        train_ids = set(train_objs["object_id"])
        val_ids = set(val_objs["object_id"])

        train_samples = df_samples[df_samples["object_id"].isin(train_ids)]
        val_samples = df_samples[df_samples["object_id"].isin(val_ids)]
        test_samples = df_samples[df_samples["object_id"].isin(test_ids)]

    else:
        print("⚠️ Using IMAGE-LEVEL (non-object-aware) split")

        train_samples, temp_samples = train_test_split(
            df_samples,
            test_size=0.3,
            stratify=df_samples["label"],
            random_state=RANDOM_SEED
        )

        val_samples, test_samples = train_test_split(
            temp_samples,
            test_size=0.5,
            stratify=temp_samples["label"],
            random_state=RANDOM_SEED
        )

    # Convert to (image, label) tuples
    train_samples = list(train_samples[["image", "label", "object_id"]].itertuples(index=False, name=None))
    val_samples = list(val_samples[["image", "label", "object_id"]].itertuples(index=False, name=None))
    test_samples = list(test_samples[["image", "label", "object_id"]].itertuples(index=False, name=None))

    # -------------------------
    # Label encoding
    # -------------------------
    le = LabelEncoder()
    le.fit([label for _, label, _ in train_samples])
    print("Final classes:", list(le.classes_))

    # -------------------------
    # Class-weighted loss based on entire dataset
    # -------------------------
    if not UNDERSAMPLE_QING_TO_MING:
        all_labels = df_samples["label"]
        label_freq = all_labels.value_counts()
        # train_labels = [label for _, label in train_samples]
        # label_freq = pd.Series(train_labels).value_counts()

        weights = 1.0 / label_freq
        weights = weights / weights.sum()
        class_weights = torch.tensor(
            [weights[c] for c in le.classes_],
            dtype=torch.float32
        ).to(device)

        print("Class weights:")
        for c, w in zip(le.classes_, class_weights):
            print(f"{c}: {w.item():.4f}")

    # -------------------------
    # Transforms & DataLoaders
    # -------------------------
    if USE_ALTERNATIVE_AUGMENTATION:
        print("⚠️ ALTERNATIVE AUGMENTATION ENABLED")

        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        val_tf = transforms.Compose([
            transforms.Resize(2 ** math.ceil(math.log2(IMAGE_SIZE))),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        print("ℹ️ BASELINE AUGMENTATION ENABLED")

        train_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        val_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        PorcelainDataset(train_samples, le, train_tf),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    val_loader = DataLoader(
        PorcelainDataset(val_samples, le, val_tf),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    test_loader = DataLoader(
        PorcelainDataset(test_samples, le, val_tf),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    # -------------------------
    # Model
    # -------------------------
    if resnet_arch== "18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif resnet_arch== "50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unsupported ResNet: {resnet_arch}")

    # Freeze backbone if requested
    if freeze_backbone:
        print("🔒 Freezing backbone parameters")
        for p in model.parameters():
            p.requires_grad = False

    # Replace classifier
    # model.fc = nn.Sequential(
    #     nn.Linear(model.fc.in_features, 256),
    #     nn.ReLU(),
    #     nn.Dropout(0.5),
    #     nn.Linear(256, len(le.classes_))
    # )

    # Replace classifier head
    num_classes = len(le.classes_)  # your label encoder classes
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )

    # Ensure classifier is trainable
    for p in model.fc.parameters():
        p.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    model.to(device)

    if UNDERSAMPLE_QING_TO_MING:
        print("ℹ️ Using unweighted CrossEntropyLoss (balanced dataset)")
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    # -------------------------
    # Early Stopping Setup
    # -------------------------
    class EarlyStopping:
        def __init__(self, patience=10, min_delta=0.0):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_loss = float("inf")
            self.early_stop = False

        def step(self, val_loss):
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

    early_stopper = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA
    )


    # -------------------------
    # Training Loop
    # -------------------------
    best_model_state = None
    best_epoch = None
    best_val_report = None
    best_val_acc = None

    for epoch in range(MAX_EPOCHS):
        model.train()
        running_loss = 0.0

        for imgs, labels, _ , _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}"):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        running_loss /= len(train_loader)

        # -------------------------
        # Compute training predictions for object-level accuracy
        # -------------------------
        model.eval()
        train_all_labels = []
        train_all_oids = []

        if USE_SOFTMAX_VOTING:
            train_all_probs = []
        else:
            train_all_preds = []

        with torch.no_grad():
            for imgs, labels, oids, _ in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)

                train_all_labels.extend(labels.cpu().numpy())
                train_all_oids.extend(oids)

                if USE_SOFTMAX_VOTING:
                    probs = torch.softmax(outputs, dim=1)
                    train_all_probs.extend(probs.cpu().numpy())
                else:
                    preds = outputs.argmax(1)
                    train_all_preds.extend(preds.cpu().numpy())

        # Object-level aggregation if desired
        if EVALUATE_AT_OBJECT_LEVEL:
            if USE_SOFTMAX_VOTING:
                obj_preds, obj_labels = aggregate_object_predictions_softmax(
                    train_all_probs, train_all_labels, train_all_oids
                )
            else:
                obj_preds, obj_labels = aggregate_object_predictions(
                    train_all_preds, train_all_labels, train_all_oids
                )
            train_acc = np.mean(np.array(obj_preds) == np.array(obj_labels))
        else:
            train_acc = np.mean(np.array(train_all_preds) == np.array(train_all_labels))

        from sklearn.metrics import classification_report

        # Validation
        model.eval()
        val_loss = 0.0
        # correct = total = 0
        # all_preds = []
        all_labels = []
        all_oids = []

        if USE_SOFTMAX_VOTING:
            all_probs = []  # store softmax probabilities
        else:
            all_preds = []

        with torch.no_grad():
            for imgs, labels, oids, fnames in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)

                if USE_SOFTMAX_VOTING:
                    probs = torch.softmax(outputs, dim=1)
                    all_probs.extend(probs.cpu().numpy())
                else:
                    preds = outputs.argmax(1)
                    all_preds.extend(preds.cpu().numpy())

                val_loss += criterion(outputs, labels).item()

                # correct += (preds == labels).sum().item()
                # total += labels.size(0)

                # Collect predictions and labels for per-class metrics
                all_labels.extend(labels.cpu().numpy())
                all_oids.extend(oids)

        val_loss /= len(val_loader)
        # val_acc = correct / total

        # -------------------------
        # EVALUATION MODE SWITCH
        # -------------------------
        if EVALUATE_AT_OBJECT_LEVEL:
            if USE_SOFTMAX_VOTING:
                obj_preds, obj_labels = aggregate_object_predictions_softmax(
                    all_probs, all_labels, all_oids
                )
            else:
                obj_preds, obj_labels = aggregate_object_predictions(
                    all_preds, all_labels, all_oids
                )

            val_acc = np.mean(np.array(obj_preds) == np.array(obj_labels))

            report = classification_report(
                le.inverse_transform(obj_labels),
                le.inverse_transform(obj_preds),
                digits=4
            )
        else:
            val_acc = np.mean(np.array(all_preds) == np.array(all_labels))

            report = classification_report(
                le.inverse_transform(all_labels),
                le.inverse_transform(all_preds),
                digits=4
            )

        # Print per-class metrics
        # print(f"Epoch {epoch + 1}: train_loss={running_loss:.3f}, val_loss={val_loss:.3f}, val_acc={val_acc:.3f}")
        print(
            f"Epoch {epoch + 1}: train_loss={running_loss:.3f}, train_acc={train_acc:.3f}, val_loss={val_loss:.3f}, val_acc={val_acc:.3f}")

        print("Per-class metrics:\n", report)

        # Early stopping check
        if early_stopper.best_loss is None or val_loss < early_stopper.best_loss:
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            best_val_acc = val_acc
            best_val_report = report

        early_stopper.step(val_loss)

        if early_stopper.early_stop:
            print("\n🏆 BEST MODEL (Early Stopped)")
            print(f"Epoch: {best_epoch}")
            print(f"Validation Accuracy: {best_val_acc:.4f}")
            print("Validation Classification Report:")
            print(best_val_report)
            break

    # -------------------------
    # Save best model
    # -------------------------
    output_model_path = OUTPUT_MODEL_TEMPLATE.format(seed=seed)

    torch.save({
        "model_state_dict": best_model_state,
        "label_classes": le.classes_,
        "label_column": "Derived Dynasty",
        "run_seed": seed,
        "best_epoch": best_epoch,
        "val_accuracy": best_val_acc
    }, output_model_path)

    print(f"💾 Model saved to {output_model_path}")

    # -------------------------
    # Evaluate on Test Set
    # -------------------------
    from sklearn.metrics import classification_report
    model.load_state_dict(best_model_state)
    model.eval()

    # all_test_preds = []
    # all_test_labels = []

    # test_preds = []
    test_labels = []
    test_oids = []
    test_fnames = []

    if USE_SOFTMAX_VOTING:
        test_probs = []  # store softmax probabilities per image
    else:
        test_preds = []  # store argmax predictions per image

    with torch.no_grad():
        for imgs, labels, oids, fnames in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            # preds = model(imgs).argmax(1)
            outputs = model(imgs)

            # test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            test_oids.extend(oids)
            test_fnames.extend(fnames)

            if USE_SOFTMAX_VOTING:
                probs = torch.softmax(outputs, dim=1)  # softmax probabilities
                test_probs.extend(probs.cpu().numpy())
            else:
                preds = outputs.argmax(1)
                test_preds.extend(preds.cpu().numpy())

            # all_test_preds.extend(preds.cpu().numpy())
            # all_test_labels.extend(labels.cpu().numpy())

    class_names = le.classes_

    if EVALUATE_AT_OBJECT_LEVEL:
        if USE_SOFTMAX_VOTING:
            # obj_preds, obj_labels = aggregate_object_predictions_softmax(
            #     test_probs, test_labels, test_oids
            # )
            obj_preds, obj_labels, details = aggregate_object_predictions_softmax_verbose(
                test_probs, test_labels, test_oids, test_fnames, class_names
            )
        else:
            # obj_preds, obj_labels = aggregate_object_predictions(
            #     test_preds, test_labels, test_oids
            # )
            obj_preds, obj_labels, details = aggregate_object_predictions_majority_verbose(
                test_preds, test_labels, test_oids, test_fnames, le.classes_
            )

        print("\n🧪 OBJECT-LEVEL TEST RESULTS")
        print(classification_report(
            le.inverse_transform(obj_labels),
            le.inverse_transform(obj_preds),
            digits=4
        ))

        print("\n❌ INCORRECT OBJECTS (WITH IMAGE VOTES):")

        for oid, info in details.items():
            if info["true_label"] != info["pred_label"]:
                print(f"\nObject ID: {oid}")
                print(f"  True: {info['true_label']}")
                print(f"  Pred: {info['pred_label']}")
                print(f"  Images: {info['num_images']}")

                if USE_SOFTMAX_VOTING:
                    print("  Avg softmax:")
                    for cls, p in sorted(
                            info["avg_probs"].items(), key=lambda x: -x[1]
                    ):
                        print(f"    {cls}: {p:.4f}")
                else:
                    print("  Vote counts:")
                    for cls, v in sorted(
                            info["vote_counts"].items(), key=lambda x: -x[1]
                    ):
                        print(f"    {cls}: {v}")

                print("  Image-level votes:")
                for f in info["files"]:
                    print(f"    {f['filename']} → {f['pred']}")

        test_acc = accuracy_score(obj_labels, obj_preds)
        test_precision_macro = precision_score(
            obj_labels, obj_preds, average="macro", zero_division=0
        )
        test_recall_macro = recall_score(
            obj_labels, obj_preds, average="macro", zero_division=0
        )
        test_f1_macro = f1_score(obj_labels, obj_preds, average="macro")
        # test_f1_weighted = f1_score(obj_labels, obj_preds, average="weighted")

        return {
            "accuracy": test_acc,
            "precision_macro": test_precision_macro,
            "recall_macro": test_recall_macro,
            "f1_macro": test_f1_macro
        }
    else:
        print("\n🧪 IMAGE-LEVEL TEST RESULTS")
        print(classification_report(
            le.inverse_transform(test_labels),
            le.inverse_transform(test_preds),
            digits=4
        ))

        test_acc = accuracy_score(test_labels, test_preds)
        test_precision_macro = precision_score(
            test_labels, test_preds, average="macro", zero_division=0
        )
        test_recall_macro = recall_score(
            test_labels, test_preds, average="macro", zero_division=0
        )
        test_f1_macro = f1_score(test_labels, test_preds, average="macro")
        # test_f1_weighted = f1_score(test_labels, test_preds, average="weighted")

        return {
            "accuracy": test_acc,
            "precision_macro": test_precision_macro,
            "recall_macro": test_recall_macro,
            "f1_macro": test_f1_macro
        }

if __name__ == "__main__":
    args = parse_args()

    BASE_SEED = args.base_seed
    GPUS = args.gpus

    mp.set_start_method("spawn", force=True)

    manager = mp.Manager()
    return_dict = manager.dict()
    MAX_CONCURRENT = len(GPUS)  # 2 in your case
    processes = []

    for run_id in range(args.n_runs):
        gpu_id = GPUS[run_id % len(GPUS)]  # alternate GPU assignment
        p = mp.Process(
            target=run_experiment,
            args=(run_id, gpu_id, args.base_seed, return_dict, args.resnet, args.freeze_backbone)
        )
        p.start()
        processes.append(p)

        # Wait if max concurrent processes reached
        if len(processes) == MAX_CONCURRENT:
            for proc in processes:
                proc.join()
            processes = []

    for p in processes:
        p.join()

    # Collect results
    df = pd.DataFrame([return_dict[i] for i in range(args.n_runs)])
    df.to_csv("multi_seed_results.csv", index=False)

    # -------------------------
    # Print final summary
    # -------------------------
    print("\n📊 FINAL TEST RESULTS (mean ± std)")
    for col in df.columns:
        print(f"{col}: {df[col].mean():.4f} ± {df[col].std():.4f}")

    # -------------------------
    # Save final summary to a file
    # -------------------------
    summary_file = "final_results_summary.txt"
    with open(summary_file, "w") as f:
        f.write("📊 FINAL TEST RESULTS (mean ± std)\n")
        for col in df.columns:
            f.write(f"{col}: {df[col].mean():.4f} ± {df[col].std():.4f}\n")

    print(f"\n💾 Final summary saved to {summary_file}")