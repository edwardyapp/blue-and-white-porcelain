import os
import copy
import random
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# =========================
# CONFIG
# =========================
CSV_FILE = "filtered_chinese_porcelain.csv"
IMAGE_ROOT = "images_clean"

IMAGE_SIZE = 448
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 20
RANDOM_SEED = 42

OUTPUT_MODEL = "porcelain_resnet50_object_level.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LABEL INFERENCE
# =========================
def infer_dynasty(row):
    text = str(row.get("Period", "")).lower()
    if "ming" in text:
        return "Ming"
    if "qing" in text:
        return "Qing"
    return None

# =========================
# DATASET
# =========================
class PorcelainDataset(Dataset):
    def __init__(self, samples, label_encoder, transform=None):
        self.samples = samples  # (img_path, label, object_id)
        self.le = label_encoder
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, object_id = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label_id = self.le.transform([label])[0]
        return img, label_id, object_id

# =========================
# OBJECT-LEVEL AGGREGATION
# =========================
def aggregate_majority(preds, labels, object_ids):
    obj_preds = defaultdict(list)
    obj_labels = {}

    for p, y, oid in zip(preds, labels, object_ids):
        obj_preds[oid].append(p)
        obj_labels[oid] = y

    final_preds = []
    final_labels = []

    for oid in obj_preds:
        final_preds.append(Counter(obj_preds[oid]).most_common(1)[0][0])
        final_labels.append(obj_labels[oid])

    return final_preds, final_labels

# =========================
# MAIN
# =========================
def main():
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # -------------------------
    # Load CSV + images
    # -------------------------
    df = pd.read_csv(CSV_FILE)
    samples = []

    for _, row in df.iterrows():
        object_id = str(row.get("Object ID", "")).strip()
        label = infer_dynasty(row)
        if not object_id or label is None:
            continue

        folder = os.path.join(IMAGE_ROOT, object_id)
        if not os.path.isdir(folder):
            continue

        for f in os.listdir(folder):
            if f.lower().endswith(".jpg"):
                samples.append((os.path.join(folder, f), label, object_id))

    if not samples:
        raise RuntimeError("No images found")

    df_samples = pd.DataFrame(samples, columns=["image", "label", "object_id"])

    # -------------------------
    # OBJECT-LEVEL SPLIT
    # -------------------------
    objects = df_samples[["object_id", "label"]].drop_duplicates()

    train_objs, temp_objs = train_test_split(
        objects,
        test_size=0.3,
        stratify=objects["label"],
        random_state=RANDOM_SEED
    )

    val_objs, test_objs = train_test_split(
        temp_objs,
        test_size=0.5,
        stratify=temp_objs["label"],
        random_state=RANDOM_SEED
    )

    train_ids = set(train_objs.object_id)
    val_ids = set(val_objs.object_id)
    test_ids = set(test_objs.object_id)

    train_df = df_samples[df_samples.object_id.isin(train_ids)]
    val_df   = df_samples[df_samples.object_id.isin(val_ids)]
    test_df  = df_samples[df_samples.object_id.isin(test_ids)]

    # -------------------------
    # Label encoding
    # -------------------------
    le = LabelEncoder()
    le.fit(train_df.label)
    print("Classes:", list(le.classes_))

    train_samples = list(train_df.itertuples(index=False, name=None))
    val_samples   = list(val_df.itertuples(index=False, name=None))
    test_samples  = list(test_df.itertuples(index=False, name=None))

    # -------------------------
    # Transforms
    # -------------------------
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(
        PorcelainDataset(train_samples, le, train_tf),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=8
    )

    val_loader = DataLoader(
        PorcelainDataset(val_samples, le, eval_tf),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    test_loader = DataLoader(
        PorcelainDataset(test_samples, le, eval_tf),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    # -------------------------
    # Model
    # -------------------------
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for p in model.parameters():
        p.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, len(le.classes_))
    )

    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)

    # -------------------------
    # Training
    # -------------------------
    best_state = None
    best_val = 0

    for epoch in range(EPOCHS):
        model.train()
        loss_sum = 0

        for imgs, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        # -------------------------
        # VALIDATION (OBJECT LEVEL)
        # -------------------------
        model.eval()
        preds, labels, oids = [], [], []

        with torch.no_grad():
            for imgs, y, oid in val_loader:
                imgs = imgs.to(DEVICE)
                out = model(imgs).argmax(1).cpu().numpy()
                preds.extend(out)
                labels.extend(y.numpy())
                oids.extend(oid)

        obj_preds, obj_labels = aggregate_majority(preds, labels, oids)
        acc = np.mean(np.array(obj_preds) == np.array(obj_labels))

        print(f"Epoch {epoch+1} | Train loss {loss_sum/len(train_loader):.3f} | Val obj-acc {acc:.3f}")

        if acc > best_val:
            best_val = acc
            best_state = copy.deepcopy(model.state_dict())

    # -------------------------
    # TEST (OBJECT LEVEL)
    # -------------------------
    model.load_state_dict(best_state)
    model.eval()

    preds, labels, oids = [], [], []

    with torch.no_grad():
        for imgs, y, oid in test_loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs).argmax(1).cpu().numpy()
            preds.extend(out)
            labels.extend(y.numpy())
            oids.extend(oid)

    obj_preds, obj_labels = aggregate_majority(preds, labels, oids)

    print("\n🧪 OBJECT-LEVEL TEST RESULTS")
    print(classification_report(
        le.inverse_transform(obj_labels),
        le.inverse_transform(obj_preds),
        digits=4
    ))

    # -------------------------
    # Save
    # -------------------------
    torch.save({
        "model": best_state,
        "classes": le.classes_
    }, OUTPUT_MODEL)

    print(f"Saved model → {OUTPUT_MODEL}")

# =========================
if __name__ == "__main__":
    main()
