import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# =====================
# CONFIG
# =====================
CSV_FILE = "filtered_chinese_porcelain.csv"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# OBJECT GROUPING
# =====================
OBJECT_GROUPS = {
    "VASE_FORM": ["Vase", "Bottle vase", "Miniature vase", "Covered vase", "Jardinière"],
    "JAR_FORM": ["Jar", "Covered jar", "Jar with cover", "Covered tureen"],
    "BOWL_FORM": ["Bowl", "Footed bowl", "Stem bowl", "Altar bowl", "Covered bowl"],
    "DISH_PLATE": ["Dish", "Plate", "Saucer", "Large dish"],
    "CUP_FORM": ["Cup", "Stem cup", "Covered cup", "Pair of stem cups", "Cups"],
    "BOTTLE_FLASK": ["Bottle", "Flask", "Pilgrim bottle", "Gallipot"],
    "EWER_POT": [
        "Ewer", "Teapot", "Winepot", "Coffeepot", "Jug",
        "Covered winepot or teapot", "Winepot or teapot"
    ],
    "BOX_CONTAINER": ["Box", "Covered box", "Seal paste box", "Ink well (?)"],
    "INCENSE_RITUAL": ["Incense burner", "Censer", "Cistern"],
    "WATER_SPRINKLER": ["Sprinkler", "Rosewater sprinkler", "Water sprinkler"],
    "BRUSH_OBJECT": ["Brush holder", "Brush pot", "Brush rest", "Brush"],
    "FIGURINE": ["Figure", "Garden seat", "Bird feeder"],
    "VESSEL_OTHER": ["Vessel", "Kendi", "Water pot", "Water coupe"],
    "FRAGMENT": ["Shard", "Fragment"]
}

# =====================
# DERIVE DYNASTY / PERIOD
# =====================
def infer_dynasty(row):
    text = str(row.get("Period", "")).lower()

    if "ming" in text:
        return "Ming"
    if "qing" in text:
        return "Qing"

    begin = row.get("Object Begin Date")
    end = row.get("Object End Date")
    year = begin if pd.notna(begin) else end

    if pd.isna(year):
        return None

    year = int(year)
    if 1368 <= year <= 1644:
        return "Ming"
    if 1644 <= year <= 1912:
        return "Qing"

    return None

def map_object_group(name):
    name = str(name).strip()
    for group, keywords in OBJECT_GROUPS.items():
        if name in keywords:
            return group
    return "OTHER"

# =====================
# TAG GROUPING
# =====================
TAG_GROUPS = {
    "FLORA": ["Flowers", "Leaves", "Trees", "Plants", "Bamboo", "Lotuses", "Peonies", "Vines", "Fruit"],
    "FAUNA": ["Birds", "Fish", "Horses", "Deer", "Dogs", "Butterflies", "Crabs", "Rabbits"],
    "MYTHICAL": ["Dragons", "Phoenix", "Mythical Creatures", "Bats"],
    "HUMAN": ["Men", "Women", "Boys", "Human Figures", "Couples", "Warriors", "Children"],
    "LANDSCAPE": ["Landscapes", "Hills", "Mountains", "Gardens", "Villages"],
    "ARCHITECTURE": ["Houses", "Buildings", "Boats", "Bridges", "Pavilions"],
    "OBJECTS": ["Saucers", "Ewers", "Fans", "Boxes", "Chairs", "Musical Instruments"],
    "RELIGION": ["Buddhism", "Bodhisattvas", "Daoism"],
    "ACTIVITY": ["Fishing", "Playing", "Working", "Games", "Archery"],
    "SYMBOLIC": ["Coat of Arms", "Genre Scene"]
}

def map_tags(tag_string):
    raw_tags = str(tag_string).split("|")
    groups = set()

    for tag in raw_tags:
        tag = tag.strip()
        for group, keywords in TAG_GROUPS.items():
            if tag in keywords:
                groups.add(group)

    return list(groups)

# =====================
# LOAD METADATA
# =====================
df = pd.read_csv(CSV_FILE)

# ---- Dynasty ----
dynasty_encoder = LabelEncoder()
df["dynasty_id"] = dynasty_encoder.fit_transform(df["Period"])

# ---- Object (grouped) ----
df["object_group"] = df["Object Name"].apply(map_object_group)
object_encoder = LabelEncoder()
df["object_id"] = object_encoder.fit_transform(df["object_group"])

# ---- Tags (grouped, multi-label) ----
df["grouped_tags"] = df["Tags"].apply(map_tags)
tag_encoder = MultiLabelBinarizer()
tags_encoded = tag_encoder.fit_transform(df["grouped_tags"])

# =====================
# DATASET
# =====================
class PorcelainDataset(Dataset):
    def __init__(self, df, tags_encoded, transform=None):
        self.df = df.reset_index(drop=True)
        self.tags = torch.tensor(tags_encoded, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row.image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "dynasty": torch.tensor(row.dynasty_id),
            "object": torch.tensor(row.object_id),
            "tags": self.tags[idx]
        }

# =====================
# TRANSFORMS
# =====================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =====================
# TRAIN / VAL SPLIT
# =====================
train_df, val_df, train_tags, val_tags = train_test_split(
    df, tags_encoded, test_size=0.2, random_state=42
)

train_ds = PorcelainDataset(train_df, train_tags, transform)
val_ds   = PorcelainDataset(val_df, val_tags, transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

# =====================
# MODEL
# =====================
class MultiTaskCNN(nn.Module):
    def __init__(self, num_dynasties, num_objects, num_tags):
        super().__init__()
        self.backbone = models.resnet50(weights="IMAGENET1K_V2")
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.dynasty_head = nn.Linear(feat_dim, num_dynasties)
        self.object_head  = nn.Linear(feat_dim, num_objects)
        self.tags_head    = nn.Linear(feat_dim, num_tags)

    def forward(self, x):
        feat = self.backbone(x)
        return {
            "dynasty": self.dynasty_head(feat),
            "object": self.object_head(feat),
            "tags": self.tags_head(feat)
        }

model = MultiTaskCNN(
    len(dynasty_encoder.classes_),
    len(object_encoder.classes_),
    len(tag_encoder.classes_)
).to(DEVICE)

# =====================
# LOSS & OPTIMIZER
# =====================
loss_dynasty = nn.CrossEntropyLoss()
loss_object  = nn.CrossEntropyLoss()
loss_tags    = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# =====================
# TRAINING LOOP
# =====================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        images = batch["image"].to(DEVICE)
        d = batch["dynasty"].to(DEVICE)
        o = batch["object"].to(DEVICE)
        t = batch["tags"].to(DEVICE)

        out = model(images)

        loss = (
            loss_dynasty(out["dynasty"], d) +
            loss_object(out["object"], o) +
            loss_tags(out["tags"], t)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

# =====================
# SAVE MODEL
# =====================
torch.save({
    "model": model.state_dict(),
    "dynasty_encoder": dynasty_encoder,
    "object_encoder": object_encoder,
    "tag_encoder": tag_encoder
}, "multitask_porcelain_grouped.pth")

print("Training complete ✅")
