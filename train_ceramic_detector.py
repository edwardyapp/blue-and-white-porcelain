import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# =========================
# CONFIG
# =========================
IMAGE_ROOT = "images"
BOXES_FILE = "pseudo_boxes.csv"  # image,x1,y1,x2,y2
NUM_CLASSES = 2  # background + ceramic
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# DATASET
# =========================
class CeramicDataset(Dataset):
    def __init__(self, csv_file):
        import pandas as pd
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.image).convert("RGB")

        box = torch.tensor([[row.x1, row.y1, row.x2, row.y2]], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)

        target = {
            "boxes": box,
            "labels": labels
        }

        img = F.to_tensor(img)
        return img, target

# =========================
# MODEL
# =========================
model = fasterrcnn_resnet50_fpn(pretrained=True)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features, NUM_CLASSES
)

model.to(DEVICE)

# =========================
# TRAINING
# =========================
dataset = CeramicDataset(BOXES_FILE)
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(100):
    model.train()
    total_loss = 0

    for imgs, targets in loader:
        imgs = [img.to(DEVICE) for img in imgs]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}: loss={total_loss:.3f}")

torch.save(model.state_dict(), "ceramic_detector.pt")
