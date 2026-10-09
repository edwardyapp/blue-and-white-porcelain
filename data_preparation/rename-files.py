import os
import shutil
import csv

SRC_ROOT = "images_clean"                 # images/<object_id>/*.jpg
DST_ROOT = "images_clean_flat"
ALLOWED_IDS_CSV = "blue-object-ids.csv"

# ---- Load allowed object IDs ----
allowed_object_ids = set()

with open(ALLOWED_IDS_CSV, newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        if not row:
            continue
        allowed_object_ids.add(row[0].strip())

print(f"🔎 Loaded {len(allowed_object_ids)} allowed object IDs")

# ---- Prepare destination ----
os.makedirs(DST_ROOT, exist_ok=True)

# ---- Flatten only allowed object IDs ----
for object_id in os.listdir(SRC_ROOT):
    if object_id not in allowed_object_ids:
        continue

    obj_dir = os.path.join(SRC_ROOT, object_id)
    if not os.path.isdir(obj_dir):
        continue

    for fname in os.listdir(obj_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        src_path = os.path.join(obj_dir, fname)
        new_name = f"{object_id}__{fname}"
        dst_path = os.path.join(DST_ROOT, new_name)

        if os.path.exists(dst_path):
            raise RuntimeError(f"File already exists: {dst_path}")

        shutil.copy2(src_path, dst_path)

print("✅ Flattening complete (blue objects only)")
