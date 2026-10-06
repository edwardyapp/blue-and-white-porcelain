import os
import shutil
from collections import defaultdict

SRC_ROOT = "images_flat"     # flat folder
DST_ROOT = "images_lessStrictFilter"    # reconstructed folders

os.makedirs(DST_ROOT, exist_ok=True)

# ---- Group files by object_id ----
objects = defaultdict(list)

for fname in os.listdir(SRC_ROOT):
    if "__" not in fname:
        continue
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    object_id, original_name = fname.split("__", 1)
    objects[object_id].append((fname, original_name))

# ---- Rebuild folders ----
for object_id, files in objects.items():
    obj_dir = os.path.join(DST_ROOT, object_id)
    os.makedirs(obj_dir, exist_ok=True)

    for flat_name, original_name in files:
        src = os.path.join(SRC_ROOT, flat_name)
        dst = os.path.join(obj_dir, original_name)
        shutil.move(src, dst)

# ---- Delete objects without primary.jpg ----
deleted = 0

for object_id in os.listdir(DST_ROOT):
    obj_dir = os.path.join(DST_ROOT, object_id)
    if not os.path.isdir(obj_dir):
        continue

    has_primary = any(
        f.lower() == "primary.jpg"
        for f in os.listdir(obj_dir)
    )

    if not has_primary:
        shutil.rmtree(obj_dir)
        deleted += 1

print(f"🧹 Cleanup complete — deleted {deleted} objects without primary.jpg")
