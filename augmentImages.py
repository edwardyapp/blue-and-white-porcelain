import os
import random
from PIL import Image, ImageEnhance
import torchvision.transforms.functional as TF
import numpy as np

# ==========================
# CONFIG
# ==========================
INPUT_ROOT = "images_clean"     # Each object has a subfolder with images
OUTPUT_ROOT = "images_clean_augmented"   # Output root with same object folder structure
AUGMENTATIONS_PER_IMAGE = 20        # How many augmented images per original image
IMAGE_SIZE = (448, 448)             # Resize/crop size

# Random rotation range in degrees
ROTATION_DEGREES = 20

# Brightness & contrast ranges
BRIGHTNESS_RANGE = (0.7, 1.3)
CONTRAST_RANGE = (0.7, 1.3)

# Gaussian noise
NOISE_STD = 0.02

random.seed(42)
np.random.seed(42)

# ==========================
# AUGMENTATION FUNCTIONS
# ==========================
def add_gaussian_noise(image, std=0.02):
    """Add Gaussian noise to a PIL image"""
    arr = np.array(image).astype(np.float32) / 255.0
    noise = np.random.normal(0, std, arr.shape)
    arr += noise
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)

def augment_image(image):
    """Apply random augmentations to a PIL image"""
    # Random horizontal flip
    if random.random() < 0.5:
        image = TF.hflip(image)

    # Random rotation
    angle = random.uniform(-ROTATION_DEGREES, ROTATION_DEGREES)
    image = TF.rotate(image, angle, expand=True)

    # Random brightness
    enhancer = ImageEnhance.Brightness(image)
    factor = random.uniform(*BRIGHTNESS_RANGE)
    image = enhancer.enhance(factor)

    # Random contrast
    enhancer = ImageEnhance.Contrast(image)
    factor = random.uniform(*CONTRAST_RANGE)
    image = enhancer.enhance(factor)

    # Resize/crop to target
    image = image.resize(IMAGE_SIZE)

    # Add Gaussian noise
    image = add_gaussian_noise(image, NOISE_STD)

    return image

# ==========================
# MAIN LOOP
# ==========================
if not os.path.exists(OUTPUT_ROOT):
    os.makedirs(OUTPUT_ROOT)

# Loop over object folders
for obj_folder in os.listdir(INPUT_ROOT):
    obj_path = os.path.join(INPUT_ROOT, obj_folder)
    if not os.path.isdir(obj_path):
        continue

    out_obj_path = os.path.join(OUTPUT_ROOT, obj_folder)
    os.makedirs(out_obj_path, exist_ok=True)

    for img_file in os.listdir(obj_path):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(obj_path, img_file)
        image = Image.open(img_path).convert("RGB")

        # Save original image first
        image.save(os.path.join(out_obj_path, f"orig_{img_file}"))

        # Generate augmented images
        for i in range(AUGMENTATIONS_PER_IMAGE):
            aug_img = augment_image(image)
            aug_filename = f"{os.path.splitext(img_file)[0]}_aug{i+1}.jpg"
            aug_img.save(os.path.join(out_obj_path, aug_filename))

    print(f"✅ Processed object {obj_folder}")

print("\n🎉 Augmentation complete!")
