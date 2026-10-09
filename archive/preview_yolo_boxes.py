import os
import cv2
import random

BASE_DIR = "数据集"
IMG_DIR = os.path.join(BASE_DIR, "train")
LBL_DIR = os.path.join(BASE_DIR, "train label")

N_SAMPLES = 100  # number of images to preview

img_files = [f for f in os.listdir(IMG_DIR) if f.endswith(".png")]
samples = random.sample(img_files, min(N_SAMPLES, len(img_files)))

for img_name in samples:
    img_path = os.path.join(IMG_DIR, img_name)
    lbl_path = os.path.join(LBL_DIR, img_name.replace(".png", ".txt"))

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    if os.path.exists(lbl_path):
        with open(lbl_path, "r") as f:
            for line in f:
                cls, xc, yc, bw, bh = map(float, line.split())

                # YOLO (normalized) → pixel coordinates
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"{int(cls)}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    cv2.imshow("YOLO Preview", img)
    key = cv2.waitKey(0)

    if key == 27:  # ESC
        break

cv2.destroyAllWindows()