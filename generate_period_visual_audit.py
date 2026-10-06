import os
import pandas as pd
from collections import defaultdict

# =========================
# CONFIG
# =========================
CSV_FILE = "filtered_chinese_porcelain.csv"
IMAGE_ROOT = "images_clean"
OUTPUT_HTML = "period_visual_audit.html"

PRIMARY_IMAGE_NAME = "primary.jpg"
CLASSIFY_BY_PERIOD = False

# -------------------------
# Period inference (reuse yours)
# -------------------------
MING_PERIODS = ["Tianqi", "Chenghua", "Chongzhen", "Jiajing", "Longqing", "Wanli", "Xuande", "Zhengde"]
QING_PERIODS = ["Yongzheng", "Kangxi", "Daoguang", "Guangxu", "Jiaqing", "Qianlong", "Shunzhi"]

def infer_period(row):
    period_text = row.get("Period")

    if pd.notna(period_text):
        text = str(period_text).lower()
        for p in MING_PERIODS + QING_PERIODS:
            if p.lower() in text:
                return p

    begin = row.get("Object Begin Date")
    end = row.get("Object End Date")

    year = begin if pd.notna(begin) else end
    if pd.isna(year):
        return None

    year = int(year)

    MING_PERIOD_YEARS = [
        ("Xuande", 1426, 1435),
        ("Chenghua", 1465, 1487),
        ("Jiajing", 1522, 1566),
        ("Wanli", 1573, 1620),
        ("Tianqi", 1621, 1627),
        ("Chongzhen", 1628, 1644),
    ]

    QING_PERIOD_YEARS = [
        ("Shunzhi", 1644, 1661),
        ("Kangxi", 1662, 1722),
        ("Yongzheng", 1723, 1735),
        ("Qianlong", 1736, 1795),
        ("Jiaqing", 1796, 1820),
        ("Daoguang", 1821, 1850),
        ("Guangxu", 1875, 1908),
    ]

    for name, start, end_ in MING_PERIOD_YEARS + QING_PERIOD_YEARS:
        if start <= year <= end_:
            return name

    return None

# =========================
# MAIN
# =========================
df = pd.read_csv(CSV_FILE)

period_to_objects = defaultdict(list)

for _, row in df.iterrows():
    object_id = str(row.get("Object ID", "")).strip()
    if not object_id:
        continue

    period = infer_period(row)
    if period is None:
        continue

    obj_folder = os.path.join(IMAGE_ROOT, object_id)
    if not os.path.isdir(obj_folder):
        continue

    primary_img = os.path.join(obj_folder, PRIMARY_IMAGE_NAME)
    if not os.path.exists(primary_img):
        continue

    period_to_objects[period].append((object_id, primary_img))

# =========================
# WRITE HTML
# =========================
with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write("""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Period Visual Audit</title>
<style>
body { font-family: Arial, sans-serif; }
h2 { margin-top: 40px; }
.grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 20px;
}
.card {
    border: 1px solid #ccc;
    padding: 10px;
    text-align: center;
}
.card img {
    max-width: 100%;
    height: auto;
}
.object-id {
    font-weight: bold;
    margin-top: 8px;
}
</style>
</head>
<body>

<h1>Jingdezhen Blue-and-White Porcelain – Period Visual Audit</h1>
""")

    for period in sorted(period_to_objects.keys()):
        f.write(f"<h2>{period} Period</h2>\n")
        f.write('<div class="grid">\n')

        for object_id, img_path in period_to_objects[period]:
            rel_path = os.path.relpath(img_path, os.path.dirname(OUTPUT_HTML))
            f.write(f"""
<div class="card">
    <img src="{rel_path}">
    <div class="object-id">Object ID: {object_id}</div>
</div>
""")

        f.write("</div>\n")

    f.write("</body></html>")

print(f"✅ Period visual audit saved to: {OUTPUT_HTML}")
