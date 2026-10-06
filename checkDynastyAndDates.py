import pandas as pd

CSV_FILE = "filtered_chinese_porcelain.csv"

def dynasty_from_year(year):
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


def infer_dynasty_from_period(period):
    if pd.isna(period):
        return None
    p = period.lower()
    if "qing" in p:
        return "Qing"
    if "ming" in p:
        return "Ming"
    if "yuan" in p:
        return "Yuan"
    if "song" in p:
        return "Song"
    if "tang" in p:
        return "Tang"
    if "sui" in p:
        return "Sui"
    if "han" in p:
        return "Han"
    return None


df = pd.read_csv(CSV_FILE)

# Extract dynasty from Period text
df["dynasty_from_period"] = df["Period"].apply(infer_dynasty_from_period)

# Use begin date if available, else end date
df["year_used"] = df.apply(
    lambda r: r["Object Begin Date"]
    if pd.notna(r["Object Begin Date"])
    else r["Object End Date"],
    axis=1
)

# Infer dynasty from year
df["dynasty_from_year"] = df["year_used"].apply(dynasty_from_year)

# Compare
valid = df[
    df["dynasty_from_period"].notna() &
    df["dynasty_from_year"].notna()
]

matches = valid[valid["dynasty_from_period"] == valid["dynasty_from_year"]]
mismatches = valid[valid["dynasty_from_period"] != valid["dynasty_from_year"]]

print(f"Total comparable rows: {len(valid)}")
print(f"Matches: {len(matches)} ({len(matches)/len(valid):.2%})")
print(f"Mismatches: {len(mismatches)} ({len(mismatches)/len(valid):.2%})")

print("\nSample mismatches:")
print(
    mismatches[
        ["Period", "Object Begin Date", "Object End Date",
         "dynasty_from_period", "dynasty_from_year"]
    ].head(15)
)
