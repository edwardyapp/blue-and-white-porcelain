import re
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Regex for parsing
# =========================
epoch_pattern = re.compile(
    r"Epoch\s+(\d+):\s+"
    r"train_loss=([\d.]+),\s+"
    r"train_acc=([\d.]+),\s+"
    r"val_loss=([\d.]+),\s+"
    r"val_acc=([\d.]+)"
)

# =========================
# Loop through experiments
# =========================
experiment_dirs = sorted(glob.glob("experiments/resnet18*"))

for exp_dir in experiment_dirs:

    log_dir = os.path.join(exp_dir, "logs")
    if not os.path.exists(log_dir):
        continue

    print(f"\nProcessing {exp_dir}")

    records = []

    log_files = sorted(glob.glob(os.path.join(log_dir, "results_seed*.txt")))

    if len(log_files) == 0:
        print("⚠️ No log files found, skipping.")
        continue

    for path in log_files:
        seed_match = re.search(r"seed(\d+)\.txt$", path)
        if seed_match is None:
            continue

        seed = int(seed_match.group(1))

        with open(path, "r") as f:
            for line in f:
                m = epoch_pattern.search(line)
                if m:
                    epoch, train_loss, train_acc, val_loss, val_acc = m.groups()

                    records.append({
                        "seed": seed,
                        "epoch": int(epoch),
                        "train_loss": float(train_loss),
                        "train_acc": float(train_acc),
                        "val_loss": float(val_loss),
                        "val_acc": float(val_acc),
                    })

    if len(records) == 0:
        print("⚠️ No epochs parsed, skipping.")
        continue

    df = pd.DataFrame(records)

    # =========================
    # Align epochs across seeds
    # =========================
    max_epoch_per_seed = df.groupby("seed")["epoch"].max()
    min_common_epoch = max_epoch_per_seed.min()

    df = df[df["epoch"] <= min_common_epoch]

    # =========================
    # Aggregate mean ± std
    # =========================
    summary = (
        df.groupby("epoch")
          .agg(
              train_loss_mean=("train_loss", "mean"),
              train_loss_std=("train_loss", "std"),
              val_loss_mean=("val_loss", "mean"),
              val_loss_std=("val_loss", "std"),
              train_acc_mean=("train_acc", "mean"),
              train_acc_std=("train_acc", "std"),
              val_acc_mean=("val_acc", "mean"),
              val_acc_std=("val_acc", "std"),
          )
          .reset_index()
    )

    # Save CSV inside experiment folder
    csv_path = os.path.join(exp_dir, "learning_curves_summary.csv")
    summary.to_csv(csv_path, index=False)

    # =========================
    # Plot
    # =========================
    plt.figure(figsize=(8, 9))

    # ---- Loss ----
    plt.subplot(2, 1, 1)

    plt.plot(summary["epoch"], summary["train_loss_mean"], label="Train loss")
    plt.fill_between(
        summary["epoch"],
        summary["train_loss_mean"] - summary["train_loss_std"],
        summary["train_loss_mean"] + summary["train_loss_std"],
        alpha=0.25
    )

    plt.plot(summary["epoch"], summary["val_loss_mean"], label="Validation loss")
    plt.fill_between(
        summary["epoch"],
        summary["val_loss_mean"] - summary["val_loss_std"],
        summary["val_loss_mean"] + summary["val_loss_std"],
        alpha=0.25
    )

    plt.ylabel("Loss")
    plt.ylim(bottom=0)
    plt.title(os.path.basename(exp_dir))
    plt.legend()
    plt.grid(True)

    # ---- Accuracy ----
    plt.subplot(2, 1, 2)

    plt.plot(summary["epoch"], summary["train_acc_mean"], label="Train accuracy")
    plt.fill_between(
        summary["epoch"],
        summary["train_acc_mean"] - summary["train_acc_std"],
        summary["train_acc_mean"] + summary["train_acc_std"],
        alpha=0.25
    )

    plt.plot(summary["epoch"], summary["val_acc_mean"], label="Validation accuracy")
    plt.fill_between(
        summary["epoch"],
        summary["val_acc_mean"] - summary["val_acc_std"],
        summary["val_acc_mean"] + summary["val_acc_std"],
        alpha=0.25
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # =========================
    # Save PDF
    # =========================
    pdf_path = os.path.join(exp_dir, "learning_curves.pdf")
    plt.savefig(pdf_path, format="pdf")
    plt.close()

    print(f"✅ Saved: {pdf_path}")

print("\n🎉 All experiments processed.")