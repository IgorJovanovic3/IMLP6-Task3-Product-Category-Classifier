"""Recreate the README chart from the supplied data without retraining."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main():
    # Keep the data-selection steps aligned with train_model.py.
    data = pd.read_csv(ROOT / "products.csv")
    data.columns = data.columns.str.strip()
    data = data.rename(
        columns={"Product Title": "product_title", "Category Label": "category"}
    )
    data = data.dropna(subset=["product_title", "category"]).copy()
    data["product_title"] = data["product_title"].astype(str).str.strip()
    data["category"] = data["category"].astype(str).str.strip().replace(
        {"fridge": "Fridges", "CPU": "CPUs", "Mobile Phone": "Mobile Phones"}
    )
    data = data[data["product_title"] != ""]
    category_counts = data.groupby("product_title")["category"].nunique()
    conflicts = category_counts[category_counts > 1].index
    data = data[~data["product_title"].isin(conflicts)]
    data = data.drop_duplicates(subset=["product_title"], keep="first")
    counts = data["category"].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6.2))
    fig.subplots_adjust(left=0.21, right=0.95, top=0.81, bottom=0.16)
    fig.text(0.06, 0.94, "Product categories", fontsize=22, weight="bold", color="#142b3d")
    fig.text(
        0.06, 0.885,
        f"{len(data):,} unique cleaned titles across {len(counts)} categories",
        fontsize=12, color="#526273",
    )
    bars = ax.barh(counts.index, counts.values, color="#247a9a", height=0.64)
    for bar, count in zip(bars, counts.values):
        ax.text(count + 65, bar.get_y() + bar.get_height() / 2,
                f"{count:,}", va="center", fontsize=10, color="#142b3d")
    ax.set_xlim(0, max(counts) * 1.15)
    ax.set_xlabel("Unique product titles", labelpad=10, color="#526273")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e7edf1")
    ax.tick_params(axis="both", length=0, labelcolor="#344659")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(0.06, 0.04, "Source: products.csv · Cleaning follows train_model.py",
             fontsize=9, color="#647486")
    output = ROOT / "docs" / "category_distribution.png"
    fig.savefig(output, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {output.name}: {len(data):,} rows, {len(counts)} categories")


if __name__ == "__main__":
    main()
