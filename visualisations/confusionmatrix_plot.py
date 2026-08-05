# Plot a clean confusion matrix, based on the figures for L5 from the test run
# Values for the confusion matrix in CSV format
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Load confusion matrix
# -----------------------------

OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

csv_path = Path(__file__).parent / "confusionmatrixL5.csv"
cm = pd.read_csv(csv_path, index_col=0)

values = cm.values
labels_x = cm.columns
labels_y = cm.index

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(9.5, 9))

im = ax.imshow(
    values,
    cmap="Blues",        # linear scale by default
    vmin=0,
    vmax=values.max()
)

ax.set_xticks(np.arange(len(labels_x)))
ax.set_yticks(np.arange(len(labels_y)))
ax.set_xticklabels(labels_x, rotation=45, ha="right")
ax.set_yticklabels(labels_y)

ax.set_xlabel("Predicted label", fontsize=12)
ax.set_ylabel("True label", fontsize=12)
# ax.set_title("Confusion Matrix (Counts) - Hierarchy Level 5")

# Annotate raw counts
for i in range(values.shape[0]):
    for j in range(values.shape[1]):
        ax.text(
            j, i, values[i, j],
            ha="center", va="center",
            fontsize=9,
            color="black"
        )

# Colorbar (log scale, counts)
# cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
# cbar.set_label("Image count (log scale)")

ax.set_aspect("equal")
plt.tight_layout()

# -----------------------------
# Save
# -----------------------------
plt.savefig(
    OUTPUT_DIR / "confusion_matrix_L5.svg",
    format="svg",
    bbox_inches="tight"
)

plt.savefig(
    OUTPUT_DIR / "confusion_matrix_L5.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()