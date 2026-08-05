# Plot a row-normalised confusion matrix
# Colouring represents fractions within each true class (row sums to 1)

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

counts = cm.values.astype(float)
labels_x = cm.columns
labels_y = cm.index

# -----------------------------
# Row normalisation
# -----------------------------
row_sums = counts.sum(axis=1, keepdims=True)

values_norm = np.divide(
    counts,
    row_sums,
    out=np.zeros_like(counts, dtype=float),
    where=row_sums != 0
)

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(9.5, 9))

im = ax.imshow(
    values_norm,
    cmap="Blues",
    vmin=0.0,
    vmax=1.0
)

ax.set_xticks(np.arange(len(labels_x)))
ax.set_yticks(np.arange(len(labels_y)))

ax.set_xticklabels(labels_x, rotation=45, ha="right")
ax.set_yticklabels(labels_y)

ax.set_xlabel("Predicted label", fontsize=12)
ax.set_ylabel("True label", fontsize=12)

# -----------------------------
# Cell annotations
# -----------------------------
threshold = 0.5

for i in range(values_norm.shape[0]):
    for j in range(values_norm.shape[1]):

        text_colour = (
            "white"
            if values_norm[i, j] > threshold
            else "black"
        )

        ax.text(
            j,
            i,
            f"{int(counts[i, j])}\n({values_norm[i, j]:.2f})",
            ha="center",
            va="center",
            fontsize=8,
            color=text_colour
        )

# -----------------------------
# Colour bar
# -----------------------------
cbar = fig.colorbar(
    im,
    ax=ax,
    fraction=0.046,
    pad=0.04
)

cbar.set_label("Row-normalised fraction")

ax.set_aspect("equal")
plt.tight_layout()

# -----------------------------
# Save
# -----------------------------
plt.savefig(
    OUTPUT_DIR / "confusion_matrix_L5_row_normalised.svg",
    format="svg",
    bbox_inches="tight"
)

plt.savefig(
    OUTPUT_DIR / "confusion_matrix_L5_row_normalised.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()