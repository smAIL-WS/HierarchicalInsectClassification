# Plot accuracy and coverage against probability threshold.
# One line per hierarchy level (L1-L5).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------
# Configuration
# -------------------------------------------------

CSV_FILE = Path(__file__).parent / "preds_at_var_thresholds.csv"

OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

THRESHOLDS = [0.4, 0.6, 0.8, 0.9]

FIGURE_TITLE = None

Y_MIN = 0.75
Y_MAX = 1.01
Y_TICKS = np.arange(0.75, 1.01, 0.05)

TITLE_SIZE = 12
LABEL_SIZE = 12
TICK_SIZE = 11
LEGEND_SIZE = 8
LEGEND_TITLE_SIZE = 10

LINE_WIDTH = 1.0
MARKER_SIZE = 4


# -------------------------------------------------
# Load data
# -------------------------------------------------

df = pd.read_csv(CSV_FILE)

# Ensure levels appear in correct order
df["Level"] = pd.Categorical(
    df["Level"],
    categories=["L1", "L2", "L3", "L4", "L5"],
    ordered=True
)

df = df.sort_values("Level")

# -------------------------------------------------
# Create figure
# -------------------------------------------------

fig, (ax_cov, ax_acc) = plt.subplots(
    1,
    2,
    figsize=(9.5, 4.5)
)

if FIGURE_TITLE:
    fig.suptitle(FIGURE_TITLE)

# Distinct colours for levels
LEVEL_COLOURS = {
    "L1": "#0072B2",  # blue
    "L2": "#009E73",  # green
    "L3": "#E69F00",  # orange
    "L4": "#CC79A7",  # purple
    "L5": "#D55E00",  # vermillion
}

# -------------------------------------------------
# Plot one line per hierarchy level
# -------------------------------------------------

for _, row in df.iterrows():

    level = row["Level"]

    accuracy = [
        row["Accuracy_0.4"],
        row["Accuracy_0.6"],
        row["Accuracy_0.8"],
        row["Accuracy_0.9"]
    ]

    coverage = [
        row["Proportion_P=0.4"],
        row["Proportion_P=0.6"],
        row["Proportion_P=0.8"],
        row["Proportion_P=0.9"]
    ]

    ax_acc.plot(
        THRESHOLDS,
        accuracy,
        marker="o",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        color=LEVEL_COLOURS[level],
        label=level
    )

    ax_cov.plot(
        THRESHOLDS,
        coverage,
        marker="o",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        color=LEVEL_COLOURS[level],
        label=level
    )

# -------------------------------------------------
# Format Accuracy plot
# -------------------------------------------------

# ax_acc.set_title("Accuracy", fontsize=TITLE_SIZE)
ax_acc.set_xlabel("Probability Threshold", fontsize=LABEL_SIZE)
ax_acc.set_ylabel("Accuracy-on-Covered (%)", fontsize=LABEL_SIZE)

ax_acc.set_xticks(THRESHOLDS)
ax_acc.set_ylim(Y_MIN, Y_MAX)
ax_acc.set_yticks(Y_TICKS)

ax_acc.tick_params(axis="both", labelsize=TICK_SIZE)

ax_acc.grid(True, linestyle="--", alpha=0.4)

legend_acc = ax_acc.legend(
    title="Hierarchy\nLevel",
    loc="lower right",
    frameon=True,
    fontsize=LEGEND_SIZE,
    title_fontsize=LEGEND_TITLE_SIZE,
    labelspacing=0.2,
    handlelength=1.4,
    borderpad=0.15
)

legend_acc.get_title().set_ha("center")

frame = legend_acc.get_frame()
frame.set_facecolor("white")
frame.set_edgecolor("0.7")
frame.set_linewidth(0.4)
frame.set_alpha(1.0)

# -------------------------------------------------
# Format Coverage plot
# -------------------------------------------------

# ax_cov.set_title("Coverage", fontsize=TITLE_SIZE)
ax_cov.set_xlabel("Probability Threshold", fontsize=LABEL_SIZE)
ax_cov.set_ylabel("Coverage (%)", fontsize=LABEL_SIZE)

ax_cov.set_xticks(THRESHOLDS)
ax_cov.set_ylim(Y_MIN, Y_MAX)
ax_cov.set_yticks(Y_TICKS)

ax_cov.tick_params(axis="both", labelsize=TICK_SIZE)

ax_cov.grid(True, linestyle="--", alpha=0.4)

legend_cov = ax_cov.legend(
    title="Hierarchy\nLevel",
    loc="lower left",
    frameon=True,
    fontsize=LEGEND_SIZE,
    title_fontsize=LEGEND_TITLE_SIZE,
    labelspacing=0.2,
    handlelength=1.4,
    borderpad=0.15
)

legend_cov.get_title().set_ha("center")

frame = legend_cov.get_frame()
frame.set_facecolor("white")
frame.set_edgecolor("0.7")
frame.set_linewidth(0.4)
frame.set_alpha(1.0)

plt.tight_layout()

# -------------------------------------------------
# Save
# -------------------------------------------------

plt.savefig(
    OUTPUT_DIR / "accuracy_coverage_by_threshold.svg",
    format="svg",
    bbox_inches="tight"
)

plt.savefig(
    OUTPUT_DIR / "accuracy_coverage_by_threshold.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()