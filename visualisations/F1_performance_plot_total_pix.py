# Plot the F1-score performance as a function of the number of the total number of pixels seen by the model per class.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import linregress
from matplotlib.ticker import LogLocator, LogFormatterMathtext

# -------------------------------------------------
# Configuration
# -------------------------------------------------

CSV_FILE = Path(__file__).parent / "data_graphs.csv"

OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Column selection:
# Use column names (recommended) OR integer indices
LABEL_COLUMN = 1          # labels (strings)
X_COLUMN = 9              # independent variable
Y_COLUMN = 11              # dependent variable
GROUP_COLUMN = 0            # hierarchy level

USE_POINT_LABELS = True  # set True if you want labels drawn

FIGURE_TITLE = None #"F1-score vs. total pixels per class"
X_AXIS_LABEL = "Total pixels per class (log. scale)"
Y_AXIS_LABEL = "Test F1-score"

# -------------------------------------------------
# Grouping configuration (fully optional)
# -------------------------------------------------

# Whether to use group-based colours/markers
USE_GROUP_STYLES = True

# Define styles for each group (exact names must match your CSV values)
# You can freely change colours/markers later.
GROUP_STYLES = {
    "L1": {"color": "darkred",    "marker": "o"},   # circle
    "L2": {"color": "navy",       "marker": "s"},   # square
    "L3": {"color": "forestgreen","marker": "^"},   # triangle-up
    "L4": {"color": "orange",     "marker": "D"},   # diamond
    "L5": {"color": "purple",     "marker": "X"}    # X marker
}

# Fallback style when grouping is off or group missing
DEFAULT_STYLE = {"color": "darkred", "marker": "o"}
POINT_SIZE = 20

LABEL_PLACEMENT = {
# label_name: (horizontal_align, vertical_align, dx, dy)
    "Brachycera": ("right", "top",    -1, -1),
    "Hymenoptera": ("left", "top",    1, -1),
    "Nematocera": ("left", "bottom",    1, 1),
    "Orthoptera": ("right", "bottom",    -1, 0),
    "Asiloidea": ("left", "bottom",    -1, -0),
    "Coccinelloidea": ("right", "bottom",    -1, 1),
    "Lauxanioidea": ("right", "bottom", -1, 1),
    "Heterocera": ("right", "bottom",    -1, 1),
    "Muscoidea": ("right", "bottom", -1, 1),
    "Oestroidea": ("left", "bottom",    1, 1),
    "Syrphidae": ("right", "bottom",    -1, -1),
    "Aculeata": ("left", "bottom",    1, 0),
    "Coleoptera": ("right", "top", -1, -1),
    "Apocrita": ("right", "top", -1, -1),
    "Sepsidae": ("right", "bottom", -1, 1),
    "Curculionoidea": ("left", "top", 1, -1),
    "Dermaptera": ("right", "bottom", -1, 1),
}


# -------------------------------------------------
# Load data
# -------------------------------------------------

df = pd.read_csv(CSV_FILE)

labels = df.iloc[:, LABEL_COLUMN].astype(str)
x = df.iloc[:, X_COLUMN].astype(float).to_numpy()
y = df.iloc[:, Y_COLUMN].astype(float).to_numpy()

# Remove invalid values
mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
x = x[mask]
y = y[mask]
labels = labels[mask]

# -------------------------------------------------
# Linear regression (for R² and fit line)
# -------------------------------------------------

slope, intercept, r_value, p_value, std_err = linregress(np.log10(x), y)
r_squared = r_value ** 2

# Fit line evaluated at sorted x (for clean drawing)
x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
y_fit = slope * np.log10(x_fit) + intercept

# -------------------------------------------------
# Plot
# -------------------------------------------------
# print(df.iloc[:, GROUP_COLUMN].head(20))
fig, ax = plt.subplots(figsize=(9.5, 7))

if FIGURE_TITLE is not None:
    fig.suptitle(FIGURE_TITLE, fontsize=14)


if USE_GROUP_STYLES:
    # Load group column
    groups = df.iloc[:, GROUP_COLUMN].astype(str).to_numpy()
    groups = groups[mask]   # apply same mask as x,y

    # Plot each group separately
    for group_name in np.unique(groups):

        # Determine style for this group
        style = GROUP_STYLES.get(group_name, DEFAULT_STYLE)

        # Extract group-specific points
        gx = x[groups == group_name]
        gy = y[groups == group_name]

        ax.scatter(
            gx,
            gy,
            s=POINT_SIZE,
            color=style["color"],
            marker=style["marker"],
            edgecolors="none",
            label=group_name
        )
else:
    # No grouping: one colour + one marker for all points
    ax.scatter(
        x,
        y,
        s=POINT_SIZE,
        color=DEFAULT_STYLE["color"],
        marker=DEFAULT_STYLE["marker"],
        edgecolors="none"
    )


# Best-fit line
ax.plot(
    x_fit, y_fit,
    color="black",
    linewidth=1.0,
    label=f"Fit: $R^2 = {r_squared:.4f}$"
)

# -------------------------------------------------
# Point labels (manual placement)
# -------------------------------------------------
if USE_POINT_LABELS:
    for xi, yi, lbl in zip(x, y, labels):

        # Default placement: top-right
        ha, va, dx, dy = ("left", "bottom", 1, 1)

        # Override if label is listed
        if lbl in LABEL_PLACEMENT:
            ha, va, dx, dy = LABEL_PLACEMENT[lbl]

        ax.annotate(
            lbl,
            xy=(xi, yi),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=8,
        )

# -------------------------------------------------
# Axes formatting
# -------------------------------------------------

# Log-scaled x-axis with 10^ ticks
ax.set_xscale("log")
ax.xaxis.set_major_locator(LogLocator(base=10))
ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))

# Y-axis: 0 to 1.05, ticks every 0.2 (not including 1.05)
ax.set_ylim(0, 1.05)
ax.set_yticks(np.arange(0.0, 1.01, 0.2))

ax.set_xlabel(X_AXIS_LABEL, fontsize=12)
ax.set_ylabel(Y_AXIS_LABEL, fontsize=12)

# Legend (fit line only)
ax.legend(title="Hierarchy level", frameon=False, title_fontsize=11)

ax.grid(True, which="both", linewidth=0.5, alpha=0.3)
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig(
    OUTPUT_DIR / "F1_performance_total_pix.svg",
    format="svg",
    bbox_inches="tight"
)

plt.savefig(
    OUTPUT_DIR / "F1_performance_total_pix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()