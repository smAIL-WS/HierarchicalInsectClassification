# Created 18.12.2025
"""
Counts-based histograms for image size metrics (width, height, area, aspect ratio),
comparing years 2021 vs 2023, aggregated across all samples.

- Width/Height: integer bins (exact per-pixel counts up to observed max)
- Area: histogram in log10-space (counts); x-axis ticks show raw pixel² (compact)
- Aspect Ratio: counts with adaptive bins (Freedman–Diaconis)
- Legends (inside, top-right, two rows = ncol=1): include Mode, Min, Max per year
- Extra headroom ONLY on the Area subplot to ensure legend doesn't overlap bars
- Prints min/max for each metric (overall and per year)
- Saves figure at dpi=300

Requirements:
- duckdb
- pandas
- numpy
- seaborn
- matplotlib
"""

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches


# -----------------------------
# Configuration
# -----------------------------
DB_PATH = '/Projects/FAIR_Device_data/Zaki/preprocess/insect_images_final.duckdb'
TABLE = 'cropped_images_optim'
# SAVE_PATH = '/Projects/FAIR_Device_data/Zaki/preprocess/plots/image_size_distr_all_class.png'

OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

FIGSIZE = (9.5, 8)
COLOR_2021 = '#0072B2'
COLOR_2023 = '#D55E00'
ALPHA_FILL = 0.40

# Font sizes (increased by +2 points compared to previous script)
TITLE_SIZE = 12      # subplot titles
AX_LABEL_SIZE = 12   # axes labels
LEGEND_SIZE = 9     # legend text

AREA_HEADROOM_Y = 0.28  # reserve top headroom inside the area subplot only


# -----------------------------
# Helpers
# -----------------------------
def format_compact(n: float) -> str:
    """Format large numbers in compact notation (e.g., 2.3k, 1.2M)."""
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}k"
    else:
        return f"{n:.0f}"

def fd_bin_edges(data: np.ndarray, min_bins: int = 50, max_bins: int = 400):
    """
    Compute bin edges using the Freedman–Diaconis rule.
    Returns a numpy array of edges. Falls back to min_bins if IQR=0 or too few points.
    """
    x = np.asarray(data)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.linspace(0, 1, 11)

    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        bins = min_bins
    else:
        bin_width = 2 * iqr / np.cbrt(x.size)
        data_range = x.max() - x.min()
        bins = int(np.ceil(data_range / bin_width)) if bin_width > 0 else min_bins
        bins = max(min_bins, min(max_bins, bins))
    bins = max(2, bins)
    return np.linspace(x.min(), x.max(), bins + 1)

def mode_bin_center(values: np.ndarray, edges: np.ndarray):
    """
    Return (center, count) of the mode bin given values and precomputed bin edges.
    """
    x = values[np.isfinite(values)]
    counts, e = np.histogram(x, bins=edges)
    if counts.size == 0:
        return None, 0
    idx = int(np.argmax(counts))
    center = (e[idx] + e[idx + 1]) / 2.0
    return center, int(counts[idx])

def legend_label_for_metric(year_label: str, mode_value_str: str, min_str: str, max_str: str):
    """
    Build a legend label string like:
    '2021 — Mode: 512 px | Min: 12 px | Max: 349 px'
    """
    #return f"{year_label} — Mode: {mode_value_str} | Min: {min_str} | Max: {max_str}"
    return (
        f"{year_label}\n"
        f"Mode: {mode_value_str}\n"
        f"Min: {min_str}\n"
        f"Max: {max_str}"
    )

def add_manual_legend_inside(ax, labels_colors, *, title='Year', ncol=1, fontsize=LEGEND_SIZE):
    """
    Add a manual legend inside the axes (top-right), with ncol columns.
    Using ncol=1 gives two rows (one entry per row) for two years.
    """
    if not labels_colors:
        if hasattr(ax, 'legend_') and ax.legend_:
            ax.legend_.remove()
        return

    handles = [mpatches.Patch(color=color, label=label) for label, color in labels_colors]
    leg = ax.legend(
        handles=handles,
        title=title,
        loc='upper right',  # inside the axes
        ncol=ncol,          # ncol=1 -> rows
        frameon=True,
        fontsize=fontsize,
        labelspacing=1.0,  # vertical space between entries
        handlelength=1.5  # width of color patch

    )
    leg.get_title().set_fontsize(fontsize)


# -----------------------------
# Load and prepare data
# -----------------------------
con = duckdb.connect(database=DB_PATH, read_only=True)
df = con.execute(f"SELECT year, width, height FROM {TABLE};").fetchdf()
con.close()

# Clean: drop invalid/missing
df = df.dropna(subset=['year', 'width', 'height'])
df = df[(df['width'] > 0) & (df['height'] > 0)]

# Only years 2021 and 2023
df = df[df['year'].isin([2021, 2023])]

# Derived metrics
df['area'] = df['width'] * df['height']
df['aspect_ratio'] = df['width'] / df['height']
df['log_area'] = np.log10(df['area'].astype(float))

# Split by year
df_2021 = df[df['year'] == 2021].copy()
df_2023 = df[df['year'] == 2023].copy()

# Arrays
w_2021, w_2023 = df_2021['width'].to_numpy(), df_2023['width'].to_numpy()
h_2021, h_2023 = df_2021['height'].to_numpy(), df_2023['height'].to_numpy()
a_2021, a_2023 = df_2021['area'].to_numpy(), df_2023['area'].to_numpy()
ar_2021, ar_2023 = df_2021['aspect_ratio'].to_numpy(), df_2023['aspect_ratio'].to_numpy()
log_a_2021, log_a_2023 = df_2021['log_area'].to_numpy(), df_2023['log_area'].to_numpy()

# -----------------------------
# Print min/max (to terminal)
# -----------------------------
print("\n=== Min/Max Summary ===")
# Width
w_all = df['width'].to_numpy()
print(f"Width (px): overall min={np.min(w_all):.0f}, max={np.max(w_all):.0f}")
if w_2021.size: print(f"  2021: min={np.min(w_2021):.0f}, max={np.max(w_2021):.0f}")
if w_2023.size: print(f"  2023: min={np.min(w_2023):.0f}, max={np.max(w_2023):.0f}")
# Height
h_all = df['height'].to_numpy()
print(f"Height (px): overall min={np.min(h_all):.0f}, max={np.max(h_all):.0f}")
if h_2021.size: print(f"  2021: min={np.min(h_2021):.0f}, max={np.max(h_2021):.0f}")
if h_2023.size: print(f"  2023: min={np.min(h_2023):.0f}, max={np.max(h_2023):.0f}")
# Area (compact)
a_all = df['area'].to_numpy()
print(f"Area (px²): overall min={format_compact(np.min(a_all))}, max={format_compact(np.max(a_all))}")
if a_2021.size: print(f"  2021: min={format_compact(np.min(a_2021))}, max={format_compact(np.max(a_2021))}")
if a_2023.size: print(f"  2023: min={format_compact(np.min(a_2023))}, max={format_compact(np.max(a_2023))}")
# Aspect Ratio
ar_all = df['aspect_ratio'].to_numpy()
print(f"Aspect Ratio (w/h): overall min={np.min(ar_all):.3f}, max={np.max(ar_all):.3f}")
if ar_2021.size: print(f"  2021: min={np.min(ar_2021):.3f}, max={np.max(ar_2021):.3f}")
if ar_2023.size: print(f"  2023: min={np.min(ar_2023):.3f}, max={np.max(ar_2023):.3f}")

# -----------------------------
# Plot
# -----------------------------
sns.set_theme(style='whitegrid')
fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)

# Common histplot kwargs (counts, overlay, step + fill)
hist_kwargs = dict(
    stat='count',
    multiple='layer',
    element='step',
    fill=True,
    alpha=ALPHA_FILL,
)

# -----------------------------
# 1) Width (integer bins 0..max observed)
# -----------------------------
ax = axes[0, 0]
w_min, w_max = int(np.floor(w_all.min())), int(np.ceil(w_all.max()))  # based on your data: 12..611
width_bin_edges = np.arange(0, w_max + 1)

# Plot per year
if w_2021.size:
    sns.histplot(df_2021, x='width', bins=width_bin_edges, color=COLOR_2021, ax=ax, **hist_kwargs)
if w_2023.size:
    sns.histplot(df_2023, x='width', bins=width_bin_edges, color=COLOR_2023, ax=ax, **hist_kwargs)

# Mode (mode bin center) per year
w_center_2021, _ = mode_bin_center(w_2021, width_bin_edges)
w_center_2023, _ = mode_bin_center(w_2023, width_bin_edges)

# Legend labels per year (Mode, Min, Max) in two rows (ncol=1)
legend_items = []
if w_2021.size:
    label_2021 = legend_label_for_metric(
        "2021",
        f"{int(round(w_center_2021))} px" if w_center_2021 is not None else "n/a",
        f"{int(np.min(w_2021))} px",
        f"{int(np.max(w_2021))} px",
    )
    legend_items.append((label_2021, COLOR_2021))
if w_2023.size:
    label_2023 = legend_label_for_metric(
        "2023",
        f"{int(round(w_center_2023))} px" if w_center_2023 is not None else "n/a",
        f"{int(np.min(w_2023))} px",
        f"{int(np.max(w_2023))} px",
    )
    legend_items.append((label_2023, COLOR_2023))

add_manual_legend_inside(ax, legend_items, ncol=1)

ax.set_title('Width', fontsize=TITLE_SIZE)
ax.set_xlabel('px', fontsize=AX_LABEL_SIZE)
ax.set_ylabel('Images', fontsize=AX_LABEL_SIZE)
ax.tick_params(axis='both', labelsize=AX_LABEL_SIZE - 2)

# -----------------------------
# 2) Height (integer bins 0..max observed)
# -----------------------------
ax = axes[0, 1]
h_min, h_max = int(np.floor(h_all.min())), int(np.ceil(h_all.max()))  # based on your data: 13..600
height_bin_edges = np.arange(0, h_max + 1)

# Plot per year
if h_2021.size:
    sns.histplot(df_2021, x='height', bins=height_bin_edges, color=COLOR_2021, ax=ax, **hist_kwargs)
if h_2023.size:
    sns.histplot(df_2023, x='height', bins=height_bin_edges, color=COLOR_2023, ax=ax, **hist_kwargs)

# Mode per year
h_center_2021, _ = mode_bin_center(h_2021, height_bin_edges)
h_center_2023, _ = mode_bin_center(h_2023, height_bin_edges)

# Legend labels
legend_items = []
if h_2021.size:
    label_2021 = legend_label_for_metric(
        "2021",
        f"{int(round(h_center_2021))} px" if h_center_2021 is not None else "n/a",
        f"{int(np.min(h_2021))} px",
        f"{int(np.max(h_2021))} px",
    )
    legend_items.append((label_2021, COLOR_2021))
if h_2023.size:
    label_2023 = legend_label_for_metric(
        "2023",
        f"{int(round(h_center_2023))} px" if h_center_2023 is not None else "n/a",
        f"{int(np.min(h_2023))} px",
        f"{int(np.max(h_2023))} px",
    )
    legend_items.append((label_2023, COLOR_2023))

add_manual_legend_inside(ax, legend_items, ncol=1)

ax.set_title('Height', fontsize=TITLE_SIZE)
ax.set_xlabel('px', fontsize=AX_LABEL_SIZE)
ax.set_ylabel('Images', fontsize=AX_LABEL_SIZE)
ax.tick_params(axis='both', labelsize=AX_LABEL_SIZE - 2)

# -----------------------------
# 3) Area (log10 histogram; ticks show raw px² compact)
# -----------------------------
ax = axes[1, 0]
log_area_all = df['log_area'].to_numpy()
log_area_bin_edges = fd_bin_edges(log_area_all, min_bins=80, max_bins=400)

# Plot per year in log-space
if log_a_2021.size:
    sns.histplot(df_2021, x='log_area', bins=log_area_bin_edges, color=COLOR_2021, ax=ax, **hist_kwargs)
if log_a_2023.size:
    sns.histplot(df_2023, x='log_area', bins=log_area_bin_edges, color=COLOR_2023, ax=ax, **hist_kwargs)

# Reserve EXTRA headroom ONLY for area subplot so legend doesn't overlap bars
ax.margins(y=AREA_HEADROOM_Y)  # try 0.28; adjust to 0.25–0.35 if needed

# Mode bin centers in log-space; convert to raw px² for legend
la_center_2021, _ = mode_bin_center(log_a_2021, log_area_bin_edges)
la_center_2023, _ = mode_bin_center(log_a_2023, log_area_bin_edges)
mode_area_2021 = format_compact(10 ** la_center_2021) + " px²" if la_center_2021 is not None else "n/a"
mode_area_2023 = format_compact(10 ** la_center_2023) + " px²" if la_center_2023 is not None else "n/a"

# Legend labels
legend_items = []
if a_2021.size:
    label_2021 = legend_label_for_metric(
        "2021",
        mode_area_2021,
        f"{format_compact(np.min(a_2021))} px²",
        f"{format_compact(np.max(a_2021))} px²",
    )
    legend_items.append((label_2021, COLOR_2021))
if a_2023.size:
    label_2023 = legend_label_for_metric(
        "2023",
        mode_area_2023,
        f"{format_compact(np.min(a_2023))} px²",
        f"{format_compact(np.max(a_2023))} px²",
    )
    legend_items.append((label_2023, COLOR_2023))

add_manual_legend_inside(ax, legend_items, ncol=1)

ax.set_title('Area', fontsize=TITLE_SIZE)
ax.set_xlabel('px²', fontsize=AX_LABEL_SIZE)
ax.set_ylabel('Images', fontsize=AX_LABEL_SIZE)
ax.tick_params(axis='both', labelsize=AX_LABEL_SIZE - 2)

# Format x-ticks: raw area values while axis is log10(area)
def log_to_raw_formatter(x, pos):
    return format_compact(10 ** x)

if log_area_all.size:
    lo, hi = np.floor(log_area_all.min()), np.ceil(log_area_all.max())
    decade_ticks = np.arange(lo, hi + 1)
    ax.set_xticks(decade_ticks)
ax.xaxis.set_major_formatter(FuncFormatter(log_to_raw_formatter))

# -----------------------------
# 4) Aspect Ratio (counts, FD bins)
# -----------------------------
ax = axes[1, 1]
ar_all = df['aspect_ratio'].to_numpy()
ar_bin_edges = fd_bin_edges(ar_all, min_bins=100, max_bins=400)

# Plot per year
if ar_2021.size:
    sns.histplot(df_2021, x='aspect_ratio', bins=ar_bin_edges, color=COLOR_2021, ax=ax, **hist_kwargs)
if ar_2023.size:
    sns.histplot(df_2023, x='aspect_ratio', bins=ar_bin_edges, color=COLOR_2023, ax=ax, **hist_kwargs)

# Mode per year
ar_center_2021, _ = mode_bin_center(ar_2021, ar_bin_edges)
ar_center_2023, _ = mode_bin_center(ar_2023, ar_bin_edges)


# Legend labels
legend_items = []
if ar_2021.size:
    label_2021 = legend_label_for_metric(
        "2021",
        f"{ar_center_2021:.2f}" if ar_center_2021 is not None else "n/a",
        f"{np.min(ar_2021):.3f}",
        f"{np.max(ar_2021):.3f}",
    )
    legend_items.append((label_2021, COLOR_2021))

if ar_2023.size:
    label_2023 = legend_label_for_metric(
        "2023",
        f"{ar_center_2023:.2f}" if ar_center_2023 is not None else "n/a",
        f"{np.min(ar_2023):.3f}",
        f"{np.max(ar_2023):.3f}",
    )
    legend_items.append((label_2023, COLOR_2023))

add_manual_legend_inside(ax, legend_items, ncol=1)

ax.set_title('Aspect Ratio', fontsize=TITLE_SIZE)
ax.set_xlabel('width/height', fontsize=AX_LABEL_SIZE)
ax.set_ylabel('Images', fontsize=AX_LABEL_SIZE)
ax.tick_params(axis='both', labelsize=AX_LABEL_SIZE - 2)

# -----------------------------
# Layout, save, show
# -----------------------------
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "image_size_distr_all_class.svg",
    format="svg",
    bbox_inches="tight"
)

plt.savefig(
    OUTPUT_DIR / "image_size_distr_all_class.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


'/Projects/FAIR_Device_data/Zaki/preprocess/plots/image_size_distr_all_class.png'