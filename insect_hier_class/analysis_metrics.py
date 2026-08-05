from pathlib import Path
from datetime import datetime
from collections import defaultdict
import argparse
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

import config
from insect_dataset_loader import InsectDataset
from transforms_utils import build_test_transform
from hierarchical_target_generation_utils import get_5_level_targets
from hierarchical_classification_metrics import (
    load_level_name_maps,
    hierarchical_predict_map_truncate,
)
from tree_loss import TreeLoss


# ============================================================
# ================= USER CONFIGURATION =======================
# ============================================================

CONF_THRESHOLD = 0.6
BATCH_SIZE = 1024
DEVICE_ID = "0"

SCRIPT_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = SCRIPT_DIR / "models_Insect" / "model_Insect_100_96_bz1024_resnet18_OneCycle_2026-01-19_01-09.pth"


# ============================================================
# ================= HELPERS ==================================
# ============================================================

LEVELS = ["pf4", "pf3", "pf2", "pf1", "leaf"]
LEVEL_DISPLAY = {
    "pf4": "PF4",
    "pf3": "PF3",
    "pf2": "PF2",
    "pf1": "PF1",
    "leaf": "Leaf",
}

# =================================================================
# Paper display names (conversion of common to latin names for publication)
# =================================================================

PAPER_LABEL_MAP = {
    "spider": "Arachnida",
    "leaf_beetle": "Chrysomeloidea",
    "ladybug": "Coccinelloidea",
    "weevil": "Curculionoidea",
    "ant_beetle": "Anthicidae",
    "earwig": "Dermaptera",
    "grasshopper": "Orthoptera",
    "scorpion_fly": "Mecoptera",
    "robber_fly": "Asiloidea",
    "dance_fly": "Empidoidea",
    "fruit_fly": "Ephydroidea",
    "lauxaniid_fly": "Lauxanioidea",
    "muscoid_fly": "Muscoidea",
    "flesh_fly": "Oestroidea",
    "scavenger_fly": "Sepsidae",
    "hoverfly": "Syrphidae",
    "true_fruit_fly": "Tephritidae",
    "fever_fly": "Bibionidae",
    "parasitoid_wasp": "Parasitica",
    "saw_fly": "Symphyta",
    "ant": "Formicidae",
    "wasp": "Vespidae",
    "moth": "Heterocera",
}

def fmt_thr(threshold: float) -> str:
    return str(threshold).replace(".", "p")


def timestamp_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def level_idx(level: str) -> int:
    return LEVELS.index(level)


def level_depth_relation(pred_level_idx: int, true_level_idx: int) -> str:
    if pred_level_idx < true_level_idx:
        return "shorter_than_gt"
    if pred_level_idx == true_level_idx:
        return "same_depth_as_gt"
    return "longer_than_gt"


def build_label_lookup(level_name_maps: dict) -> dict[str, dict[int, str]]:
    mapping = {
        "pf4": "parent_folder_4",
        "pf3": "parent_folder_3",
        "pf2": "parent_folder_2",
        "pf1": "parent_folder_1",
        "leaf": "classification",
    }
    out = {}
    for lvl in LEVELS:
        raw = level_name_maps.get(mapping[lvl], {})
        # out[lvl] = {int(k): str(v) for k, v in raw.items()}
        out[lvl] = {
            int(k): PAPER_LABEL_MAP.get(str(v), str(v))
            for k, v in raw.items()
        }
    return out


def global_to_local(level: str, global_idx: int) -> int:
    start, _ = config.GLOBAL_INDEX_RANGES[level]
    return int(global_idx) - start


def local_to_global(level: str, local_idx: int) -> int:
    start, _ = config.GLOBAL_INDEX_RANGES[level]
    return start + int(local_idx)


def load_full_model(weights_path: Path, device: torch.device):
    print(f"Loading full model from: {weights_path}")
    net = torch.load(weights_path, map_location=device, weights_only=False)
    net.to(device)
    net.eval()
    return net


def compute_relation_to_gt_depth(pred_level_idx: int, true_level_idx: int) -> str:
    return level_depth_relation(pred_level_idx, true_level_idx)


def write_excel_workbook(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)


def write_pdf_confusion_tables(path: Path, confusion_tables: dict[str, pd.DataFrame], title_prefix: str) -> None:
    with PdfPages(path) as pdf:
        for lvl in LEVELS:
            df = confusion_tables[lvl]
            if df.empty:
                continue

            plot_df = df.set_index("true_class")
            data = plot_df.values

            fig, ax = plt.subplots(figsize=(max(8, 1 + 0.7 * len(plot_df.columns)),
                                            max(6, 1 + 0.45 * len(plot_df.index))))
            im = ax.imshow(data, cmap="Blues", aspect="auto")
            fig.colorbar(im, ax=ax)

            ax.set_xticks(np.arange(len(plot_df.columns)))
            ax.set_xticklabels(plot_df.columns, rotation=45, ha="right")
            ax.set_yticks(np.arange(len(plot_df.index)))
            ax.set_yticklabels(plot_df.index)
            ax.set_title(f"{title_prefix} - {LEVEL_DISPLAY[lvl]}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

            thresh = data.max() / 2 if data.size else 0
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    ax.text(
                        j, i, int(data[i, j]),
                        ha="center", va="center",
                        color="white" if data[i, j] > thresh else "black",
                        fontsize=8
                    )

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


# ============================================================
# ================= DATA ====================================
# ============================================================

test_transform = build_test_transform(
    image_size=config.PROGRESSIVE_RESIZE_SCHEDULE["final"],
    aug=config.TEST_AUGMENTATION,
)

testset = InsectDataset(
    config.get_test_list_path(),
    input_transform=test_transform,
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=config.DEFAULT_NUM_WORKERS // 2,
    pin_memory=True,
)

# ============================================================
# ================= DEVICE ==================================
# ============================================================

device = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")

# ============================================================
# ================= LOAD MODEL / TREE LOSS ===================
# ============================================================

net = load_full_model(WEIGHTS_PATH, device)

tree_loss = TreeLoss(
    hierarchy=config.HIERARCHY,
    device=device,
    run_folder=Path(config.RUN_FOLDER),
)

level_name_maps = load_level_name_maps(config.LEVEL_NAME_MAPS_FILE, Path(config.RUN_FOLDER))
label_lookup = build_label_lookup(level_name_maps)

# ============================================================
# ================= OUTPUT PATHS =============================
# ============================================================

ANALYSIS_DIR = SCRIPT_DIR.parent / "analysis_paper"
ANALYSIS_DIR.mkdir(exist_ok=True)

thr_str = fmt_thr(CONF_THRESHOLD)
timestamp = timestamp_str()

LEVEL_SUMMARY_CSV = ANALYSIS_DIR / f"analysis_{timestamp}_thr{thr_str}_level_summary.csv"
CLASS_METRICS_CSV = ANALYSIS_DIR / f"analysis_{timestamp}_thr{thr_str}_class_metrics.csv"
CONF_COVERAGE_XLSX = ANALYSIS_DIR / f"analysis_{timestamp}_thr{thr_str}_confusion_coverage.xlsx"
CONF_EXTENDED_XLSX = ANALYSIS_DIR / f"analysis_{timestamp}_thr{thr_str}_confusion_extended.xlsx"
CONF_PDF = ANALYSIS_DIR / f"analysis_{timestamp}_thr{thr_str}_confusion_matrices.pdf"
CONF_EXTENDED_PDF = ANALYSIS_DIR / f"analysis_{timestamp}_thr{thr_str}_confusion_matrices_extended.pdf"

print(f"Confidence threshold: {CONF_THRESHOLD}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Output directory: {ANALYSIS_DIR}")
print(f"Model: {WEIGHTS_PATH}")

# ============================================================
# ================= ANALYSIS STATE ===========================
# ============================================================

all_level_events = defaultdict(list)

# # Accumulate raw MAP-vs-GT depth relation across all batches
# depth_relation_counts = {
#     "shorter_than_gt": 0,
#     "same_depth_as_gt": 0,
#     "longer_than_gt": 0,
# }

LEVEL_ORDER = ["pf4", "pf3", "pf2", "pf1", "leaf"]

def build_mask_dict(targets_dict):
    return {k: (v != config.SENTINEL_VALUE) for k, v in targets_dict.items()}


@torch.no_grad()
def decode_batch(inputs, targets, trees, run_folder):
    pf4_t, pf3_t, pf2_t, pf1_t, leaf_t = get_5_level_targets(
        targets, device, config.DATASET_NAME, trees, run_folder=run_folder
    )
    # print("Generated hierarchical targets")

    targets_dict = {
        "pf4": pf4_t,
        "pf3": pf3_t,
        "pf2": pf2_t,
        "pf1": pf1_t,
        "leaf": leaf_t,
    }

    masks_dict = build_mask_dict(targets_dict)

    outputs = net(inputs, masks_dict=masks_dict)
    pf4_logits, pf3_logits, pf2_logits, pf1_logits, leaf_logits = outputs
    # print("Forward pass done")

    pf4_sig = torch.sigmoid(pf4_logits)
    pf3_sig = torch.sigmoid(pf3_logits)
    pf2_sig = torch.sigmoid(pf2_logits)
    pf1_sig = torch.sigmoid(pf1_logits)
    leaf_sig = torch.sigmoid(leaf_logits)

    combined_output = torch.cat([pf4_sig, pf3_sig, pf2_sig, pf1_sig, leaf_sig], dim=1)
    # print("Sigmoid + concat done")

    return targets_dict, combined_output

@torch.no_grad()
def hierarchical_predict_map_backoff_sigmoid(
    fs_sigmoid,
    tree_loss_module,
    targets_per_level,
    level_order,
    global_index_ranges,
    device,
    threshold,
    sentinel=-1,
):
    """
    MAP path inference + GT-bounded hierarchical back-off.
    Thresholding is applied after GT-depth alignment.

    Returns:
        truncated_map_paths: per-level MAP path nodes, truncated by GT depth
        accepted_level_idx: level at which prediction was accepted, or -1
        accepted_node_global: final accepted node global id, or -1
        threshold_rejected: per-level rejection flags
        depth_relation: -1 shallower, 0 same, +1 deeper
    """

    B = fs_sigmoid.shape[0]
    S = tree_loss_module.stateSpace_unweighted.to(device)

    # ---------- MAP decode (tree-consistent) ----------
    scores = torch.matmul(S, fs_sigmoid.T)          # [states, batch]
    best_state_idx = scores.argmax(dim=0)           # [batch]
    best_states = S[best_state_idx]                 # [batch, N]

    # Per-level MAP path nodes
    map_paths = {}
    for lvl in level_order:
        start, end = global_index_ranges[lvl]
        ids = torch.arange(start, end, device=device)
        active = best_states[:, ids] > 0

        preds = torch.full((B,), -1, device=device, dtype=torch.long)
        has_any = active.any(dim=1)
        idx_in_level = active.float().argmax(dim=1)
        preds[has_any] = ids[idx_in_level[has_any]]
        map_paths[lvl] = preds

    # ---------- GT depth ----------
    avail = torch.stack(
        [(targets_per_level[lvl] != sentinel) for lvl in level_order],
        dim=1
    )
    gt_depth = (
        (len(level_order) - 1)
        - torch.flip(avail.int(), dims=[1]).argmax(dim=1)
    )

    # ---------- MAP depth ----------
    map_depth = torch.full((B,), -1, device=device, dtype=torch.long)
    for i, lvl in enumerate(level_order):
        map_depth = torch.where(map_paths[lvl] >= 0, torch.full_like(map_depth, i), map_depth)

    # ---------- depth relation ----------
    depth_relation = torch.zeros(B, dtype=torch.long, device=device)
    depth_relation[map_depth < gt_depth] = -1   # shallower
    depth_relation[map_depth == gt_depth] = 0   # same
    depth_relation[map_depth > gt_depth] = 1    # deeper

    # ---------- Truncate MAP path at GT depth ----------
    truncated_paths = {}
    for i, lvl in enumerate(level_order):
        preds = map_paths[lvl].clone()
        preds[gt_depth < i] = -1
        truncated_paths[lvl] = preds

    # ---------- Hierarchical back-off ----------
    accepted_level = torch.full((B,), -1, device=device, dtype=torch.long)
    accepted_node = torch.full((B,), -1, device=device, dtype=torch.long)

    threshold_rejected = {
        lvl: torch.zeros(B, dtype=torch.bool, device=device)
        for lvl in level_order
    }

    force_accept_root = True

    # Walk from deepest to coarsest
    for d in reversed(range(len(level_order))):
        lvl = level_order[d]

        # Only consider samples whose GT reaches this depth and which are not yet accepted
        mask = (accepted_level == -1) & (gt_depth >= d)
        if not mask.any():
            continue

        batch_idx = torch.where(mask)[0]
        candidates = truncated_paths[lvl][mask]          # [M]
        valid = candidates >= 0                          # [M]

        confs = torch.zeros(candidates.shape[0], dtype=torch.float32, device=device)

        if valid.any():
            valid_batch_idx = batch_idx[valid]           # original batch indices
            valid_candidates = candidates[valid]         # global node ids

            # Safe explicit gather: one confidence per valid sample/candidate
            confs[valid] = fs_sigmoid[valid_batch_idx, valid_candidates]

        if d == 0 and force_accept_root:
            accept = valid
        else:
            accept = valid & (confs >= threshold)

        reject = valid & (~accept)

        # Record threshold rejections at this level
        if reject.any():
            threshold_rejected[lvl][batch_idx[reject]] = True

        # Accept this level for passing samples
        if accept.any():
            accepted_batch_idx = batch_idx[accept]
            accepted_level[accepted_batch_idx] = d
            accepted_node[accepted_batch_idx] = candidates[accept]

    return {
        "truncated_map_paths": truncated_paths,
        "accepted_level_idx": accepted_level,
        "accepted_node_global": accepted_node,
        "threshold_rejected": threshold_rejected,
        "depth_relation": depth_relation,
    }

# ============================================================
# ================= EVALUATION LOOP ==========================
# ============================================================

print("Starting evaluation...")

with torch.no_grad():
    for batch_idx, (inputs, targets, indices, class_names) in enumerate(testloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # ----------------------------------------------------
        # Forward pass + sigmoid
        # ----------------------------------------------------
        targets_dict, combined_output = decode_batch(
            inputs,
            targets,
            config.HIERARCHY,
            Path(config.RUN_FOLDER),
        )

        fs_sigmoid = combined_output  # already sigmoid in decode_batch

        # ----------------------------------------------------
        # Hierarchical MAP inference (GT-aware, sigmoid-thresholded)
        # ----------------------------------------------------
        results = hierarchical_predict_map_backoff_sigmoid(
            fs_sigmoid=fs_sigmoid,
            tree_loss_module=tree_loss,
            targets_per_level=targets_dict,
            level_order=LEVEL_ORDER,
            global_index_ranges=config.GLOBAL_INDEX_RANGES,
            device=device,
            threshold=CONF_THRESHOLD,
            sentinel=config.SENTINEL_VALUE,
        )

        # # Update raw MAP-vs-GT depth relation totals
        # dr = results["depth_relation"]
        # depth_relation_counts["shorter_than_gt"] += int((dr < 0).sum().item())
        # depth_relation_counts["same_depth_as_gt"] += int((dr == 0).sum().item())
        # depth_relation_counts["longer_than_gt"] += int((dr > 0).sum().item())

        # ----------------------------------------------------
        # Emit per-level evaluation rows (GT bounded)
        # ----------------------------------------------------
        for lvl_idx, lvl in enumerate(LEVEL_ORDER):
            gt = targets_dict[lvl]
            valid = (gt != config.SENTINEL_VALUE)
            if not valid.any():
                continue

            preds = results["truncated_map_paths"][lvl]
            acc_lvl = results["accepted_level_idx"]

            start, _ = config.GLOBAL_INDEX_RANGES[lvl]

            for i in torch.where(valid)[0].tolist():
                true_local = int(gt[i].item())

                # Compute the GT deepest level index directly from targets_dict.
                gt_depth_i = 0
                for d, lvl_name in enumerate(LEVEL_ORDER):
                    if targets_dict[lvl_name][i].item() != config.SENTINEL_VALUE:
                        gt_depth_i = d

                # depth_relation encodes: -1 = MAP shallower than GT, 0 = same, +1 = deeper.
                dr = int(results["depth_relation"][i].item())

                if dr >= 0:
                    # MAP reached at least GT depth — raw MAP depth is >= gt_depth_i
                    raw_map_depth_i = gt_depth_i  # we only need to know it's >= gt_depth_i
                else:
                    # MAP was shallower. Find its actual depth from truncated paths,
                    # which for levels <= gt_depth_i are NOT zeroed by truncation.
                    raw_map_depth_i = -1
                    for d, lvl_name in enumerate(LEVEL_ORDER):
                        if d > gt_depth_i:
                            break
                        if results["truncated_map_paths"][lvl_name][i].item() >= 0:
                            raw_map_depth_i = d

                # Per-level depth-relation flags:
                # - shorter_than_gt: MAP stopped before GT depth, and this level is beyond where MAP stopped
                # - longer_than_gt: MAP went deeper than GT, and this is the GT deepest level row
                # - same_depth_as_gt: everything else (GT valid here, MAP reached this level)
                is_shallower = (dr < 0) and (lvl_idx > raw_map_depth_i)
                is_deeper = (dr > 0) and (lvl_idx == gt_depth_i)
                is_same = not is_shallower and not is_deeper

                pred_global_t = preds[i]
                pred_global = int(pred_global_t.item()) if pred_global_t >= 0 else None
                pred_local = int(pred_global - start) if pred_global is not None else None

                # This level is part of the accepted path if the final accepted
                # depth is this level or deeper.
                path_accepted_here_or_deeper = (
                        pred_global is not None
                        and acc_lvl[i].item() >= lvl_idx
                )

                if path_accepted_here_or_deeper:
                    conf = float(fs_sigmoid[i, pred_global].item())

                    # -------------------------------
                    # RELATED ERROR COMPUTATION
                    # -------------------------------
                    related_error = False
                    if pred_local != true_local and lvl_idx > 0:
                        parent_lvl = LEVEL_ORDER[lvl_idx - 1]
                        parent_gt = targets_dict[parent_lvl][i]
                        parent_pred_global = results["truncated_map_paths"][parent_lvl][i]

                        if parent_gt != config.SENTINEL_VALUE and parent_pred_global >= 0:
                            parent_start, _ = config.GLOBAL_INDEX_RANGES[parent_lvl]
                            parent_pred_local = int(parent_pred_global.item() - parent_start)
                            related_error = (parent_pred_local == int(parent_gt.item()))

                    all_level_events[LEVEL_DISPLAY[lvl]].append({
                        "level": LEVEL_DISPLAY[lvl],
                        "class_id": true_local,
                        "pred_local": pred_local,
                        "pred_global": pred_global,
                        "predicted": True,
                        "threshold_rejected": False,
                        "too_shallow": False,
                        "shorter_than_gt": is_shallower,
                        "same_depth_as_gt": is_same,
                        "longer_than_gt": is_deeper,
                        "correct": (pred_local == true_local),
                        "incorrect": (pred_local != true_local),
                        "related_error": related_error,
                        "confidence": conf,
                    })

                elif results["threshold_rejected"][lvl][i]:
                    # Prediction was made but rejected due to low confidence
                    all_level_events[LEVEL_DISPLAY[lvl]].append({
                        "level": LEVEL_DISPLAY[lvl],
                        "class_id": true_local,
                        "pred_local": None,
                        "pred_global": None,
                        "predicted": False,
                        "threshold_rejected": True,
                        "too_shallow": False,
                        "shorter_than_gt": is_shallower,
                        "same_depth_as_gt": is_same,
                        "longer_than_gt": is_deeper,
                        "correct": False,
                        "incorrect": False,
                        "related_error": False,
                        "confidence": None,
                    })

                else:
                    # MAP never reached this level (too shallow)
                    all_level_events[LEVEL_DISPLAY[lvl]].append({
                        "level": LEVEL_DISPLAY[lvl],
                        "class_id": true_local,
                        "pred_local": None,
                        "pred_global": None,
                        "predicted": False,
                        "threshold_rejected": False,
                        "too_shallow": True,
                        "shorter_than_gt": True,
                        "same_depth_as_gt": False,
                        "longer_than_gt": False,
                        "correct": False,
                        "incorrect": False,
                        "related_error": False,
                        "confidence": None,
                    })

# ============================================================
# ================= METRIC TABLES ============================
# ============================================================

level_summary_rows = []
class_rows = []
coverage_tables = {}
extended_tables = {}

# ------------------------------------------------------------
# Dataset-level depth relation counts
# These are based on the raw MAP path depth relation, not on
# final accepted predictions and not on per-level rows.
# ------------------------------------------------------------
depth_relation_tensor = results["depth_relation"]
depth_shorter_count = int((depth_relation_tensor < 0).sum().item())
depth_same_count = int((depth_relation_tensor == 0).sum().item())
depth_longer_count = int((depth_relation_tensor > 0).sum().item())

for lvl in LEVELS:
    rows = all_level_events[LEVEL_DISPLAY[lvl]]
    if not rows:
        continue

    df = pd.DataFrame(rows)

    total_samples = len(df)
    predicted_df = df[df["predicted"]].copy()

    coverage_count = int(predicted_df.shape[0])
    coverage_pct = coverage_count / total_samples if total_samples else 0.0

    # ========================================================
    # CORRECTED Level-aggregate metrics
    # ========================================================
    # TP: correct predictions
    # FP: incorrect predictions (predicted something but got it wrong)
    # FN: GT exists but either not predicted or predicted incorrectly
    #     = threshold_rejected + too_shallow + incorrect predictions
    # TN: not applicable at aggregate level (needs per-class)

    TP = int(predicted_df["correct"].sum())
    FP = int(predicted_df["incorrect"].sum())

    # FN includes all samples where we failed to predict correctly:
    # - threshold rejected
    # - too shallow (MAP didn't reach)
    # - predicted but wrong (already counted as FP, but also counts as FN for the true class)
    # FN = int((~df["correct"]).sum()) - FP  # All non-correct minus those that were predicted wrong
    # Simpler: FN = total_samples - TP (everything that's not a true positive is a false negative for its true class)
    FN = total_samples - TP

    # TP = int(predicted_df["correct"].sum())
    # FP = int(predicted_df["incorrect"].sum())
    # FN = int((df["threshold_rejected"] | df["too_shallow"]).sum())

    # Accuracy should be based on coverage, not all valid samples
    accuracy_on_covered = TP / max(1, coverage_count)

    precision = TP / max(1, TP + FP)
    recall = TP / max(1, TP + FN)
    f1 = (
        2 * precision * recall / max(1e-8, precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # MCC should be computed from actual covered predictions
    if coverage_count > 0:
        y_true_cov = predicted_df["class_id"].astype(int).to_numpy()
        y_pred_cov = predicted_df["pred_local"].astype(int).to_numpy()
        mcc = matthews_corrcoef(y_true_cov, y_pred_cov)
    else:
        mcc = 0.0

    # Confidence only for accepted/predicted rows
    avg_conf = (
        float(predicted_df["confidence"].mean())
        if coverage_count > 0
        else 0.0
    )

    level_summary_rows.append(
        {
            "level": lvl,
            "total_samples": total_samples,
            "coverage_count": coverage_count,
            "coverage_pct": coverage_pct,
            "accuracy_on_covered": accuracy_on_covered,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mcc": mcc,
            "too_shallow_count": int(df["too_shallow"].sum()),
            "too_shallow_pct": float(df["too_shallow"].mean()),
            "threshold_rejected_count": int(df["threshold_rejected"].sum()),
            "threshold_rejected_pct": float(df["threshold_rejected"].mean()),
            # Depth-relation counts summed from per-row flags, giving correct per-level totals.
            # These are equivalent to the sum of the per-class counts for this level.
            "shorter_than_gt_count": int(df["shorter_than_gt"].sum()),
            "shorter_than_gt_pct": float(df["shorter_than_gt"].mean()),
            "same_depth_as_gt_count": int(df["same_depth_as_gt"].sum()),
            "same_depth_as_gt_pct": float(df["same_depth_as_gt"].mean()),
            "longer_than_gt_count": int(df["longer_than_gt"].sum()),
            "longer_than_gt_pct": float(df["longer_than_gt"].mean()),
            "avg_sigmoid_confidence_above_threshold": avg_conf,
        }
    )

    # ========================================================
    # Per-class metrics (one-vs-rest for each class)
    # ========================================================
    for cls in sorted(df["class_id"].unique()):
        cls_df = df[df["class_id"] == cls]
        support_true = int(len(cls_df))

        # For this class (one-vs-rest):
        # TP: samples where GT=cls AND predicted correctly as cls
        TP_c = int((cls_df["predicted"] & cls_df["correct"]).sum())

        # FN: samples where GT=cls but NOT correctly predicted
        # Includes: predicted wrong, threshold_rejected, too_shallow
        FN_c = support_true - TP_c

        # FP: samples where GT≠cls but predicted AS cls
        # Look at OTHER classes' samples that were wrongly predicted as THIS class
        other_classes_df = df[df["class_id"] != cls]
        FP_c = int(
            (other_classes_df["predicted"] &
             (other_classes_df["pred_local"] == cls)).sum()
        )

        # TN: samples where GT≠cls AND (NOT predicted as cls)
        # = other class samples that were either:
        #   - predicted as a different class (not cls)
        #   - threshold rejected
        #   - too shallow
        other_classes_count = len(other_classes_df)
        TN_c = other_classes_count - FP_c

        # Coverage: how many of this class's samples got any prediction
        coverage_c = int(cls_df["predicted"].sum())

        # Standard binary classification metrics
        precision_c = TP_c / max(1, TP_c + FP_c)
        recall_c = TP_c / max(1, TP_c + FN_c)
        f1_c = (
            2 * precision_c * recall_c / max(1e-8, precision_c + recall_c)
            if (precision_c + recall_c) > 0
            else 0.0
        )

        # Accuracy: (correct predictions) / (all samples)
        accuracy_c = (TP_c + TN_c) / max(1, total_samples)

        # MCC: Matthews Correlation Coefficient
        if total_samples > 0:
            numerator = (TP_c * TN_c) - (FP_c * FN_c)
            denominator = math.sqrt(
                (TP_c + FP_c) * (TP_c + FN_c) * (TN_c + FP_c) * (TN_c + FN_c)
            )
            mcc_c = numerator / max(1e-8, denominator)
        else:
            mcc_c = 0.0

        # Confidence statistics (only for predicted samples of this class)
        cls_pred_df = cls_df[cls_df["predicted"]]
        avg_conf_cls = (
            float(cls_pred_df["confidence"].mean())
            if len(cls_pred_df) > 0
            else 0.0
        )
        std_conf_cls = (
            float(cls_pred_df["confidence"].std(ddof=0))
            if len(cls_pred_df) > 1
            else 0.0
        )

        # Related error: incorrect predictions where parent was correct
        related_error_count = int(cls_pred_df["related_error"].sum()) if "related_error" in cls_pred_df.columns else 0
        incorrect_count_for_this_class = int((cls_pred_df["incorrect"]).sum())
        related_error_rate = (
                    related_error_count / incorrect_count_for_this_class) if incorrect_count_for_this_class > 0 else 0.0

        # Depth-relation counts for this class
        shorter_count = int(cls_df["shorter_than_gt"].sum())
        same_count = int(cls_df["same_depth_as_gt"].sum())
        longer_count = int(cls_df["longer_than_gt"].sum())

        class_rows.append(
            {
                "level": lvl,
                "class_id": cls,
                "class_name": label_lookup[lvl].get(cls, f"Class {cls}"),
                "support_true": support_true,
                "predicted_count": coverage_c,
                "correct_count": TP_c,
                "incorrect_count": int((cls_pred_df["incorrect"]).sum()),
                "TP": TP_c,
                "FP": FP_c,
                "FN": FN_c,
                "TN": TN_c,
                "precision": precision_c,
                "recall": recall_c,
                "f1": f1_c,
                "accuracy": accuracy_c,
                "mcc": mcc_c,
                "avg_sigmoid_confidence_above_threshold": avg_conf_cls,
                "std_sigmoid_confidence_above_threshold": std_conf_cls,
                "related_error_count": related_error_count,
                "related_error_rate": related_error_rate,
                "shorter_than_gt_count": shorter_count,
                "same_depth_as_gt_count": same_count,
                "longer_than_gt_count": longer_count,
                "too_shallow_count": int(cls_df["too_shallow"].sum()),
                "threshold_rejected_count": int(cls_df["threshold_rejected"].sum()),
            }
        )

    # ========================================================
    # Confusion tables
    # ========================================================
    true_classes = sorted(df["class_id"].unique().tolist())
    pred_classes = sorted(df.loc[df["predicted"], "pred_local"].dropna().unique().tolist())
    pred_col_names = sorted(set(true_classes) | set(pred_classes))

    cov_rows = []
    ext_rows = []

    for cls in true_classes:
        cov_row = {"true_class": label_lookup[lvl].get(cls, str(cls))}
        ext_row = {"true_class": label_lookup[lvl].get(cls, str(cls))}
        for p in pred_col_names:
            nm = label_lookup[lvl].get(p, str(p))
            cov_row[nm] = 0
            ext_row[nm] = 0
        ext_row["REJECT_THRESHOLD"] = 0
        ext_row["MISSING_TOO_SHALLOW"] = 0
        cov_rows.append(cov_row)
        ext_rows.append(ext_row)

    cls_to_idx = {cls: idx for idx, cls in enumerate(true_classes)}

    for _, row in df.iterrows():
        idx = cls_to_idx[row["class_id"]]
        if row["predicted"] and row["pred_local"] is not None:
            nm = label_lookup[lvl].get(int(row["pred_local"]), str(int(row["pred_local"])))
            cov_rows[idx][nm] += 1
            ext_rows[idx][nm] += 1
        elif row["threshold_rejected"]:
            ext_rows[idx]["REJECT_THRESHOLD"] += 1
        elif row["too_shallow"]:
            ext_rows[idx]["MISSING_TOO_SHALLOW"] += 1

    coverage_tables[lvl] = pd.DataFrame(cov_rows)
    extended_tables[lvl] = pd.DataFrame(ext_rows)

# ============================================================
# ================= SAVE OUTPUTS =============================
# ============================================================

level_summary_df = pd.DataFrame(level_summary_rows)
class_metrics_df = pd.DataFrame(class_rows)

level_summary_df.to_csv(LEVEL_SUMMARY_CSV, index=False)
class_metrics_df.to_csv(CLASS_METRICS_CSV, index=False)

write_excel_workbook(
    CONF_COVERAGE_XLSX,
    {LEVEL_DISPLAY[lvl]: df for lvl, df in coverage_tables.items()}
)

write_excel_workbook(
    CONF_EXTENDED_XLSX,
    {LEVEL_DISPLAY[lvl]: df for lvl, df in extended_tables.items()}
)

write_pdf_confusion_tables(
    CONF_PDF,
    coverage_tables,
    title_prefix=f"Coverage Confusion (thr={CONF_THRESHOLD})"
)

write_pdf_confusion_tables(
    CONF_EXTENDED_PDF,
    extended_tables,
    title_prefix=f"Extended Confusion (thr={CONF_THRESHOLD})"
)

print("\nDone.")
print(f"Level summary CSV      : {LEVEL_SUMMARY_CSV}")
print(f"Class metrics CSV      : {CLASS_METRICS_CSV}")
print(f"Coverage confusion XLSX: {CONF_COVERAGE_XLSX}")
print(f"Extended confusion XLSX: {CONF_EXTENDED_XLSX}")
print(f"Confusion PDF          : {CONF_PDF}")
print(f"Extended confusion PDF  : {CONF_EXTENDED_PDF}")