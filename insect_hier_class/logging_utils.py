"""
Logging utilities for training metrics.
Saves per-class and per-epoch metrics to CSV files.
"""

import matplotlib.pyplot as plt
import csv
import numpy as np
import fitz #PyMuPDF
from pathlib import Path
from hierarchical_classification_metrics import compute_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

import config


def log_per_class_metrics(epoch, level_metrics_dict, label_maps, split, run_folder, 
                          csv_filename=None, sentinel=-1, level_confidences=None, related_info=None):
    """
    Log per-class metrics to CSV file.
    
    Args:
        epoch: Current epoch number
        level_metrics_dict: Dict mapping level names to (y_true, y_pred) tuples
        label_maps: Dict mapping level names to label-to-name mappings
        split: 'Train' or 'Test'
        run_folder: Path to run folder
        csv_filename: CSV filename (uses default from config if None)
        sentinel: Sentinel value for invalid labels
        level_confidences: Optional dict mapping level names to arrays of confidences
        related_info: Optional dict mapping level names to lists of (true_id, was_related) tuples
    """
    if csv_filename is None:
        csv_filename = config.PER_CLASS_METRICS_CSV
    
    csv_path = run_folder / csv_filename
    file_exists = csv_path.exists()
    
    with open(csv_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write header if file is new
        if not file_exists:
            writer.writerow([
                "Epoch", "Split", "Level", "Class ID", "Class Name", 
                "Precision", "Recall", "F1 Score", "Avg Confidence", "Related Err %"
            ])

        # Iterate over each level
        for level_name, (y_true, y_pred) in level_metrics_dict.items():
            # 1. Convert to arrays
            y_true_arr = np.array(y_true)
            y_pred_arr = np.array(y_pred)

            # 2. Derive valid mask (sentinel check)
            valid_mask = y_true_arr != sentinel

            # 3. Final metrics data
            y_t_final = y_true_arr[valid_mask]
            y_p_final = y_pred_arr[valid_mask]

            if len(y_t_final) == 0:
                continue

            # 4. Filter and Align Confidences
            # Dictionary-based alignment: Class ID -> List of confidences
            class_conf_map = {}
            if level_confidences and level_name in level_confidences:
                confs = np.array(level_confidences[level_name])

                # If lengths match exactly (pre-filtered by compute_hierarchical_accuracy_with_inference)
                if len(confs) == len(y_t_final):
                    for tid, c_val in zip(y_t_final, confs):
                        if tid not in class_conf_map: class_conf_map[tid] = []
                        class_conf_map[tid].append(c_val)
                # Fallback for legacy behavior (raw batch mask)
                elif len(confs) == len(valid_mask):
                    filtered_confs = confs[valid_mask]
                    for tid, c_val in zip(y_t_final, filtered_confs):
                        if tid not in class_conf_map: class_conf_map[tid] = []
                        class_conf_map[tid].append(c_val)

            # 5. Compute metrics using the aligned data
            precision = precision_score(y_t_final, y_p_final, average=None, zero_division=0)
            recall = recall_score(y_t_final, y_p_final, average=None, zero_division=0)
            f1 = f1_score(y_t_final, y_p_final, average=None, zero_division=0)
            unique_classes = sorted(set(y_t_final))

            # Get label map for current level
            label_map = label_maps.get(level_name, {})

            # Extract related error info from level_metrics_dict context or counters
            related_map = {}
            rel_source = related_info.get(level_name) or related_info.get(level_name.upper()) or related_info.get(level_name.lower())
            if rel_source:
                for tid, was_rel in rel_source:
                    if tid not in related_map:
                        related_map[tid] = []
                    related_map[tid].append(was_rel)
            
            # Write per-class metrics
            for i, cls in enumerate(unique_classes):
                if i < len(precision):
                    class_name = label_map.get(str(cls), f"Class {cls}")

                    # Calculate average confidence for this class
                    avg_conf = 0.0
                    if cls in class_conf_map:
                        avg_conf = np.mean(class_conf_map[cls])

                    # Calculate related error % for this class
                    rel_pct = 0.0
                    if cls in related_map:
                        rel_pct = np.mean(related_map[cls])
                    elif precision[i] >= 1.0 and recall[i] >= 1.0:
                        # If accuracy is 100% for this class, related accuracy is technically 1.0
                        rel_pct = 1.0

                    writer.writerow([
                        epoch,
                        split,
                        level_name,
                        cls,
                        class_name,
                        round(precision[i], 4),
                        round(recall[i], 4),
                        round(f1[i], 4),
                        round(float(avg_conf), 4),
                        round(float(rel_pct), 4)
                    ])


def log_epoch_summary_to_csv(
    epoch,
    train_pf4_acc,
    train_pf3_acc,
    train_pf2_acc,
    train_pf1_acc,
    train_leaf_acc,
    train_loss,
    train_tree_loss,
    train_ce_loss,
    # train_entropy_loss,
    test_pf4_acc,
    test_pf3_acc,
    test_pf2_acc,
    test_pf1_acc,
    test_leaf_acc,
    test_loss,
    test_tree_loss,
    test_ce_loss,
    # test_entropy_loss,
    epoch_time,
    run_folder,
    csv_filename=None  # Optional override
):
    """
   Log epoch summary metrics to CSV file.

   Args:
       epoch: Current epoch number
       train_*_acc: Training accuracies for each level
       train_loss: Training loss
       test_*_acc: Test accuracies for each level
       test_loss: Test loss
       epoch_time: Time taken for epoch (seconds)
       run_folder: Path to run folder
       csv_filename: CSV filename (uses default from config if None)
   """

    # Use default from config if no filename is provided
    if csv_filename is None:
        csv_filename = config.EPOCH_SUMMARY_CSV

    # Construct full path to CSV file
    csv_path = run_folder / csv_filename

    # Define header and row
    header = [
        "Epoch",
        "Train PF4 Acc", "Train PF3 Acc", "Train PF2 Acc", "Train PF1 Acc", "Train Leaf Acc",
        "Train Loss", "Train Tree Loss", "Train CE Loss", # "Train Entropy Loss",
        "Test PF4 Acc", "Test PF3 Acc", "Test PF2 Acc", "Test PF1 Acc", "Test Leaf Acc",
        "Test Loss", "Test Tree Loss", "Test CE Loss", # "Test Entropy Loss",
        "Epoch Time (s)"
    ]

    row = [
        epoch,
        train_pf4_acc, train_pf3_acc, train_pf2_acc, train_pf1_acc, train_leaf_acc,
        train_loss, train_tree_loss, train_ce_loss, # train_entropy_loss,
        test_pf4_acc, test_pf3_acc, test_pf2_acc, test_pf1_acc, test_leaf_acc,
        test_loss, test_tree_loss, test_ce_loss, # test_entropy_loss,
        epoch_time
    ]

    # Check if file exists
    file_exists = csv_path.exists()

    # Write to CSV
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)



def log_confusion_matrices_to_pdf(epoch, level_metrics_dict, label_maps, split, run_folder):
    """
    Append confusion matrices to a single PDF file for each level and split per epoch.

    Args:
        epoch (int): Current epoch number
        level_metrics_dict (dict): Mapping level names to (y_true, y_pred)
        label_maps (dict): Mapping level names to label-to-name dictionaries
        split (str): 'Train' or 'Test'
        run_folder (Path): Target folder to write the PDF and temp images
    """
    pdf_path = run_folder / "confusion_matrices.pdf"
    temp_images = []

    for level_name, (y_true, y_pred) in level_metrics_dict.items():
        if len(y_true) == 0 or len(y_pred) == 0:
            continue

        cm = compute_confusion_matrix(y_true, y_pred)
        labels = sorted(set(y_true))
        label_names = [label_maps.get(level_name, {}).get(str(lbl), str(lbl)) for lbl in labels]

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(xticks=np.arange(len(label_names)),
               yticks=np.arange(len(label_names)),
               xticklabels=label_names,
               yticklabels=label_names,
               title=f"Confusion Matrix - {split} - {level_name} - Epoch {epoch}",
               ylabel='True label',
               xlabel='Predicted label')

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        fmt = 'd'
        thresh = cm.max() / 2.
        # Dynamically adjust font size based on number of classes
        font_size = max(8, 14 - len(label_names))  # Shrinks as classes increase

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=font_size)

        fig.tight_layout()
        image_path = config.RUN_FOLDER / f"temp_cm_{split}_{level_name}_epoch_{epoch}.png"
        fig.savefig(image_path)
        plt.close(fig)
        temp_images.append(image_path)

    # Append images to PDF
    doc = fitz.open()

    # If the file exists, load its pages into the new doc
    if pdf_path.exists():
        existing_doc = fitz.open(pdf_path)
        doc.insert_pdf(existing_doc)
        existing_doc.close()

    for img_path in temp_images:
        img_doc = fitz.open(img_path)
        pdf_bytes = img_doc.convert_to_pdf()
        img_pdf = fitz.open("pdf", pdf_bytes)
        doc.insert_pdf(img_pdf)
        img_path.unlink()

    doc.save(pdf_path)
    doc.close()

    print(f"Confusion matrices for epoch {epoch} ({split}) appended to {pdf_path}.")


def log_per_level_metrics(
    epoch,
    level_metrics_dict,
    split,
    run_folder,
    csv_filename=None,
    sentinel=-1,
    related_info=None
):
    """
    Log per-level aggregate metrics (Accuracy, Precision, Recall, F1, MCC) for a single split (Train or Test).
    Creates a CSV with columns: Epoch, Split, Metric, PF4, PF3, PF2, PF1, Leaf.

    Args:
        epoch (int): Current epoch number
        level_metrics_dict (dict): Mapping level names to (y_true, y_pred) for the split
        split (str): 'Train' or 'Test'
        run_folder (Path): Path to run folder
        csv_filename (str): Optional override for CSV filename (uses default from config if None)
        sentinel (int): Value used for invalid labels
        related_info (dict): Optional mapping of level names to related error info (true_id, was_related)
    """
    # Use default from config if no filename provided
    if csv_filename is None:
        csv_filename = config.PER_LEVEL_METRICS_CSV  # Add this constant in config.py

    csv_path = run_folder / csv_filename
    file_exists = csv_path.exists()

    levels = ["PF4", "PF3", "PF2", "PF1", "Leaf"]
    metrics = ["Accuracy", "Precision", "Recall", "F1", "MCC", "RelatedAccuracy"]

    header = ["Epoch", "Split", "Metric"] + levels

    # Compute metrics for this split
    results = {metric: [] for metric in metrics}
    for lvl in levels:
        y_true, y_pred = level_metrics_dict.get(lvl, ([], []))
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        valid_mask = y_true != sentinel
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

        if len(y_true) == 0:
            for metric in metrics:
                results[metric].append(np.nan)
            continue

        results["Accuracy"].append(accuracy_score(y_true, y_pred))
        results["Precision"].append(precision_score(y_true, y_pred, average="macro", zero_division=0))
        results["Recall"].append(recall_score(y_true, y_pred, average="macro", zero_division=0))
        results["F1"].append(f1_score(y_true, y_pred, average="macro", zero_division=0))
        results["MCC"].append(matthews_corrcoef(y_true, y_pred))
        # Compute Global Relatedness for this level
        rel_acc = np.nan
        if related_info and lvl in related_info and len(related_info[lvl]) > 0:
            # Extract only the booleans from the (id, bool) tuples
            rel_bools = [item[1] for item in related_info[lvl]]
            rel_acc = np.mean(rel_bools)
        results["RelatedAccuracy"].append(rel_acc)

    # Write to CSV
    with open(csv_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)

        for metric in metrics:
            row = [epoch, split, metric] + [round(val, 4) if not np.isnan(val) else "" for val in results[metric]]
            writer.writerow(row)

