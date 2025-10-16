import csv
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def log_per_class_metrics(epoch, level_metrics_dict, label_maps, split, csv_filename="per_class_metrics_MobNet_15Oct25.csv", sentinel=-1):
    # Check if file exists to determine if header is needed
    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Write header if file is new
        if not file_exists:
            writer.writerow(["Epoch", "Split", "Level", "Class ID", "Class Name", "Precision", "Recall", "F1 Score"])

        # Iterate over each level
        for level_name, (y_true, y_pred) in level_metrics_dict.items():
            # Filter invalid entries
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            valid_mask = y_true != sentinel
            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]

            # Compute metrics
            precision = precision_score(y_true, y_pred, average=None, zero_division=0)
            recall = recall_score(y_true, y_pred, average=None, zero_division=0)
            f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
            unique_classes = sorted(set(y_true))

            # Get label map for current level
            label_map = label_maps.get(level_name, {})

            # Write per-class metrics
            for i, cls in enumerate(unique_classes):
                class_name = label_map.get(str(cls), f"Class {cls}")
                writer.writerow([
                    epoch,
                    split,
                    level_name,
                    cls,
                    class_name,
                    round(precision[i], 4),
                    round(recall[i], 4),
                    round(f1[i], 4)
                ])

def log_epoch_summary_to_csv(
        epoch,
        train_pf4_acc,
        train_pf3_acc,
        train_pf2_acc,
        train_pf1_acc,
        train_leaf_acc_soft,
        train_leaf_acc_sig,
        train_loss,
        test_pf4_acc,
        test_pf3_acc,
        test_pf2_acc,
        test_pf1_acc,
        test_leaf_acc_soft,
        test_leaf_acc_sig,
        test_loss,
        epoch_time,
        csv_filename="epoch_summary_metrics_MobNet_15Oct25.csv"
):
    # Define the header and row
    header = [
        "Epoch",
        "Train PF4 Acc", "Train PF3 Acc", "Train PF2 Acc", "Train PF1 Acc",
        "Train Leaf Acc Soft", "Train Leaf Acc Sig", "Train Loss",
        "Test PF4 Acc", "Test PF3 Acc", "Test PF2 Acc", "Test PF1 Acc",
        "Test Leaf Acc Soft", "Test Leaf Acc Sig", "Test Loss",
        "Epoch Time (s)"
    ]

    row = [
        epoch,
        train_pf4_acc, train_pf3_acc, train_pf2_acc, train_pf1_acc,
        train_leaf_acc_soft, train_leaf_acc_sig, train_loss,
        test_pf4_acc, test_pf3_acc, test_pf2_acc, test_pf1_acc,
        test_leaf_acc_soft, test_leaf_acc_sig, test_loss,
        epoch_time
    ]

    # Check if file exists
    file_exists = os.path.isfile(csv_filename)

    # Write to CSV
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)