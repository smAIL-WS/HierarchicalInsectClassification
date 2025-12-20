
"""
Central configuration file for the hierarchical insect classification model.
All paths, hyperparameters, and architecture settings are defined here.

To run with a different date/run:
    - Update RUN_DATE below
    - All other scripts will automatically use the correct paths
"""

from pathlib import Path

# ============================================================================
# PROJECT STRUCTURE
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / 'runs'
MODELS_DIR = BASE_DIR / 'models_Insect'
PRETRAINED_DIR = BASE_DIR.parent / 'pre-trained'

# ============================================================================
# RUN CONFIGURATION (Main setting to change between runs)
# ============================================================================
RUN_DATE = '2025-12-19'  # CHANGE THIS for each new run
RUN_FOLDER = RUNS_DIR / RUN_DATE

# Logging controls for Hyperparameter optimization
LOG_PER_CLASS = False     # per-class CSVs
LOG_PER_LEVEL = True     # per-level CSVs
LOG_CONFUSION = False     # confusion matrix PDFs

# ============================================================================
# FILE PATHS (relative to RUN_FOLDER)
# ============================================================================
TRAIN_LIST_FILE = 'insect_train_list.txt'
VAL_LIST_FILE = 'insect_val_list.txt'
TEST_LIST_FILE = 'insect_test_list.txt'
HIERARCHY_TREE_FILE = 'hierarchy_tree.txt'
LEVEL_NAME_MAPS_FILE = 'level_name_maps.json'
LEAF_SAMPLE_COUNTS_FILE = 'leaf_sample_counts.json'
NODE_SAMPLE_COUNTS_FILE = 'node_sample_counts.json'

# Logging output files
EPOCH_SUMMARY_CSV = 'epoch_summary_metrics.csv'
PER_CLASS_METRICS_CSV = 'per_class_metrics.csv'
PER_LEVEL_METRICS_CSV = 'per_level_metrics.csv'

# ============================================================================
# HIERARCHY DEFINITION (Single source of truth)
# ============================================================================
HIERARCHY = [
    [0],
    [1],
    [0, 8],
    [0, 5],
    [0, 7],
    [0, 3],
    [0, 6],
    [0, 2, 17],
    [0, 2, 14],
    [0, 3, 10],
    [0, 5, 15],
    [0, 2, 13],
    [0, 2, 12],
    [0, 3, 11],
    [0, 4, 16],
    [0, 4, 9, 22],
    [0, 3, 10, 20],
    [0, 3, 11, 21],
    [0, 3, 10, 20, 27],
    [0, 3, 10, 19, 24],
    [0, 3, 10, 20, 26],
    [0, 4, 9, 18, 23],
    [0, 4, 9, 18, 33],
    [0, 3, 10, 20, 25],
    [0, 3, 10, 19, 30],
    [0, 3, 10, 20, 31],
    [0, 3, 10, 20, 29],
    [0, 3, 10, 20, 28],
    [0, 3, 10, 20, 32],
]

# Computed from hierarchy
TOTAL_NODES = max(max(path) for path in HIERARCHY) + 1

# ============================================================================
# HIERARCHY LEVELS
# ============================================================================
HIERARCHY_LEVELS = ['parent_folder_4', 'parent_folder_3', 'parent_folder_2', 'parent_folder_1', 'classification']

# Number of classes per level
NUM_CLASSES_PER_LEVEL = {
    'pf4': 2,
    'pf3': 7,
    'pf2': 9,
    'pf1': 5,
    'leaf': 11
}

# Automatically compute global index ranges
GLOBAL_INDEX_RANGES = {}
_start = 0
for level in ['pf4', 'pf3', 'pf2', 'pf1', 'leaf']:
    count = NUM_CLASSES_PER_LEVEL[level]
    GLOBAL_INDEX_RANGES[level] = (_start, _start + count)
    _start += count

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
# # Backbone configurations
# BACKBONE_CONFIGS = {
#     'resnet18': {'num_ftrs': 512, 'feature_size': 256},
#     'resnet34': {'num_ftrs': 512, 'feature_size': 256},
#     'resnet50': {'num_ftrs': 2048, 'feature_size': 1024},
#     'resnext101': {'num_ftrs': 2048, 'feature_size': 1024},
#     'efficientnetv2_s': {'num_ftrs': 1280, 'feature_size': 640},
#     'mobilenetv3_small': {'num_ftrs': 576, 'feature_size': 128},
# }

# Backbone configurations with per-level feature sizes
BACKBONE_CONFIGS = {
    'mobilenetv3_small': {
        'num_ftrs': 576,
        'feature_sizes': {'pf4':128, 'pf3':256, 'pf2':256, 'pf1':256, 'leaf':512}
    },
    'resnet18': {
        'num_ftrs': 512,
        'feature_sizes': {'pf4':256, 'pf3':256, 'pf2':256, 'pf1':512, 'leaf':512}
    },
    'resnet50': {
        'num_ftrs': 2048,
        'feature_sizes': {'pf4':256, 'pf3':256, 'pf2':512, 'pf1':512, 'leaf':1024}
    },
    'efficientnetv2_s': {
        'num_ftrs': 1280,
        'feature_sizes': {'pf4':256, 'pf3':256, 'pf2':512, 'pf1':640, 'leaf':1024}
    }
}

# Model hyperparameters
DROPOUT_RATE_CONV = 0.2
DROPOUT_RATE_FC = 0.65
FC_HIDDEN_DIM = 512

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
DEFAULT_BATCH_SIZE = 4096
DEFAULT_NUM_EPOCHS = 100
DEFAULT_NUM_WORKERS = 8
DEFAULT_SEED = 42

# Learning rates
LR_CLASSIFIER = 0.002
LR_BACKBONE = 0.0002
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# Learning rate scheduler
LR_SCHEDULER_STEP_SIZE = 60
LR_SCHEDULER_GAMMA = 0.1

# Progressive resizing schedule
PROGRESSIVE_RESIZE_SCHEDULE = {
    'initial': 96,
    'mid': 112,
    'final': 128,
    'epoch_thresholds': [10, 20]  # Switch at epochs 10 and 20
}

# ============================================================================
# LOSS FUNCTION PARAMETERS
# ============================================================================
TREE_LOSS_ALPHA = 0.2

TREE_LOSS_BETA = 0.999995  # For effective class weighting. Must be less than 1.

HIERARCHY_WEIGHT = 3.0

# Cross-Entropy alpha schedule
ALPHA_START = 1.2        # Initial CE alpha
ALPHA_TARGET = 0.4       # Final CE alpha
CE_WARMUP_EPOCHS = 30    # Total epochs for CE alpha decay
ALPHA_LOSS_INVERT = False # If True, deeper nodes (towards leaf classes) get higher weights

# Tree Loss schedule
TREE_LOSS_START = 1.0    # Initial Tree Loss weight
TREE_LOSS_TARGET = 1.0   # Final Tree Loss weight
TREE_LOSS_WARMUP_EPOCHS = 10  # Ramp-up duration for Tree Loss
TREE_LOSS_START_EPOCH = 0    # Epoch to start Tree Loss ramp-up

# Focal loss parameters
USE_FOCAL_LOSS = False
FOCAL_LOSS_GAMMA = 1.0  # Focus on hard examples
FOCAL_LOSS_ALPHA = 1.0  # Downweight easy negatives

# Cross-entropy loss label smoothing
LABEL_SMOOTHING = 0.05

# Sentinel value for invalid targets
SENTINEL_VALUE = -1

# ============================================================================
# BI-DIRECTIONAL LEARNING RATE SCHEDULE
# ============================================================================
BIDIRECTIONAL_FEATURE_WEIGHTS = {
    'pf4': 0.05,
    'pf3': 0.1,
    'pf2': 0.15,
    'pf1': 0.2
}


# ============================================================================
# DATA AUGMENTATION
# ============================================================================
# Training augmentation parameters
TRAIN_AUGMENTATION = {
    # Geometric base
    'horizontal_flip': True,
    'vertical_flip': True,              # newly added
    'rotation_degrees': 15,
    'affine_translate': (0.1, 0.1),     # fractions of (width, height)
    'affine_zoom_max': 1.2,             # zoom-in upper bound for RandomAffine scale=(1.0, affine_zoom_max)
    'affine_zoom_min': 1.0,

    # Photometric
    'color_jitter': {
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1
    },
    'gaussian_blur_prob': 0.05,         # very low probability
    'gaussian_kernel_size': 3,          # must be odd (e.g., 3 or 5)

    # Normalization
    'normalize_mean': (0.445995, 0.435061, 0.403322),   # dataset-specific mean values
    'normalize_std': (0.229014, 0.238985, 0.23883),     # dataset-specific std values

    # Implementation details / defaults
    'rotation_fill_value': (114, 111, 103),         # dataset-specific mean color
    'affine_fill_value': (114, 111, 103),           # dataset-specific mean color
}

# Test augmentation parameters
TEST_AUGMENTATION = {
    # Normalization
    'normalize_mean': (0.445995, 0.435061, 0.403322),
    'normalize_std': (0.229014, 0.238985, 0.23883),
}

# ============================================================================
# PRETRAINED WEIGHTS PATHS
# ============================================================================
PRETRAINED_WEIGHTS = {
    'resnet50': PRETRAINED_DIR / 'resnet50-19c8e357.pth',
}

# ============================================================================
# DATASET NAME
# ============================================================================
DATASET_NAME = 'Insect'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_full_path(filename):
    """Get full path for a file in the run folder."""
    return RUN_FOLDER / filename

def get_train_list_path():
    """Get path to training list file."""
    return get_full_path(TRAIN_LIST_FILE)

def get_val_list_path():
    """Get path to validation list file."""
    return get_full_path(VAL_LIST_FILE)

def get_test_list_path():
    """Get path to test list file."""
    return get_full_path(TEST_LIST_FILE)

def get_hierarchy_tree_path():
    """Get path to hierarchy tree file."""
    return get_full_path(HIERARCHY_TREE_FILE)

def get_level_name_maps_path():
    """Get path to level name maps JSON."""
    return get_full_path(LEVEL_NAME_MAPS_FILE)

def get_leaf_sample_counts_path():
    """Get path to leaf sample counts JSON."""
    return get_full_path(LEAF_SAMPLE_COUNTS_FILE)

def get_node_sample_counts_path():
    """Get path to node sample counts JSON."""
    return get_full_path(NODE_SAMPLE_COUNTS_FILE)

def print_config():
    """Print current configuration summary."""
    print("=" * 70)
    print("CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"Run Folder: {RUN_FOLDER}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Hierarchy: {len(HIERARCHY)} paths, {TOTAL_NODES} total nodes")
    print(f"Classes per level: {NUM_CLASSES_PER_LEVEL}")
    print(f"Batch size: {DEFAULT_BATCH_SIZE}")
    print(f"Epochs: {DEFAULT_NUM_EPOCHS}")
    print(f"LR (classifier/backbone): {LR_CLASSIFIER}/{LR_BACKBONE}")
    print("=" * 70)