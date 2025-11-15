
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
RUN_DATE = '2025-11-14'  # CHANGE THIS for each new run
RUN_FOLDER = RUNS_DIR / RUN_DATE

# ============================================================================
# FILE PATHS (relative to RUN_FOLDER)
# ============================================================================
TRAIN_LIST_FILE = 'insect_train_list.txt'
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
    [1],
    [1, 5],
    [1, 8],
    [1, 5, 14],
    [1, 11, 22],
    [1, 9, 23],
    [1, 5, 19],
    [1, 10, 17, 34],
    [1, 5, 14, 31],
    [1, 7, 21, 35],
    [1, 7, 12, 32, 36],
    [1, 5, 14, 31, 37],
    [1, 7, 12, 24, 38],
    [1, 5, 14, 26, 41],
    [1, 5, 14, 31, 44],
    [1, 5, 14, 31, 45],
    [1, 5, 14, 31, 46],
    [1, 5, 14, 31, 47],
    [1, 3, 20, 28, 48],
    [1, 8, 16, 30, 50],
    [1, 5, 14, 26, 51],
    [1, 7, 12, 24, 54],
    [1, 3, 20, 28, 55],
    [1, 5, 14, 31, 52],
    [1, 3, 20, 28, 39],
    [1, 3, 20, 28, 49],
    [1, 6, 18, 33, 40],
    [1, 4, 15, 29, 42],
    [1, 5, 19, 27, 43],
    [0, 2, 13, 25, 53],
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
    'pf3': 10,
    'pf2': 12,
    'pf1': 12,
    'leaf': 20
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
        'feature_sizes': {'pf4':128, 'pf3':128, 'pf2':128, 'pf1':256, 'leaf':512}
    },
    'resnet18': {
        'num_ftrs': 512,
        'feature_sizes': {'pf4':128, 'pf3':128, 'pf2':256, 'pf1':256, 'leaf':512}
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
DROPOUT_RATE_FC = 0.5
FC_HIDDEN_DIM = 512

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
DEFAULT_BATCH_SIZE = 5120
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
TREE_LOSS_INVERT = True # If True, deeper nodes (towards leaf classes) get higher weights
TREE_LOSS_BETA = 0.99995  # For effective class weighting

# Entropy regularization settings
LAMBDA_ENTROPY_START = 1.0   # High initial weight for exploration
LAMBDA_ENTROPY_TARGET = 0.05 # Final weight after warm-up
ENTROPY_WARMUP_EPOCHS = 10   # Number of epochs to decay over

# Sentinel value for invalid targets
SENTINEL_VALUE = -1

# ============================================================================
# DATA AUGMENTATION
# ============================================================================
# Training augmentation parameters
TRAIN_AUGMENTATION = {
    'resize_crop_scale': (0.8, 1.0),
    'crop_offset': 8,  # image_size - 8
    'horizontal_flip': True,
    'color_jitter': {
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1
    },
    'rotation_degrees': 15,
    'affine_translate': (0.1, 0.1),
    'normalize_mean': (0.5, 0.5, 0.5),
    'normalize_std': (0.5, 0.5, 0.5)
}

# Test augmentation parameters
TEST_AUGMENTATION = {
    'crop_offset': 8,
    'normalize_mean': (0.5, 0.5, 0.5),
    'normalize_std': (0.5, 0.5, 0.5)
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