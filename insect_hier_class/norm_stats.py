
#!/usr/bin/env python3
"""
Compute dataset-specific per-channel mean and std for normalization,
using the same text train list and your InsectDataset loader.
No CLI: configure constants at the top and run in PyCharm.
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

# Import your utilities
from transforms_utils import LetterboxPadToSquareReflect
from insect_dataset_loader import InsectDataset

# -----------------------------
# Configuration (edit to match your setup)
# -----------------------------
TRAIN_LIST = "/path/to/train/list" # path to your text train list: <image_path> <label> <class_name>
IMAGE_SIZE = 128                         # square target size for letterbox+reflect
BATCH_SIZE = 4096
NUM_WORKERS = 16
PRELOAD_IMAGES = False                   # set True if you want OpenCV preloading to RAM
USE_THREADS = False                      # set True to use threaded preloading

# Augmentation knobs to mirror your training (set to your actual values)
AUG = {
    "horizontal_flip": True,            # p=0.5
    "vertical_flip": True,             # p=0.5 if True (enable only if orientation-insensitive)
    "color_jitter": {
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1
    },                 # e.g., {"brightness":0.2,"contrast":0.2,"saturation":0.2,"hue":0.05}
    "rotation_degrees": 15.0,            # set >0 if you rotate
    "rotation_fill_value": 128,
    "affine_translate": (0.1, 0.1),     # fractions of width/height (e.g., 0.1, 0.1)
    "affine_zoom_max": 1.2,             # consider allowing downscale too in training (0.9, 1.2)
    "affine_fill_value": 128,
    "gaussian_blur_prob": 0.05,          # set >0 to include blur in stats (e.g., 0.1)
    "gaussian_kernel_size": 3,
}

# -----------------------------
# Build transform up to ToTensor(), excluding Normalize
# -----------------------------
def build_transform_without_normalization(image_size: int, aug: dict) -> T.Compose:
    horiz_flip_p = 0.5 if aug.get("horizontal_flip", True) else 0.0
    vert_flip_p = 0.5 if aug.get("vertical_flip", False) else 0.0

    color_jitter = aug.get("color_jitter", {})
    rotation_deg = aug.get("rotation_degrees", 0.0)
    rotation_fill = aug.get("rotation_fill_value", 128)

    tx, ty = aug.get("affine_translate", (0.0, 0.0))
    affine_zoom_max = float(aug.get("affine_zoom_max", 1.2))
    affine_fill = aug.get("affine_fill_value", 128)

    gb_prob = float(aug.get("gaussian_blur_prob", 0.0))
    gb_kernel = int(aug.get("gaussian_kernel_size", 3))

    ops = [
        LetterboxPadToSquareReflect(target_size=image_size, interpolation=InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(p=horiz_flip_p),
        T.RandomVerticalFlip(p=vert_flip_p),
    ]

    # Color jitter (optional)
    if color_jitter:
        ops.append(T.ColorJitter(**color_jitter))

    # Rotation (optional)
    if rotation_deg and rotation_deg != 0:
        ops.append(T.RandomRotation(degrees=rotation_deg, expand=False, fill=rotation_fill))

    # Affine: scale-only (zoom-in) plus optional translate
    ops.append(
        T.RandomAffine(
            degrees=0,
            translate=(tx, ty),
            scale=(1.0, affine_zoom_max),
            fill=affine_fill,
        )
    )

    # Gaussian blur (optional)
    if gb_prob and gb_prob > 0.0:
        ops.append(T.RandomApply([T.GaussianBlur(kernel_size=gb_kernel, sigma=(0.1, 1.0))], p=gb_prob))

    # ToTensor and no Normalize
    ops.append(T.ToTensor())

    return T.Compose(ops)

# -----------------------------
# DataLoader helpers
# -----------------------------
def safe_collate(batch):
    """
    Collate function that filters out None samples returned by the dataset
    and stacks only image tensors. Returns a tensor [B, C, H, W] or None if the batch is empty.
    """
    imgs = []
    for sample in batch:
        if sample is None:
            continue
        img = sample[0]  # (image, target, index, class_name)
        if img is None:
            continue
        imgs.append(img)
    if len(imgs) == 0:
        return None
    return torch.stack(imgs, dim=0)

@torch.no_grad()
def compute_channel_mean_std(dataloader: DataLoader) -> tuple:
    """
    Compute per-channel mean and std across the entire dataset in streaming fashion.
    Returns (mean, std) as two torch tensors of shape (C,).
    """
    sum_c = None
    sumsq_c = None
    total_pixels = 0

    for batch in dataloader:
        if batch is None:
            continue
        images = batch  # [B, C, H, W]
        b, c, h, w = images.shape
        pixels_in_batch = b * h * w

        batch_sum = images.sum(dim=(0, 2, 3))      # [C]
        batch_sumsq = (images ** 2).sum(dim=(0, 2, 3))  # [C]

        if sum_c is None:
            sum_c = batch_sum.clone()
            sumsq_c = batch_sumsq.clone()
        else:
            sum_c += batch_sum
            sumsq_c += batch_sumsq

        total_pixels += pixels_in_batch

    mean = sum_c / total_pixels
    ex2 = sumsq_c / total_pixels
    var = ex2 - mean ** 2
    var = torch.clamp(var, min=1e-12)
    std = torch.sqrt(var)

    return mean.cpu(), std.cpu()

# -----------------------------
# Main execution (no CLI)
# -----------------------------
def main():
    if not os.path.isfile(TRAIN_LIST):
        raise FileNotFoundError(f"TRAIN_LIST not found: {TRAIN_LIST}")

    transform = build_transform_without_normalization(IMAGE_SIZE, AUG)

    dataset = InsectDataset(
        list_path=TRAIN_LIST,
        input_transform=transform,
        preload_images=PRELOAD_IMAGES,
        use_threads=USE_THREADS,
        num_workers=NUM_WORKERS,
    )

    dl = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        persistent_workers=False,
        drop_last=False,
        collate_fn=safe_collate,
    )

    mean, std = compute_channel_mean_std(dl)

    m = tuple(round(float(x), 6) for x in mean)
    s = tuple(round(float(x), 6) for x in std)

    print("Dataset-specific normalization:")
    print(f"  mean = {m}")
    print(f"  std  = {s}")
    print("\nUse in transforms:")
    print(f"  T.Normalize(mean={m}, std={s})")

if __name__ == "__main__":
    main()