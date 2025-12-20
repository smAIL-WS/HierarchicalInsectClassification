from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


class LetterboxPadToSquareReflect:
    """
    Resize preserving aspect ratio so the longer side equals target_size,
    then symmetrically pad to a square (target_size x target_size).

    Uses reflect padding when possible; falls back to edge padding when reflect
    constraints are exceeded (very small dimensions / extreme aspect ratios).
    """

    def __init__(self, target_size: int, interpolation: InterpolationMode = InterpolationMode.BILINEAR):
        if target_size < 1:
            raise ValueError("target_size must be >= 1")
        self.target_size = int(target_size)
        self.interpolation = interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise TypeError("LetterboxPadToSquareReflect expects a PIL.Image input.")

        w, h = img.size
        if w == 0 or h == 0:
            raise ValueError("Invalid image with zero width or height.")

        # Scale so that the longer side equals target_size
        scale = self.target_size / max(w, h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        # Resize with antialiasing if supported
        try:
            img = F.resize(img, (new_h, new_w), interpolation=self.interpolation, antialias=True)
        except TypeError:
            img = F.resize(img, (new_h, new_w), interpolation=self.interpolation)

        # Compute symmetric padding to reach a square canvas
        pad_left = (self.target_size - new_w) // 2
        pad_right = self.target_size - new_w - pad_left
        pad_top = (self.target_size - new_h) // 2
        pad_bottom = self.target_size - new_h - pad_top

        if any(p < 0 for p in (pad_left, pad_top, pad_right, pad_bottom)):
            raise ValueError("Negative padding computed; check target_size and input image size.")

        # Convert to tensor to use F.pad
        t = F.pil_to_tensor(img)  # uint8, (C, H, W)

        # Reflect padding constraints: per-side <= (dim - 1)
        limit_w = max(0, new_w - 1)
        limit_h = max(0, new_h - 1)

        left_reflect = min(pad_left, limit_w)
        right_reflect = min(pad_right, limit_w)
        top_reflect = min(pad_top, limit_h)
        bottom_reflect = min(pad_bottom, limit_h)

        left_edge = pad_left - left_reflect
        right_edge = pad_right - right_reflect
        top_edge = pad_top - top_reflect
        bottom_edge = pad_bottom - bottom_reflect

        # IMPORTANT: torchvision padding order is [left, top, right, bottom]
        if left_reflect or right_reflect or top_reflect or bottom_reflect:
            t = F.pad(
                t,
                padding=[left_reflect, top_reflect, right_reflect, bottom_reflect],
                padding_mode="reflect",
            )

        if left_edge or right_edge or top_edge or bottom_edge:
            t = F.pad(
                t,
                padding=[left_edge, top_edge, right_edge, bottom_edge],
                padding_mode="edge",
            )

        return F.to_pil_image(t)


def build_train_transform(image_size: int, aug: Dict[str, Any]) -> T.Compose:
    horiz_flip_p = 0.5 if aug.get("horizontal_flip", True) else 0.0
    vert_flip_p = 0.5 if aug.get("vertical_flip", False) else 0.0

    color_jitter = aug.get("color_jitter", {})
    rotation_deg = aug.get("rotation_degrees", 0)
    rotation_fill = aug.get("rotation_fill_value", 128)

    tx, ty = aug.get("affine_translate", (0.0, 0.0))
    affine_zoom_max = float(aug.get("affine_zoom_max", 1.2))
    affine_zoom_min = float(aug.get("affine_zoom_min", 0.9))
    affine_fill = aug.get("affine_fill_value", 128)

    gb_prob = float(aug.get("gaussian_blur_prob", 0.0))
    gb_kernel = int(aug.get("gaussian_kernel_size", 3))

    return T.Compose(
        [
            LetterboxPadToSquareReflect(target_size=image_size, interpolation=InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=horiz_flip_p),
            T.RandomVerticalFlip(p=vert_flip_p),
            T.ColorJitter(**color_jitter),
            T.RandomRotation(degrees=rotation_deg, expand=False, fill=rotation_fill),
            T.RandomAffine(
                degrees=0,
                translate=(tx, ty),
                scale=(affine_zoom_min, affine_zoom_max),
                fill=affine_fill,
            ),
            T.RandomApply([T.GaussianBlur(kernel_size=gb_kernel, sigma=(0.1, 1.0))], p=gb_prob),
            T.ToTensor(),
            T.Normalize(
                mean=aug.get("normalize_mean", (0.5, 0.5, 0.5)),
                std=aug.get("normalize_std", (0.5, 0.5, 0.5)),
            ),
        ]
    )


def build_test_transform(image_size: int, aug: Dict[str, Any]) -> T.Compose:
    # Test: no random aug, just letterbox + normalize
    return T.Compose(
        [
            LetterboxPadToSquareReflect(target_size=image_size, interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(
                mean=aug.get("normalize_mean", (0.5, 0.5, 0.5)),
                std=aug.get("normalize_std", (0.5, 0.5, 0.5)),
            ),
        ]
    )