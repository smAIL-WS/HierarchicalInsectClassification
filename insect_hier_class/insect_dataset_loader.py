"""
Dataset classes for hierarchical insect classification.
"""

import cv2
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import config
import random

from transforms_utils import build_train_transform

class DynamicTransform:
    def __init__(self, schedule, aug_config, epoch_tracker):
        self.schedule = schedule
        self.aug = aug_config
        self.epoch_tracker = epoch_tracker
        self._cached_transform = None
        self._cached_image_size = None

    def set_epoch(self, epoch):
        self.epoch_tracker["current"] = epoch
        # Invalidate cache when epoch changes (in case image size changes)
        self._cached_transform = None
        self._cached_image_size = None

    def get_image_size(self):
        thresholds = self.schedule['epoch_thresholds']
        if self.epoch_tracker["current"] < thresholds[0]:
            return self.schedule['initial']
        elif self.epoch_tracker["current"] < thresholds[1]:
            return self.schedule['mid']
        else:
            return self.schedule['final']

    def _build_transform(self, image_size):
        """Build the transform pipeline once and cache it."""
        return build_train_transform(image_size=image_size, aug=self.aug)

    def __call__(self, img):
        image_size = self.get_image_size()

        # Only rebuild transform if image size changed
        if self._cached_transform is None or self._cached_image_size != image_size:
            self._cached_transform = self._build_transform(image_size)
            self._cached_image_size = image_size

        return self._cached_transform(img)


class InsectDataset(data.Dataset):
    """
    Dataset for insect images with hierarchical labels.
    Loads image paths and leaf-level labels from a text file.
    """

    def __init__(self, list_path, input_transform=None, preload_images=True, use_threads=False, num_workers=12):
        """
        Args:
            list_path: Path to text file with format: <image_path> <label> <class_name>
            input_transform: Optional transform to apply to images
            preload_images: If True, load all images into RAM at initialization
        """
        super(InsectDataset, self).__init__()

        self.image_filenames = []
        self.labels = []  # Leaf-level integer labels
        self.class_names = []  # Readable class names
        self.label_to_name = {}  # Mapping from integer labels to human-readable names
        self.transform = input_transform
        self.preloaded_images = None  # Will store PIL images if preloading

        with open(list_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                imagename = parts[0]
                leaf_label = int(parts[1])
                class_name = parts[2]

                self.image_filenames.append(imagename)
                self.labels.append(leaf_label)
                self.class_names.append(class_name)

                # Build label-to-name mapping
                if leaf_label not in self.label_to_name:
                    self.label_to_name[leaf_label] = class_name

        print(f"Loaded {len(self.image_filenames)} images with leaf-level labels.")

        def load_image_cv2(path):
            try:
                img = cv2.imread(path)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    print(f"Warning: Image {path} could not be loaded.")
                    return None
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                return None

        if preload_images:
            print(f"Preloading {len(self.image_filenames)} images into RAM using OpenCV...")
            if use_threads:
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    self.preloaded_images = list(executor.map(load_image_cv2, self.image_filenames))
            else:
                self.preloaded_images = []
                for i, imagename in enumerate(self.image_filenames):
                    self.preloaded_images.append(load_image_cv2(imagename))
                    if (i + 1) % 10000 == 0:
                        print(f"  Loaded {i + 1}/{len(self.image_filenames)} images...")
            print("✓ Preloading complete! All images in RAM.")

    def __getitem__(self, index):
        """
        Returns:
            Tuple of (image, target, index, class_name)
        """
        target = self.labels[index]
        class_name = self.class_names[index]

        # Get image from RAM if preloaded, otherwise load from disk
        if self.preloaded_images is not None:
            input_image = self.preloaded_images[index]
            if input_image is None:
                return None, None, None, None
            # Make a copy since transforms modify the image
            input_image = Image.fromarray(input_image.copy())
        else:
            imagename = self.image_filenames[index]
            try:
                input_image = Image.open(imagename).convert('RGB')
            except Exception as e:
                print(f"Error loading image {imagename}: {e}")
                return None, None, None, None

        if self.transform:
            input_image = self.transform(input_image)

        return input_image, target, index, class_name

    def __len__(self):
        return len(self.image_filenames)