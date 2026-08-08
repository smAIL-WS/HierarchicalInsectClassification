"""
Dataset classes for hierarchical insect classification.
"""

import cv2
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import config
import random

from transforms_utils import build_train_transform


def load_image_cv2(path):
    """Decode an image from disk with OpenCV, converted to RGB."""
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


def safe_collate(batch):
    """
    Collate function for InsectDataset batches of (image, target, index, class_name).
    Filters out samples where the dataset returned None (failed image load) before
    handing the remainder to the default collate.
    """
    batch = [sample for sample in batch if sample[0] is not None]
    if len(batch) == 0:
        return None
    return data.default_collate(batch)


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
    Loads image paths and leaf-level labels from one or more text files.
    """

    def __init__(self, list_path, input_transform=None, preload_images=False, use_threads=False, num_workers=12):
        """
        Args:
            list_path: Either a string(single list file) or a list of strings
            specifying multiple list files to combine with format: <image_path> <label> <class_name>.
            input_transform: Optional transform to apply to images
            preload_images: If True, load all images into RAM at initialization
        """
        super(InsectDataset, self).__init__()

        # Normalize input: always operate on a list of paths
        if isinstance(list_path, (str, Path)):
            list_paths = [list_path]
        else:
            list_paths = list_path

        self.image_filenames = []
        self.labels = []  # Leaf-level integer labels
        self.class_names = []  # Readable class names
        self.label_to_name = {}  # Mapping from integer labels to human-readable names
        self.transform = input_transform
        self.preloaded_images = None  # Will store PIL images if preloading

        # Load every list file in order
        for path in list_paths:
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().split(' ')

                    path_str = parts[0]
                    path_obj = Path(path_str)

                    if path_obj.is_absolute():
                        imagename = str(path_obj)
                    else:
                        imagename = str(config.DATASET_ROOT / path_obj)

                    leaf_label = int(parts[1])
                    class_name = parts[2]

                    self.image_filenames.append(imagename)
                    self.labels.append(leaf_label)
                    self.class_names.append(class_name)

                    # Build label-to-name mapping
                    if leaf_label not in self.label_to_name:
                        self.label_to_name[leaf_label] = class_name

        print(f"Loaded {len(self.image_filenames)} images with leaf-level labels.")

        if preload_images:
            print(f"Preloading {len(self.image_filenames)} images into RAM using OpenCV...")
            cv2.setNumThreads(0)

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
            raw_image = load_image_cv2(imagename)
            if raw_image is None:
                return None, None, None, None
            input_image = Image.fromarray(raw_image)

        if self.transform:
            input_image = self.transform(input_image)

        return input_image, target, index, class_name

    def __len__(self):
        return len(self.image_filenames)

    # ---------------------------
    # Cleanup hook for HPO teardown
    # ---------------------------
    def close(self):
        """
        Release large references so GC can reclaim memory promptly between trials.
        This does NOT affect intra-trial performance; call it only at trial end.
        """
        self.preloaded_images = None
        self.image_filenames = None
        self.labels = None
        self.class_names = None
        self.label_to_name = None
        self.transform = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
