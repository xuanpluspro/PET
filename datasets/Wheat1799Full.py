"""Wheat1799 FULL-IMAGE dataset for PET.

Training and validation both use the original 1024x1024 image.
NO random crop, NO random scale, NO resizing.
Horizontal flip is the only spatial training augmentation.
Annotations in TXT are (x, y), PET requires (y, x).
"""
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Wheat1799Full(Dataset):
    def __init__(self, data_root, image_set="train", expected_size=1024):
        self.root = Path(data_root)
        self.train = image_set == "train"
        self.image_set = image_set
        self.expected_size = expected_size

        split_path = self.root / "ImageSets" / f"{image_set}.txt"
        if not split_path.is_file():
            raise FileNotFoundError(f"Missing split file: {split_path}")
        image_ids = [
            line.strip() for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not image_ids:
            raise RuntimeError(f"Empty split file: {split_path}")

        extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        self.samples = []
        for image_id in image_ids:
            path = next(
                (self.root / "JPEGImages" / (image_id + ext)
                 for ext in extensions
                 if (self.root / "JPEGImages" / (image_id + ext)).is_file()),
                None,
            )
            if path is None:
                raise FileNotFoundError(f"Image not found for ID: {image_id}")
            annotation = self.root / "PointAnnotations" / (image_id + ".txt")
            if not annotation.is_file():
                raise FileNotFoundError(f"Annotation not found: {annotation}")
            self.samples.append((path, annotation))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])
        print(
            f"Wheat1799Full {image_set}: {len(self.samples)} FULL images "
            f"(expected {expected_size}x{expected_size}; crop=OFF; scale=OFF)"
        )

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def compute_density(points):
        if len(points) <= 1:
            return torch.tensor([999.0], dtype=torch.float32)
        xy = torch.from_numpy(points.copy()).float()
        distances = torch.cdist(xy, xy)
        distances.fill_diagonal_(float("inf"))
        return distances.min(dim=1).values.mean().reshape(1)

    def __getitem__(self, index):
        image_path, annotation_path = self.samples[index]
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f"Cannot read image: {image_path}")
        height, width = bgr.shape[:2]
        if self.expected_size and (height, width) != (
            self.expected_size, self.expected_size
        ):
            raise ValueError(
                f"Image {image_path.name} has size {width}x{height}; "
                f"expected {self.expected_size}x{self.expected_size}. "
                "No crop/resize is performed in full-image training."
            )

        points = []
        with annotation_path.open("r", encoding="utf-8") as file:
            for line_no, line in enumerate(file, 1):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) < 2:
                    raise ValueError(f"Bad point {annotation_path}:{line_no}")
                x, y = float(fields[0]), float(fields[1])
                if not (0 <= x < width and 0 <= y < height):
                    raise ValueError(
                        f"Point outside image {annotation_path}:{line_no}: x={x}, y={y}"
                    )
                points.append((y, x))

        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        image = self.transform(
            Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        )

        # Preserve every pixel and every labelled leaf tip.
        # Horizontal flip does not introduce any new crop boundaries.
        if self.train and random.random() > 0.5:
            image = torch.flip(image, dims=[2])
            points[:, 1] = (width - 1) - points[:, 1]

        target = {
            "points": torch.from_numpy(points.copy()).float(),
            "labels": torch.ones(len(points), dtype=torch.long),
        }
        if self.train:
            target["density"] = self.compute_density(points)
        else:
            target["image_path"] = str(image_path)
        return image, target
