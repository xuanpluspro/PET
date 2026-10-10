import os
import random
import warnings

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as standard_transforms

warnings.filterwarnings("ignore")


class Wheat1799(Dataset):
    def __init__(self, data_root, image_set="train",
                 transform=None, flip=False):

        self.root_path = data_root
        self.image_set = image_set
        self.transform = transform
        self.flip = flip
        self.train = image_set == "train"

        self.image_dir = os.path.join(data_root, "JPEGImages")
        self.anno_dir = os.path.join(data_root, "PointAnnotations")
        self.imageset_dir = os.path.join(data_root, "ImageSets")

        split_file = os.path.join(
            self.imageset_dir,
            f"{image_set}.txt"
        )

        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"Cannot find split file: {split_file}"
            )

        # train.txt / val.txt 中保存的是不带扩展名的图片 ID
        with open(split_file, "r", encoding="utf-8") as f:
            ids = [
                line.strip()
                for line in f
                if line.strip()
            ]

        image_exts = [
            ".png", ".jpg", ".jpeg",
            ".bmp", ".tif", ".tiff"
        ]

        self.samples = []

        for image_id in ids:

            img_path = None

            for ext in image_exts:
                candidate = os.path.join(
                    self.image_dir,
                    image_id + ext
                )

                if os.path.exists(candidate):
                    img_path = candidate
                    break

            if img_path is None:
                raise FileNotFoundError(
                    f"Cannot find image for ID: {image_id}"
                )

            gt_path = os.path.join(
                self.anno_dir,
                image_id + ".txt"
            )

            if not os.path.exists(gt_path):
                raise FileNotFoundError(
                    f"Cannot find annotation: {gt_path}"
                )

            self.samples.append(
                (img_path, gt_path)
            )

        self.patch_size = 512

        print(
            f"Wheat1799 {image_set}: "
            f"{len(self.samples)} images"
        )

    def __len__(self):
        return len(self.samples)

    def compute_density(self, points):

        if points.shape[0] <= 1:
            return torch.tensor(
                999.0
            ).reshape(-1)

        points_tensor = torch.from_numpy(
            points.copy()
        ).float()

        dist = torch.cdist(
            points_tensor,
            points_tensor,
            p=2
        )

        density = (
            dist.sort(dim=1)[0][:, 1]
            .mean()
            .reshape(-1)
        )

        return density

    def __getitem__(self, index):

        img_path, gt_path = self.samples[index]

        img, points = load_data(
            img_path,
            gt_path
        )

        if self.transform is not None:
            img = self.transform(img)

        img = torch.Tensor(img)

        # PET 官方训练增强
        if self.train:

            scale_range = [0.8, 1.2]

            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)

            if scale * min_size > self.patch_size:

                img = torch.nn.functional.interpolate(
                    img.unsqueeze(0),
                    scale_factor=scale,
                    mode="bilinear",
                    align_corners=False
                ).squeeze(0)

                points *= scale

            img, points = random_crop(
                img,
                points,
                self.patch_size
            )

        # 随机水平翻转
        if (
            self.train
            and self.flip
            and random.random() > 0.5
        ):

            img = torch.flip(
                img,
                dims=[2]
            )

            if len(points) > 0:

                # PET 内部坐标：(y, x)
                points[:, 1] = (
                    img.shape[2] - points[:, 1]
                )

        target = {}

        target["points"] = torch.tensor(
            points,
            dtype=torch.float32
        )

        # 单类别：leaf tip
        target["labels"] = torch.ones(
            len(points),
            dtype=torch.long
        )

        if self.train:
            target["density"] = self.compute_density(
                points
            )
        else:
            target["image_path"] = img_path

        return img, target


def load_data(img_path, gt_path):

    img = cv2.imread(img_path)

    if img is None:
        raise FileNotFoundError(
            f"Cannot read image: {img_path}"
        )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = Image.fromarray(img)

    points = []

    with open(
        gt_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            x = float(parts[0])
            y = float(parts[1])

            # 原始 txt: (x, y)
            # PET 内部: (y, x)
            points.append([y, x])

    points = np.asarray(
        points,
        dtype=np.float32
    ).reshape(-1, 2)

    return img, points


def random_crop(
    img,
    points,
    patch_size=256
):

    img_h = img.shape[1]
    img_w = img.shape[2]

    start_h = (
        random.randint(
            0,
            img_h - patch_size
        )
        if img_h > patch_size
        else 0
    )

    start_w = (
        random.randint(
            0,
            img_w - patch_size
        )
        if img_w > patch_size
        else 0
    )

    end_h = min(
        start_h + patch_size,
        img_h
    )

    end_w = min(
        start_w + patch_size,
        img_w
    )

    if len(points) > 0:

        idx = (
            (points[:, 0] >= start_h)
            & (points[:, 0] < end_h)
            & (points[:, 1] >= start_w)
            & (points[:, 1] < end_w)
        )

        result_points = points[idx].copy()

        result_points[:, 0] -= start_h
        result_points[:, 1] -= start_w

    else:

        result_points = np.empty(
            (0, 2),
            dtype=np.float32
        )

    result_img = img[
        :,
        start_h:end_h,
        start_w:end_w
    ]

    img_h2, img_w2 = result_img.shape[-2:]

    if (
        img_h2 != patch_size
        or img_w2 != patch_size
    ):

        scale_h = patch_size / img_h2
        scale_w = patch_size / img_w2

        result_img = torch.nn.functional.interpolate(
            result_img.unsqueeze(0),
            size=(patch_size, patch_size),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        if len(result_points) > 0:
            result_points[:, 0] *= scale_h
            result_points[:, 1] *= scale_w

    return result_img, result_points


def build(image_set, args):

    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),

        standard_transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        ),
    ])

    dataset = Wheat1799(
        args.data_path,
        image_set=image_set,
        transform=transform,
        flip=(image_set == "train")
    )

    return dataset