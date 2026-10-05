import os
import json
import random
import warnings

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as standard_transforms

warnings.filterwarnings('ignore')


class RiceLeaf(Dataset):
    def __init__(self, data_root, transform=None, train=False, flip=False):
        self.root_path = data_root

        # Your dataset:
        # RiceLeaf/
        # ├── train/
        # │   ├── images/
        # │   └── annotations/
        # └── test/
        #     ├── images/
        #     └── annotations/
        prefix = "train" if train else "test"
        self.prefix = prefix

        image_dir = os.path.join(data_root, prefix, "images")
        anno_dir = os.path.join(data_root, prefix, "annotations")

        # Find images
        image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

        img_names = [
            name for name in os.listdir(image_dir)
            if name.lower().endswith(image_exts)
        ]

        # Match image and LabelMe JSON by filename stem
        self.gt_list = {}

        for img_name in img_names:
            img_path = os.path.join(image_dir, img_name)

            stem = os.path.splitext(img_name)[0]
            gt_path = os.path.join(anno_dir, stem + ".json")

            if not os.path.exists(gt_path):
                raise FileNotFoundError(
                    f"Cannot find annotation for image:\n"
                    f"image: {img_path}\n"
                    f"json : {gt_path}"
                )

            self.gt_list[img_path] = gt_path

        self.img_list = sorted(self.gt_list.keys())
        self.nSamples = len(self.img_list)

        self.transform = transform
        self.train = train
        self.flip = flip

        # Keep the same training patch size as official PET
        self.patch_size = 256

        print(
            f"RiceLeaf {'train' if train else 'test'} dataset: "
            f"{self.nSamples} images"
        )

    def compute_density(self, points):
        """
        PET density definition:
        average nearest-neighbor distance between GT points.

        Smaller value -> denser points
        Larger value  -> sparser points
        """
        if points.shape[0] <= 1:
            return torch.tensor(999.0).reshape(-1)

        points_tensor = torch.from_numpy(
            points.copy()
        ).float()

        # pairwise Euclidean distance
        dist = torch.cdist(
            points_tensor,
            points_tensor,
            p=2
        )

        # first smallest value is self-distance = 0
        # second smallest is nearest-neighbor distance
        density = (
            dist.sort(dim=1)[0][:, 1]
            .mean()
            .reshape(-1)
        )

        return density

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        if index >= len(self):
            raise IndexError("index range error")

        # --------------------------------------------------
        # 1. Load image and LabelMe point annotations
        # --------------------------------------------------
        img_path = self.img_list[index]
        gt_path = self.gt_list[img_path]

        img, points = load_data(
            (img_path, gt_path),
            self.train
        )

        points = points.astype(np.float32)

        # --------------------------------------------------
        # 2. ImageNet normalization
        # --------------------------------------------------
        if self.transform is not None:
            img = self.transform(img)

        img = torch.Tensor(img)

        # --------------------------------------------------
        # 3. Random scale (official PET augmentation)
        # --------------------------------------------------
        if self.train:

            scale_range = [0.8, 1.2]

            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)

            # Ensure scaled image is still larger than patch_size
            if scale * min_size > self.patch_size:

                img = torch.nn.functional.upsample_bilinear(
                    img.unsqueeze(0),
                    scale_factor=scale
                ).squeeze(0)

                points *= scale

        # --------------------------------------------------
        # 4. Random crop to 256 x 256
        # --------------------------------------------------
        if self.train:
            img, points = random_crop(
                img,
                points,
                patch_size=self.patch_size
            )

        # --------------------------------------------------
        # 5. Random horizontal flip
        # --------------------------------------------------
        if (
            random.random() > 0.5
            and self.train
            and self.flip
        ):
            img = torch.flip(img, dims=[2])

            # PET internally stores points as (y, x)
            if len(points) > 0:
                points[:, 1] = (
                    self.patch_size - points[:, 1]
                )

        # --------------------------------------------------
        # 6. Build PET target
        # --------------------------------------------------
        target = {}

        # PET coordinate convention: (y, x)
        target["points"] = torch.tensor(
            points,
            dtype=torch.float32
        )

        # Single class: tip
        # 1 = foreground / leaf tip
        target["labels"] = torch.ones(
            points.shape[0],
            dtype=torch.long
        )

        # Training uses density for sparse/dense supervision
        if self.train:
            target["density"] = self.compute_density(
                points
            )

        # Evaluation visualization needs image path
        if not self.train:
            target["image_path"] = img_path

        return img, target


def load_data(img_gt_path, train):
    """
    Read:
        image
        LabelMe JSON

    Only keep:
        label == "tip"

    LabelMe coordinates:
        (x, y)

    PET coordinates:
        (y, x)
    """

    img_path, gt_path = img_gt_path

    # --------------------------------------------------
    # Read image
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Read LabelMe JSON
    # --------------------------------------------------
    with open(
        gt_path,
        "r",
        encoding="utf-8"
    ) as f:
        anno = json.load(f)

    points = []

    for shape in anno.get("shapes", []):

        # Only leaf-tip annotations
        if shape.get("label") != "tip":
            continue

        shape_points = shape.get("points", [])

        if len(shape_points) == 0:
            continue

        # LabelMe point:
        # [[x, y]]
        x, y = shape_points[0]

        # PET requires:
        # (y, x)
        points.append([y, x])

    # Important:
    # Even if there are 0 points,
    # keep shape as (0, 2)
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
    """
    Randomly crop a patch and synchronously
    transform the GT point coordinates.
    """

    patch_h = patch_size
    patch_w = patch_size

    img_h = img.size(1)
    img_w = img.size(2)

    # --------------------------------------------------
    # Random crop position
    # --------------------------------------------------
    start_h = (
        random.randint(
            0,
            img_h - patch_h
        )
        if img_h > patch_h
        else 0
    )

    start_w = (
        random.randint(
            0,
            img_w - patch_w
        )
        if img_w > patch_w
        else 0
    )

    end_h = start_h + patch_h
    end_w = start_w + patch_w

    # --------------------------------------------------
    # Select points inside crop
    # PET points format: (y, x)
    # --------------------------------------------------
    if len(points) > 0:

        idx = (
            (points[:, 0] >= start_h)
            & (points[:, 0] <= end_h)
            & (points[:, 1] >= start_w)
            & (points[:, 1] <= end_w)
        )

        result_points = points[idx].copy()

        result_points[:, 0] -= start_h
        result_points[:, 1] -= start_w

    else:
        result_points = np.empty(
            (0, 2),
            dtype=np.float32
        )

    # --------------------------------------------------
    # Crop image
    # --------------------------------------------------
    result_img = img[
        :,
        start_h:end_h,
        start_w:end_w
    ]

    # --------------------------------------------------
    # Resize crop to 256 x 256 if necessary
    # --------------------------------------------------
    imgH, imgW = result_img.shape[-2:]

    fH = patch_h / imgH
    fW = patch_w / imgW

    result_img = torch.nn.functional.interpolate(
        result_img.unsqueeze(0),
        (patch_h, patch_w)
    ).squeeze(0)

    if len(result_points) > 0:
        result_points[:, 0] *= fH
        result_points[:, 1] *= fW

    return result_img, result_points


def build(image_set, args):

    # Same ImageNet normalization as PET VGG16-BN
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

    data_root = args.data_path

    if image_set == "train":

        train_set = RiceLeaf(
            data_root,
            train=True,
            transform=transform,
            flip=True
        )

        return train_set

    elif image_set == "val":

        # PET uses test set as validation set
        val_set = RiceLeaf(
            data_root,
            train=False,
            transform=transform
        )

        return val_set

    else:
        raise ValueError(
            f"Unknown image_set: {image_set}"
        )