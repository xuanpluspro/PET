import torch.utils.data
import torchvision

from .SHA import build as build_sha
from .RiceLeaf import build as build_riceleaf
from .Wheat1799 import build as build_wheat1799


data_path = {
    "SHA": "./data/ShanghaiTech/part_A/",
    "RiceLeaf": "./data/RiceLeaf/",
    "Wheat1799": "./data/wheat1799/",
}


def build_dataset(image_set, args):

    if args.dataset_file not in data_path:
        raise ValueError(
            f"dataset {args.dataset_file} not supported"
        )

    args.data_path = data_path[args.dataset_file]

    if args.dataset_file == "SHA":
        return build_sha(image_set, args)

    elif args.dataset_file == "RiceLeaf":
        return build_riceleaf(image_set, args)

    elif args.dataset_file == "Wheat1799":
        return build_wheat1799(image_set, args)

    raise ValueError(
        f"dataset {args.dataset_file} not supported"
    )