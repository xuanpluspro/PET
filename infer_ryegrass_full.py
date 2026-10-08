import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

import util.misc as utils
from models import build_model


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_args():
    parser = argparse.ArgumentParser("PET full-resolution or sliding-window inference for ryegrass")

    # Paths
    parser.add_argument(
        "--input_dir",
        default="/root/autodl-tmp/PET/test",
        type=str,
    )
    parser.add_argument(
        "--checkpoint",
        default="/root/autodl-tmp/PET/outputs/RiceLeaf/rice_quick/best_checkpoint.pth",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default="/root/autodl-tmp/PET/testresult",
        type=str,
    )

    # Full-resolution inference: no cropping, tiling, rescaling or patch-level NMS.
    parser.add_argument(
        "--full_image", action="store_true",
        help="Infer each original image in a single forward pass (no sliding windows).",
    )
    parser.add_argument(
        "--max_images", default=0, type=int,
        help="Process only the first N images; 0 means all images.",
    )

    # Sliding-window settings (ignored when --full_image is enabled).
    parser.add_argument("--patch_size", default=512, type=int)
    parser.add_argument("--overlap", default=128, type=int)
    parser.add_argument(
        "--merge_radius",
        default=12.0,
        type=float,
        help="Final global point deduplication radius in pixels.",
    )

    # Visualization
    parser.add_argument("--point_radius", default=7, type=int)

    # Inference
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA automatic mixed precision.",
    )

    # PET architecture: must match training
    parser.add_argument("--backbone", default="vgg16_bn", type=str)
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned", "fourier"),
    )
    parser.add_argument("--dec_layers", default=2, type=int)
    parser.add_argument("--dim_feedforward", default=512, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--nheads", default=8, type=int)

    # Required by build_model()
    parser.add_argument("--set_cost_class", default=1.0, type=float)
    parser.add_argument("--set_cost_point", default=0.05, type=float)
    parser.add_argument("--ce_loss_coef", default=1.0, type=float)
    parser.add_argument("--point_loss_coef", default=5.0, type=float)
    parser.add_argument("--eos_coef", default=0.5, type=float)

    # Compatibility args
    parser.add_argument("--dataset_file", default="RiceLeaf")
    parser.add_argument("--data_path", default="./data/RiceLeaf")
    parser.add_argument("--syn_bn", default=0, type=int)

    return parser.parse_args()


def build_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_model(args, device):
    model, _ = build_model(args)
    model.to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(
        f"Loaded checkpoint: {args.checkpoint} "
        f"(epoch={checkpoint.get('epoch', 'unknown')})"
    )
    return model


def make_starts(length, patch_size, stride):
    """
    Generate sliding-window start positions.
    The final patch is forced to touch the image boundary.
    """
    if length <= patch_size:
        return [0]

    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size

    if starts[-1] != last:
        starts.append(last)

    return starts


def ownership_bounds(starts, patch_size, full_length):
    """
    Split overlapping regions between neighboring patches.

    Each patch gets an exclusive ownership interval.
    This removes most duplicate detections before global point NMS.
    """
    bounds = []

    for i, s in enumerate(starts):
        if i == 0:
            left = 0.0
        else:
            prev = starts[i - 1]
            # Midpoint of the overlap between previous and current patch
            left = (s + prev + patch_size) / 2.0

        if i == len(starts) - 1:
            right = float(full_length)
        else:
            nxt = starts[i + 1]
            # Midpoint of the overlap between current and next patch
            right = (nxt + s + patch_size) / 2.0

        bounds.append((left, right))

    return bounds


@torch.inference_mode()
def infer_patch(model, patch_bgr, transform, device, use_amp=False):
    """
    Run PET on one patch.

    Returns a list of:
        {"x": local_x, "y": local_y, "score": prob}
    """
    patch_h, patch_w = patch_bgr.shape[:2]

    rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(Image.fromarray(rgb))

    samples = utils.nested_tensor_from_tensor_list([tensor]).to(device)
    padded_h, padded_w = samples.tensors.shape[-2:]

    if use_amp and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(samples, test=True)
    else:
        outputs = model(samples, test=True)

    pred_points = outputs["pred_points"][0].detach().float().cpu()
    pred_logits = outputs["pred_logits"][0].detach().float().cpu()
    pred_scores = torch.softmax(pred_logits, dim=-1)[:, 1]

    if len(pred_points) > 0:
        pred_points[:, 0] *= padded_h
        pred_points[:, 1] *= padded_w

    preds = []
    for (y, x), score in zip(pred_points.tolist(), pred_scores.tolist()):
        # Exclude possible predictions in padding.
        if 0 <= x < patch_w and 0 <= y < patch_h:
            preds.append({
                "x": float(x),
                "y": float(y),
                "score": float(score),
            })

    return preds


def radius_nms(points, radius):
    """
    Greedy score-based point NMS.

    Higher-confidence points are kept first.
    Any later point within `radius` pixels of an already kept point is removed.
    """
    if len(points) <= 1 or radius <= 0:
        return points

    points = sorted(points, key=lambda p: p["score"], reverse=True)
    kept = []
    r2 = radius * radius

    for p in points:
        duplicate = False

        for q in kept:
            dx = p["x"] - q["x"]
            dy = p["y"] - q["y"]

            if dx * dx + dy * dy <= r2:
                duplicate = True
                break

        if not duplicate:
            kept.append(p)

    # Restore a stable spatial order for saved JSON
    kept.sort(key=lambda p: (p["y"], p["x"]))
    return kept


@torch.inference_mode()
def infer_full_image(model, image_bgr, transform, device, args):
    """
    Sliding-window inference on the full-resolution image.

    Image itself is never resized.
    """
    h, w = image_bgr.shape[:2]
    patch_size = args.patch_size
    overlap = args.overlap

    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")
    if overlap < 0 or overlap >= patch_size:
        raise ValueError("overlap must satisfy 0 <= overlap < patch_size")

    stride = patch_size - overlap

    x_starts = make_starts(w, patch_size, stride)
    y_starts = make_starts(h, patch_size, stride)

    x_owner = ownership_bounds(x_starts, patch_size, w)
    y_owner = ownership_bounds(y_starts, patch_size, h)

    total_patches = len(x_starts) * len(y_starts)
    all_points = []

    patch_idx = 0

    for yi, y0 in enumerate(y_starts):
        for xi, x0 in enumerate(x_starts):
            patch_idx += 1

            x1 = min(x0 + patch_size, w)
            y1 = min(y0 + patch_size, h)

            patch = image_bgr[y0:y1, x0:x1]

            preds = infer_patch(
                model,
                patch,
                transform,
                device,
                use_amp=args.amp,
            )

            own_left, own_right = x_owner[xi]
            own_top, own_bottom = y_owner[yi]

            kept_this_patch = 0

            for p in preds:
                gx = p["x"] + x0
                gy = p["y"] + y0

                # Keep only predictions belonging to this patch's ownership region.
                # This divides overlap areas between neighboring patches.
                is_last_x = (xi == len(x_starts) - 1)
                is_last_y = (yi == len(y_starts) - 1)

                inside_x = (
                    gx >= own_left and
                    (gx < own_right or (is_last_x and gx <= own_right))
                )
                inside_y = (
                    gy >= own_top and
                    (gy < own_bottom or (is_last_y and gy <= own_bottom))
                )

                if inside_x and inside_y:
                    all_points.append({
                        "x": float(gx),
                        "y": float(gy),
                        "score": float(p["score"]),
                    })
                    kept_this_patch += 1

            print(
                f"\r  patches: {patch_idx}/{total_patches} "
                f"| raw={len(preds)} kept={kept_this_patch}",
                end="",
                flush=True,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print()

    before_nms = len(all_points)
    merged_points = radius_nms(all_points, args.merge_radius)

    return merged_points, total_patches, before_nms


def save_visualization(image, points, save_path, radius):
    vis = image.copy()

    for p in points:
        cv2.circle(
            vis,
            (int(round(p["x"])), int(round(p["y"]))),
            radius,
            (0, 255, 0),
            -1,
            lineType=cv2.LINE_AA,
        )

    text = f"Predicted tips: {len(points)}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(
        1.0,
        min(vis.shape[0], vis.shape[1]) / 1600.0,
    )
    thickness = max(2, int(round(font_scale * 2)))

    (tw, th), _ = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    cv2.rectangle(
        vis,
        (12, 12),
        (28 + tw, 36 + th),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        vis,
        text,
        (20, 20 + th),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    cv2.imwrite(
        str(save_path),
        vis,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )


def main():
    args = get_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    checkpoint_path = Path(args.checkpoint)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    if not image_paths:
        raise RuntimeError(f"No images found in: {input_dir}")
    if args.max_images < 0:
        raise ValueError("--max_images must be >= 0")
    if args.max_images:
        image_paths = image_paths[:args.max_images]

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    transform = build_transform()
    model = load_model(args, device)

    stride = args.patch_size - args.overlap

    print(f"Found {len(image_paths)} images.")
    if args.full_image:
        print("MODE: FULL ORIGINAL IMAGE; no cropping, resizing or sliding windows.")
        print("Point deduplication: DISABLED (raw PET detections, no extra radius NMS).")
        print("WARNING: 4096x3072 may exceed 32 GB GPU memory; use --amp.")
    else:
        print(
            f"Sliding-window: patch={args.patch_size}, "
            f"overlap={args.overlap}, stride={stride}"
        )
        print(f"Point deduplication radius: {args.merge_radius} px")
    print()

    rows = []

    for image_idx, image_path in enumerate(image_paths, 1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            print(f"[skip] Cannot read: {image_path}")
            continue

        h, w = image.shape[:2]

        print(
            f"[{image_idx}/{len(image_paths)}] "
            f"{image_path.name} ({w}x{h})"
        )

        try:
            if args.full_image:
                # Feed the ENTIRE original image to PET in one forward pass.
                # infer_patch() already preserves the input resolution and
                # returns global (x, y) coordinates for this full image.
                points = infer_patch(
                    model, image, transform, device, use_amp=args.amp
                )
                n_patches = 1
                before_nms = len(points)
            else:
                points, n_patches, before_nms = infer_full_image(
                    model, image, transform, device, args
                )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(
                    "\n[CUDA OOM] This image could not be inferred at full resolution. "
                    "No automatic cropping or resizing was applied. "
                    "Try --amp if not already enabled, or revert to sliding "
                    "windows without --full_image.",
                    flush=True,
                )
            raise

        vis_path = output_dir / (
            f"{image_path.stem}_pred{len(points)}.jpg"
        )

        save_visualization(
            image,
            points,
            vis_path,
            args.point_radius,
        )

        json_path = output_dir / (
            f"{image_path.stem}_points.json"
        )

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "image": image_path.name,
                    "width": w,
                    "height": h,
                    "mode": "full_image" if args.full_image else "sliding_window",
                    "patch_size": None if args.full_image else args.patch_size,
                    "overlap": None if args.full_image else args.overlap,
                    "stride": None if args.full_image else stride,
                    "merge_radius": None if args.full_image else args.merge_radius,
                    "num_patches": n_patches,
                    "pred_count_before_radius_nms": before_nms,
                    "pred_count": len(points),
                    "points": points,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        rows.append({
            "image": image_path.name,
            "mode": "full_image" if args.full_image else "sliding_window",
            "width": w,
            "height": h,
            "patches": n_patches,
            "before_nms": before_nms,
            "pred_count": len(points),
        })

        print(
            f"  result: before_nms={before_nms}, "
            f"final={len(points)}"
        )
        print(f"  saved: {vis_path}\n")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_csv = output_dir / "summary.csv"

    with summary_csv.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "mode",
                "width",
                "height",
                "patches",
                "before_nms",
                "pred_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Done.")
    print(f"Results: {output_dir}")
    print(f"Summary: {summary_csv}")


if __name__ == "__main__":
    main()
