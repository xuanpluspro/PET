"""PET Wheat1799 full-image training. Separate entrypoint; original main.py untouched."""
import argparse
import datetime
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import util.misc as utils
from datasets.Wheat1799Full import Wheat1799Full
from engine import evaluate
from main import get_args_parser
from models import build_model


def main(args):
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.batch_size == 1:
        print("WARNING: PET's density-ranked sparse/dense group losses are designed "
              "for multiple images per batch; prefer batch_size >= 2 if memory permits.")
    if args.accum_steps < 1 or args.epochs < 1:
        raise ValueError("accum_steps and epochs must be positive")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This full-resolution experiment requires a CUDA GPU")
    utils.init_distributed_mode(args)
    if args.distributed:
        raise NotImplementedError("Use single GPU only for this experiment")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")
    model, criterion = build_model(args)
    model.to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n,p in model.named_parameters() if "backbone" not in n]},
            {"params": [p for n,p in model.named_parameters() if "backbone" in n],
             "lr": args.lr_backbone},
        ],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epochs)
    train_data = Wheat1799Full(args.data_path, "train", args.expected_size)
    val_data = Wheat1799Full(args.data_path, "val", args.expected_size)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=utils.collate_fn,
        # PET's density-ranked sparse/dense supervision works best with >=2
        # images per microbatch, so avoid an undersized final training batch.
        drop_last=args.batch_size > 1,
    )
    val_loader = DataLoader(
        val_data, batch_size=1, shuffle=False, num_workers=args.num_workers,
        collate_fn=utils.collate_fn,
    )
    output_dir = Path("outputs") / "Wheat1799Full" / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results: {output_dir.resolve()}")
    print(f"Train images={len(train_data)} val images={len(val_data)}")
    print(
        f"Per-GPU microbatch={args.batch_size} images, "
        f"accumulation={args.accum_steps}, "
        f"effective batch={args.batch_size * args.accum_steps}, AMP={args.amp}"
    )

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    best_mae = float("inf")
    best_epoch = -1
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["lr_scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_mae = ckpt.get("best_mae", best_mae)
        best_epoch = ckpt.get("best_epoch", best_epoch)
        print(f"RESUMED at epoch {start_epoch} from {args.resume}")

    start_time = time.time()
    train_count = len(train_loader)
    if args.dry_run:
        train_count = min(train_count, args.accum_steps)
        print(
            f"DRY RUN: {train_count} minibatches "
            f"(up to {train_count * args.batch_size} full images), "
            "one optimizer step, no checkpoints."
        )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        criterion.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_sum = 0.0
        if args.amp:
            torch.cuda.reset_peak_memory_stats()
        epoch_start = time.time()

        for i, (samples, targets) in enumerate(train_loader):
            if args.dry_run and i >= train_count:
                break
            samples = samples.to(device)
            targets = [
                {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k,v in target.items()}
                for target in targets
            ]
            # Last accumulation group may have fewer than accum_steps items.
            group_start = (i // args.accum_steps) * args.accum_steps
            group_size = min(args.accum_steps, train_count - group_start)
            try:
                with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                    out = model(
                        samples, epoch=epoch, train=True,
                        criterion=criterion, targets=targets,
                    )
                    total_loss = out["losses"]
                value = total_loss.detach().float().item()
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite training loss: {value}")
                scaler.scale(total_loss / group_size).backward()
            except torch.cuda.OutOfMemoryError:
                print(
                    f"\nCUDA OOM with batch_size={args.batch_size} at 1024x1024. "
                    "Try a smaller --batch_size (e.g. 2 instead of 4), "
                    "retain --amp, or optimize PET memory. "
                    "Increasing --accum_steps does NOT reduce peak memory.",
                    flush=True,
                )
                raise

            epoch_sum += value
            if (i + 1) % args.accum_steps == 0 or (i + 1) == train_count:
                scaler.unscale_(optimizer)
                if args.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if (i + 1) % args.log_interval == 0 or i + 1 == train_count:
                mem_gb = torch.cuda.max_memory_allocated() / 1024**3
                print(
                    f"epoch {epoch} [{i+1}/{train_count}] "
                    f"loss={value:.5f} avg={epoch_sum/(i+1):.5f} "
                    f"peak_mem={mem_gb:.2f}GiB", flush=True,
                )
            del samples, targets, out, total_loss

        if args.dry_run:
            print("DRY RUN PASSED (forward, backward, optimizer step).")
            return

        scheduler.step()
        print(f"[epoch {epoch}] avg_loss={epoch_sum/train_count:.6f} "
              f"time={time.time()-epoch_start:.1f}s", flush=True)

        # Validate BEFORE checkpoint save so 'best' metadata is in sync.
        is_best = False
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            stats = evaluate(model, val_loader, device, epoch, None)
            mae, rmse = stats["mae"], stats["mse"]
            is_best = mae < best_mae
            if is_best:
                best_mae, best_epoch = mae, epoch
            print(f"[validation] epoch={epoch} MAE={mae:.4f} RMSE={rmse:.4f} "
                  f"best_MAE={best_mae:.4f} best_epoch={best_epoch}", flush=True)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "args": args,
            "best_mae": best_mae,
            "best_epoch": best_epoch,
        }
        torch.save(state, output_dir / "checkpoint.pth")
        if is_best:
            shutil.copy2(output_dir / "checkpoint.pth", output_dir / "best_checkpoint.pth")
        with (output_dir / "run_log.txt").open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch, "train_loss": epoch_sum/train_count,
                "best_mae": best_mae, "best_epoch": best_epoch,
                "time_s": time.time()-epoch_start,
            }) + "\n")
    print(f"Training time {datetime.timedelta(seconds=int(time.time()-start_time))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Wheat1799 PET full image", parents=[get_args_parser()]
    )
    parser.set_defaults(
        dataset_file="Wheat1799Full",
        data_path="./data/wheat1799",
        batch_size=2,
        epochs=50,
        eval_freq=5,
        output_dir="wheat1799_full1024",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--accum_steps", type=int, default=8)
    parser.add_argument("--expected_size", type=int, default=1024)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--log_interval", type=int, default=100)
    main(parser.parse_args())
