"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import numpy as np
import cv2

import torch
import torchvision.transforms as standard_transforms
import torch.nn.functional as F

import util.misc as utils
from util.misc import NestedTensor


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def visualization(samples, targets, pred, vis_dir, split_map=None):
    """
    Visualize predictions
    """
    gts = [t['points'].tolist() for t in targets]

    pil_to_tensor = standard_transforms.ToTensor()

    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])

    images = samples.tensors
    masks = samples.mask
    for idx in range(images.shape[0]):
        sample = restore_transform(images[idx])
        sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
        sample_vis = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()

        # draw ground-truth points (red)
        size = 2
        for t in gts[idx]:
            sample_vis = cv2.circle(sample_vis, (int(t[1]), int(t[0])), size, (0, 0, 255), -1)

        # draw predictions (green)
        for p in pred[idx]:
            sample_vis = cv2.circle(sample_vis, (int(p[1]), int(p[0])), size, (0, 255, 0), -1)
        
        # draw split map
        if split_map is not None:
            imgH, imgW = sample_vis.shape[:2]
            split_map = (split_map * 255).astype(np.uint8)
            split_map = cv2.applyColorMap(split_map, cv2.COLORMAP_JET)
            split_map = cv2.resize(split_map, (imgW, imgH), interpolation=cv2.INTER_NEAREST)
            sample_vis = split_map * 0.9 + sample_vis
        
        # save image
        if vis_dir is not None:
            # eliminate invalid area
            imgH, imgW = masks.shape[-2:]
            valid_area = torch.where(~masks[idx])
            valid_h, valid_w = valid_area[0][-1], valid_area[1][-1]
            sample_vis = sample_vis[:valid_h+1, :valid_w+1]

            name = targets[idx]['image_path'].split('/')[-1].split('.')[0]
            cv2.imwrite(os.path.join(vis_dir, '{}_gt{}_pred{}.jpg'.format(name, len(gts[idx]), len(pred[idx]))), sample_vis)


# training
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch {:02d}'.format(epoch + 1)
    print_freq = 50

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, compact=True):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        gt_points = [target['points'] for target in targets]

        outputs = model(samples, epoch=epoch, train=True, 
                                        criterion=criterion, targets=targets)
        loss_dict, weight_dict, losses = outputs['loss_dict'], outputs['weight_dict'], outputs['losses']

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# evaluation
@torch.no_grad()
def evaluate(model, data_loader, device, epoch=0, vis_dir=None,
             loc_radii=(4, 8, 16), metrics_dir=None):
    from counting_metrics import PointMetrics
    import torch.distributed as dist
    import time
    model.eval()
    meter = PointMetrics(loc_radii)
    metric_logger = utils.MetricLogger(delimiter="  ")
    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)
    start = time.perf_counter()
    for samples, targets in metric_logger.log_every(data_loader, 50, 'Test:', compact=True):
        if len(targets) != 1:
            raise ValueError('PET evaluation requires batch size 1')
        samples = samples.to(device)
        img_h, img_w = samples.tensors.shape[-2:]
        outputs = model(samples, test=True, targets=targets)
        points = outputs['pred_points'][0].detach().cpu().numpy().copy()
        points *= np.array([img_h, img_w])
        valid = (~samples.mask[0]).nonzero()
        valid_h = int(valid[:, 0].max().item()) + 1
        valid_w = int(valid[:, 1].max().item()) + 1
        keep = ((points[:, 0] >= 0) & (points[:, 0] < valid_h) &
                (points[:, 1] >= 0) & (points[:, 1] < valid_w))
        points = points[keep]
        gt = targets[0]['points'].detach().cpu().numpy()
        image_id = targets[0].get('image_path')
        if image_id is None:
            raise ValueError('Evaluation target must include a unique image_path')
        meter.update(image_id, points, gt)
        error = abs(len(points) - len(gt))
        metric_logger.update(mae=error)
        if vis_dir:
            split_map = (outputs['split_map_raw'][0].detach().cpu().squeeze(0) > 0.5).float().numpy()
            visualization(samples, targets, [points], vis_dir, split_map=split_map)
    meter.synchronize(expected_samples=len(data_loader.dataset))
    results = meter.summary()
    # Backward-compatible alias; its numerical meaning is RMSE.
    results['mse'] = results['rmse']
    results['evaluation_seconds'] = time.perf_counter() - start
    rank0 = not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
    if rank0:
        print('Validation | ' + ' | '.join(
            f'{key} {value:.4f}' for key, value in results.items()
            if key in ('mae', 'rmse') or key.startswith(('precision_', 'recall_', 'f1_'))))
        if metrics_dir:
            meter.export(metrics_dir, results, {'epoch_index': epoch, 'split': 'val',
                'prediction_threshold': 0.5, 'threshold_rule': 'foreground probability > 0.5',
                'padding_predictions': 'removed', 'params': sum(p.numel() for p in model.parameters())})
    return results
