"""Atomic, epoch-boundary checkpoints for both PET training entrypoints."""
import csv
import os
import random
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def distributed():
    return dist.is_available() and dist.is_initialized()


def atomic_save(state, destination):
    """A failed write leaves the previous destination intact."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + '.', suffix='.tmp',
                                     dir=str(destination.parent))
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        # Best effort directory flush; filesystem durability still depends on
        # the storage provider. Atomic replacement protects against partial files.
        try:
            directory_fd = os.open(str(destination.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def capture_rng():
    return {
        'python': random.getstate(), 'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state['cuda'].cpu())


def save_training_checkpoint(destination, model, optimizer, scheduler, epoch,
                             args, best_mae, best_epoch, scaler=None, best_path=None):
    """All DDP ranks must call this function; rank zero writes shared files."""
    local_rng = capture_rng()
    rng_states = [local_rng]
    if distributed():
        rng_states = [None] * dist.get_world_size()
        dist.all_gather_object(rng_states, local_rng)
    error = None
    if not distributed() or dist.get_rank() == 0:
        try:
            state = {
                'checkpoint_version': 2, 'completed_epoch': True,
                'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(), 'epoch': int(epoch),
                'args': dict(vars(args)) if hasattr(args, '__dict__') else dict(args),
                'best_mae': best_mae, 'best_epoch': best_epoch,
                'rng_states': rng_states, 'world_size': len(rng_states),
            }
            if scaler is not None:
                state['scaler'] = scaler.state_dict()
            # Best first: interruption cannot leave latest claiming a new best
            # whose weights have not been written yet.
            if best_path is not None:
                atomic_save(state, best_path)
            atomic_save(state, destination)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    if distributed():
        result = [error]
        dist.broadcast_object_list(result, src=0)
        error = result[0]
    if error is not None:
        raise RuntimeError('Checkpoint save failed: ' + error)


def select_resume_path(resume, auto_resume, output_dir):
    if resume:
        if str(resume).startswith('https://'):
            return str(resume)
        path = Path(resume).expanduser()
        if path.is_dir():
            path = path / 'checkpoint.pth'
        if not path.is_file():
            raise FileNotFoundError(f'Resume checkpoint not found: {path}')
        return str(path)
    latest = Path(output_dir) / 'checkpoint.pth'
    if latest.is_file():
        if auto_resume:
            return str(latest)
        raise FileExistsError(f'{latest} exists. Use --auto-resume, --resume, or a new --output_dir.')
    return ''


def load_training_checkpoint(path, model, optimizer, scheduler, args, scaler=None):
    # Training state includes Python/NumPy RNG objects: only load trusted files.
    if str(path).startswith('https://'):
        try:
            state = torch.hub.load_state_dict_from_url(path, map_location='cpu',
                                                       check_hash=True, weights_only=False)
        except TypeError:
            state = torch.hub.load_state_dict_from_url(path, map_location='cpu', check_hash=True)
    else:
        try:
            state = torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:
            state = torch.load(path, map_location='cpu')
    required = ('model', 'optimizer', 'lr_scheduler', 'epoch')
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f'Checkpoint lacks {missing}; it can initialize weights but cannot resume training.')
    saved = state.get('args', {})
    saved = vars(saved) if hasattr(saved, '__dict__') else saved
    current = vars(args) if hasattr(args, '__dict__') else args
    # Changing epochs would also silently change the intended LR schedule.
    fixed = ('epochs', 'dataset_file', 'backbone', 'position_embedding', 'dec_layers',
             'hidden_dim', 'dim_feedforward', 'nheads', 'dropout', 'expected_size',
             'amp', 'ce_loss_coef', 'point_loss_coef', 'eos_coef', 'set_cost_class',
             'set_cost_point', 'clip_max_norm', 'weight_decay', 'lr', 'lr_backbone')
    for key in fixed:
        if key in saved and key in current and saved[key] != current[key]:
            raise ValueError(f'Resume setting {key} differs: saved={saved[key]!r}, current={current[key]!r}. '
                             'Keep the original training configuration; epochs is the total planned count.')
    for key in ('batch_size', 'num_workers', 'accum_steps', 'data_path'):
        if key in saved and key in current and saved[key] != current[key]:
            warnings.warn(f'Resume changed {key}: {saved[key]!r} -> {current[key]!r}; '
                          'continuation is supported but the training trajectory may differ.')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['lr_scheduler'])
    if scaler is not None:
        if state.get('scaler'):
            scaler.load_state_dict(state['scaler'])
        elif scaler.is_enabled():
            warnings.warn('Old checkpoint has no AMP scaler state; starting a fresh scaler.')
    states = state.get('rng_states')
    rank = dist.get_rank() if distributed() else 0
    world_size = dist.get_world_size() if distributed() else 1
    if states and len(states) == world_size:
        restore_rng(states[rank])
    else:
        warnings.warn('Old checkpoint lacks RNG state or GPU process count changed; '
                      'weights/optimizer/epoch are restored, exact trajectory is not guaranteed.')
    return int(state['epoch']) + 1, state.get('best_mae', float('inf')), state.get('best_epoch', -1)


def trim_metric_history(path, completed_epochs):
    """Remove metrics written after the last durable checkpoint on a restart."""
    path = Path(path)
    if not path.is_file():
        return
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or 'epoch' not in fields:
        raise ValueError(f'Missing epoch column in {path}')
    kept = [row for row in rows if int(row['epoch']) <= completed_epochs]
    migrated = 'mse' in fields
    if migrated:
        # Historical PET exports used mse as an alias for RMSE, not raw MSE.
        if 'rmse' in fields:
            fields = [field for field in fields if field != 'mse']
            for row in kept:
                row.pop('mse', None)
        else:
            fields = ['rmse' if field == 'mse' else field for field in fields]
            for row in kept:
                row['rmse'] = row.pop('mse')
    if len(kept) == len(rows) and not migrated:
        return
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(kept)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
