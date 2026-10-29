import argparse
import csv
import random
import socket
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from checkpointing import (atomic_save, load_training_checkpoint,
                           save_training_checkpoint, select_resume_path,
                           trim_metric_history)


def seed_all():
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)


def build():
    network = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Dropout(.25),
                                  torch.nn.Linear(5, 1))
    optimizer = torch.optim.AdamW(network.parameters(), lr=.003)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=.5)
    return network, optimizer, scheduler


def train_epoch(network, optimizer, scheduler):
    network.train()
    for _ in range(3):
        x = torch.randn(4, 3) + random.random() + float(np.random.rand())
        loss = (network(x) - x.sum(dim=1, keepdim=True)).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()


class AugmentedDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return torch.randn(3) + random.random() + float(np.random.rand()) + index


def loader_epoch(network, optimizer, scheduler, loader):
    for x in loader:
        loss = (network(x) - x.sum(dim=1, keepdim=True)).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()


class CheckpointTest(unittest.TestCase):
    def test_multiworker_shuffle_and_augmentation_resume(self):
        try:
            probe = socket.socket(socket.AF_UNIX)
            probe.close()
        except PermissionError:
            self.skipTest('Environment blocks multiprocessing Unix sockets')
        args = argparse.Namespace(epochs=3, batch_size=4, num_workers=2)
        seed_all()
        expected, opt, sched = build()
        loader = torch.utils.data.DataLoader(AugmentedDataset(), batch_size=4,
                                            shuffle=True, num_workers=2)
        for _ in range(3):
            loader_epoch(expected, opt, sched, loader)
        seed_all()
        network, optimizer, scheduler = build()
        loader = torch.utils.data.DataLoader(AugmentedDataset(), batch_size=4,
                                            shuffle=True, num_workers=2)
        loader_epoch(network, optimizer, scheduler, loader)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'checkpoint.pth'
            save_training_checkpoint(path, network, optimizer, scheduler, 0, args, 3., 0)
            restored, ropt, rsched = build()
            rloader = torch.utils.data.DataLoader(AugmentedDataset(), batch_size=4,
                                                 shuffle=True, num_workers=2)
            start, _, _ = load_training_checkpoint(path, restored, ropt, rsched, args)
            for _ in range(start, args.epochs):
                loader_epoch(restored, ropt, rsched, rloader)
            for key, tensor in expected.state_dict().items():
                torch.testing.assert_close(tensor, restored.state_dict()[key], rtol=0, atol=0)

    def test_resumed_training_matches_uninterrupted(self):
        args = argparse.Namespace(epochs=4, batch_size=4)
        seed_all()
        expected, opt, sched = build()
        for _ in range(4):
            train_epoch(expected, opt, sched)
        expected_rng = (random.random(), np.random.rand(), torch.rand(3))

        seed_all()
        network, optimizer, scheduler = build()
        for _ in range(2):
            train_epoch(network, optimizer, scheduler)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'checkpoint.pth'
            best = Path(td) / 'best_checkpoint.pth'
            save_training_checkpoint(path, network, optimizer, scheduler, 1, args, 3.2, 1,
                                     best_path=best)
            restored, restored_opt, restored_sched = build()
            start, best_mae, best_epoch = load_training_checkpoint(
                path, restored, restored_opt, restored_sched, args)
            self.assertEqual((start, best_mae, best_epoch), (2, 3.2, 1))
            for _ in range(start, args.epochs):
                train_epoch(restored, restored_opt, restored_sched)
            for key, tensor in expected.state_dict().items():
                torch.testing.assert_close(tensor, restored.state_dict()[key], rtol=0, atol=0)
            self.assertEqual(restored_sched.state_dict(), sched.state_dict())
            self.assertEqual(expected_rng[0], random.random())
            self.assertEqual(expected_rng[1], np.random.rand())
            torch.testing.assert_close(expected_rng[2], torch.rand(3), rtol=0, atol=0)
            self.assertEqual(path.read_bytes(), best.read_bytes())

    def test_failed_atomic_write_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'checkpoint.pth'
            atomic_save({'old': 123}, path)
            old = path.read_bytes()

            def fail(state, stream):
                stream.write(b'incomplete checkpoint')
                raise OSError('simulated storage failure')

            with patch('checkpointing.torch.save', side_effect=fail):
                with self.assertRaises(OSError):
                    atomic_save({'new': 456}, path)
            self.assertEqual(path.read_bytes(), old)
            self.assertFalse(list(Path(td).glob('*.tmp')))

    def test_auto_resume_and_existing_run_guard(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(select_resume_path('', True, td), '')
            path = Path(td) / 'checkpoint.pth'
            path.touch()
            self.assertEqual(select_resume_path('', True, td), str(path))
            self.assertEqual(select_resume_path(td, False, td), str(path))
            with self.assertRaises(FileExistsError):
                select_resume_path('', False, td)

    def test_legacy_checkpoint_restores_optimizer_and_epoch(self):
        network, optimizer, scheduler = build()
        train_epoch(network, optimizer, scheduler)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'legacy.pth'
            torch.save({'model': network.state_dict(), 'optimizer': optimizer.state_dict(),
                        'lr_scheduler': scheduler.state_dict(), 'epoch': 0,
                        'args': argparse.Namespace(epochs=4)}, path)
            restored, opt, sched = build()
            with warnings.catch_warnings(record=True) as messages:
                start, _, _ = load_training_checkpoint(
                    path, restored, opt, sched, argparse.Namespace(epochs=4))
            self.assertEqual(start, 1)
            self.assertTrue(any('RNG' in str(m.message) for m in messages))
            self.assertEqual(sched.state_dict(), scheduler.state_dict())

    def test_incompatible_schedule_and_weights_only_are_rejected(self):
        network, optimizer, scheduler = build()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'checkpoint.pth'
            save_training_checkpoint(path, network, optimizer, scheduler, 0,
                                     argparse.Namespace(epochs=50), 2., 0)
            with self.assertRaisesRegex(ValueError, 'epochs'):
                load_training_checkpoint(path, network, optimizer, scheduler,
                                         argparse.Namespace(epochs=100))
            torch.save({'model': network.state_dict()}, path)
            with self.assertRaisesRegex(ValueError, 'cannot resume'):
                load_training_checkpoint(path, network, optimizer, scheduler,
                                         argparse.Namespace(epochs=50))

    def test_scaler_state_and_metric_history(self):
        network, optimizer, scheduler = build()
        scaler = unittest.mock.Mock()
        scaler.state_dict.return_value = {'scale': 8192.}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'checkpoint.pth'
            args = argparse.Namespace(epochs=2)
            save_training_checkpoint(path, network, optimizer, scheduler, 0, args, 2., 0,
                                     scaler=scaler)
            load_training_checkpoint(path, network, optimizer, scheduler, args, scaler=scaler)
            scaler.load_state_dict.assert_called_once_with({'scale': 8192.})
            history = Path(td) / 'metrics_history.csv'
            history.write_text('epoch,mae\n1,3\n2,2\n')
            trim_metric_history(history, 1)
            with history.open() as stream:
                self.assertEqual(list(csv.DictReader(stream)), [{'epoch': '1', 'mae': '3'}])
            history.write_text('epoch,mae,rmse,mse\n1,3,4,4\n')
            trim_metric_history(history, 1)
            self.assertEqual(history.read_text(), 'epoch,mae,rmse\n1,3,4\n')


if __name__ == '__main__':
    unittest.main()
