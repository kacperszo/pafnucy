import collections
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from rotation_schedule import ROTATIONS, RotationBatchSampler, transform
from train import ComplexGrids, HDFGrids, epoch


class TinyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 5
    def __getitem__(self, index):
        item, rotation = index
        return torch.tensor([float(item), float(rotation)]), torch.tensor([1.])


class CountingSGD(torch.optim.SGD):
    def __init__(self, params):
        super().__init__(params, lr=.0001)
        self.calls = 0
    def step(self, closure=None):
        self.calls += 1
        return super().step(closure)


class ScheduleTests(unittest.TestCase):
    def test_cube_group_and_identity(self):
        np.testing.assert_array_equal(ROTATIONS[0], np.eye(3))
        self.assertEqual(len({tuple(m.ravel()) for m in ROTATIONS}), 24)
        for m in ROTATIONS:
            np.testing.assert_array_equal(m.T @ m, np.eye(3))
            self.assertAlmostEqual(np.linalg.det(m), 1.)
            for n in ROTATIONS:
                self.assertTrue(any(np.array_equal(m @ n, k) for k in ROTATIONS))
        coords = np.array([[1., 2., 3.], [.5, -.5, 1.5]])
        for i in range(24):
            rotated = transform(coords, i)
            np.testing.assert_array_equal(np.sort(abs(rotated), axis=1), np.sort(abs(coords), axis=1))

    def test_rotation_major_coverage_and_partial_batches(self):
        batches = list(RotationBatchSampler(5, 2, seed=9))
        self.assertEqual(len(batches), 24 * 3)
        for rotation in range(24):
            block = batches[3*rotation:3*(rotation+1)]
            self.assertEqual([len(b) for b in block], [2, 2, 1])
            self.assertEqual(sorted(i for b in block for i, _ in b), list(range(5)))
            self.assertEqual({r for b in block for _, r in b}, {rotation})
        counts = collections.Counter(pair for b in batches for pair in b)
        self.assertEqual(set(counts.values()), {1})
        self.assertEqual(len(counts), 5 * 24)

    def test_seed_and_epoch_shuffle(self):
        a, b = RotationBatchSampler(9, 4, seed=8), RotationBatchSampler(9, 4, seed=8)
        first = list(a)
        self.assertEqual(first, list(b))
        second = list(a)
        self.assertEqual(second, list(b))
        self.assertNotEqual(first, second)

    def test_actual_optimizer_updates_and_workers(self):
        for schedule, expected in [('author24',72), ('single-none',3), ('single-random',3)]:
            for workers in (0, 2):
                sampler = RotationBatchSampler(5, 2, schedule, seed=9)
                loader = torch.utils.data.DataLoader(TinyDataset(), batch_sampler=sampler,
                                                     num_workers=workers)
                model = torch.nn.Linear(2, 1)
                optimiser = CountingSGD(model.parameters())
                loss, _ = epoch(model, loader, 'cpu', torch.nn.MSELoss(), optimiser)
                self.assertTrue(np.isfinite(loss))
                self.assertEqual(optimiser.calls, expected)

    def test_complex_grids_apply_discrete_rotations_before_voxelization(self):
        ds = ComplexGrids.__new__(ComplexGrids)
        ds.coords = [np.array([[1.,2.,3.]], dtype=np.float32)]
        ds.features = [np.zeros((1,19), dtype=np.float32)]
        ds.features[0][0,0] = 1.
        ds.features[0][0,12] = .425896
        ds.affinities = [7.]
        ds.max_dist, ds.spacing, ds.rotate = 10., 1., False
        points = []
        for rotation in range(24):
            grid, target = ds[(0, rotation)]
            points.append(tuple(torch.nonzero(grid[0])[0].tolist()))
            self.assertEqual(target.item(),7.)
            self.assertAlmostEqual(grid[12].sum().item(), 1., places=6)
            self.assertEqual(grid[0].sum().item(),1.)
        self.assertEqual(len(set(points)),24)
        torch.testing.assert_close(ds[0][0], ds[(0,0)][0], rtol=0, atol=0)
        np.testing.assert_array_equal(ds.coords[0], [[1.,2.,3.]])

    def test_conflicting_cli_schedule_is_rejected(self):
        import train
        for argv in [
            ['train.py', '--out', '/tmp/unused', '--rotate', '--rotation-schedule', 'author24'],
            ['train.py', '--out', '/tmp/unused', '--epochs', '0'],
        ]:
            with patch('sys.argv', argv), self.assertRaises(SystemExit) as raised:
                train.main()
            self.assertEqual(raised.exception.code, 2)

    def test_hdf_rotation_matches_complex_path(self):
        try:
            import h5py
        except ImportError:
            self.skipTest('h5py absent from offline base image')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'fixture.hdf'
            row=np.zeros((1,22),dtype=np.float32)
            row[0,:3]=[1.,2.,3.]
            row[0,3]=1.
            with h5py.File(path,'w') as f:
                f['a']=row
                f['a'].attrs['affinity']=7.
            ds=HDFGrids(str(path))
            self.assertEqual(len({tuple(torch.nonzero(ds[(0,r)][0][0])[0].tolist())
                                  for r in range(24)}),24)
            ds._file.close()


if __name__ == '__main__':
    unittest.main()
