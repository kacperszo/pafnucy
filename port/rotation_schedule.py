"""Discrete cube rotations and rotation-major batching, without worker-local state."""
from itertools import permutations, product
import math

import numpy as np


def cube_rotations():
    """The 24 orientation-preserving signed permutations; identity first.

    Integer matrices keep exact half-grid coordinates on the same rounding ties.
    The set is the cube rotation group; indexing is local to this port.
    """
    matrices = []
    for axes in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            matrix = np.eye(3, dtype=np.int8)[:, axes] * signs
            if round(np.linalg.det(matrix)) == 1:
                matrices.append(matrix)
    matrices.sort(key=lambda m: (not np.array_equal(m, np.eye(3)), tuple(m.ravel())))
    return np.stack(matrices)


ROTATIONS = cube_rotations()
SCHEDULES = ('author24', 'single-none', 'single-random')


def transform(coords, rotation):
    if rotation == -1:
        # Legacy continuous-QR augmentation: one draw per item, not the cube group.
        q, _ = np.linalg.qr(np.random.randn(3, 3))
        return coords @ (q * np.sign(np.linalg.det(q)))
    return coords @ ROTATIONS[rotation]


class RotationBatchSampler:
    """Visit all complexes before advancing to the next rotation.

    Each rotation gets its own shuffled pass and final partial batch, as in the
    authors' training.py. Passing (item, rotation) to workers avoids race-prone
    mutation of a dataset rotation attribute between prefetches.
    """
    def __init__(self, size, batch_size, schedule='author24', seed=0):
        if size < 1 or batch_size < 1 or schedule not in SCHEDULES:
            raise ValueError('positive dataset/batch sizes and a known schedule are required')
        self.size, self.batch_size, self.schedule = size, batch_size, schedule
        self.rotations = tuple(range(24)) if schedule == 'author24' else (
            (-1,) if schedule == 'single-random' else (0,))
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.rotations) * math.ceil(self.size / self.batch_size)

    def __iter__(self):
        for rotation in self.rotations:
            indices = self.rng.permutation(self.size)
            for start in range(0, self.size, self.batch_size):
                yield [(int(i), rotation) for i in indices[start:start + self.batch_size]]
