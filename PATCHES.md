# Rotation schedule repair — 2026-10-20

`port/train.py` defaults to `--rotation-schedule author24`: a shuffled full pass
per discrete cube orientation, with partial final batches kept for every pass.
This restores coverage and update count of training.py:281-290. Matrices are
exact signed permutations, identity first; the remaining ordering is local to
the port. Original tfbio's numeric rotation IDs cannot be verified offline.

Explicit reduced-budget modes are `single-none` (the previous default) and
`single-random` (continuous QR augmentation, also accepted as legacy `--rotate`).
Contradictory CLI options fail. History and summary record presentation and
optimiser-step counts; validation remains unrotated and runs once per epoch.
Training metrics are online augmented metrics, not the authors' extra unaugmented
end-of-epoch training evaluation. TF L2/Adam trajectory equivalence is not claimed.

Rotation IDs travel with sample indices through worker queues, without mutable
per-worker schedule state. The same mechanism supports ComplexGrids and HDFGrids.
Run port/test_rotation_schedule.py in the isolated torch image; its HDF5 check is
explicitly skipped if h5py is absent. The harness repair evidence records actual
train/reload and original-graph checks, using the existing offline images.
