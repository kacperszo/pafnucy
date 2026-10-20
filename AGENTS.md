# Pafnucy port

Read README.md and PATCHES.md. Work on model changes in a separate checkout,
then fast-forward the harness submodule only after isolated tests pass.
Use the existing PodmanRunner, offline, read-only inputs and dedicated outputs.

## Hard-won facts

- A full author epoch visits every training complex for each of 24 cube rotations.
  One random rotation per item is neither that group nor that optimisation budget.
- Rotation-major batches preserve the partial batch at the end of every pass.
  Pass rotation IDs with sample indices; mutating a dataset attribute is unsafe
  with worker prefetch. Validation remains at the identity orientation.
- The cube group uses proper rotations (determinant +1), never reflections.
  Integer signed permutations avoid floating-point drift at half-voxel boundaries.
  Original tfbio index ordering is unavailable locally; do not claim its order
  or an identical TensorFlow optimiser trajectory has been reproduced.
- Epochs with single-none/single-random have 24 times fewer sample presentations
  than author24. Record steps and schedule; compare budgets explicitly.
- TF-to-torch weights require NDHWC flattening and SAME pooling. Do not simplify
  the model layouts without comparison against the local original graph.
- port/weights.npz is gitignored and may exist only in the existing image.
  Preserve it when building offline overlays; see port/Containerfile.
- The offline torch base lacks h5py. Complex-directory training works, while the
  HDF5 entry point needs that dependency in a separately prepared environment.
- Open Babel charge/pocket preparation differs from the authors' AMBER pipeline.
  Passing a network reference test does not close that preprocessing gap.
