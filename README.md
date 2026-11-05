# Pafnucy — a 3-D convolutional network on a voxelised complex

> A maintained fork. The authors' own README is kept as
> [README.upstream.md](README.upstream.md) for attribution and for their description of
> the method — **its build and run instructions are not current for this fork.**

## What it is

The complex is voxelised onto a 21x21x21 grid of 19 channels centred on the ligand,
and three convolutional blocks feed three dense layers. `features()` returns the 200-dim last hidden
layer and the head is a single `Linear(200, 1)`, which makes the encoder/head boundary the authors'
own and unusually clean.

Ported from TensorFlow 1.2 to torch 2.5.1. The weights are derived rather than downloaded:
`port/weights.npz` holds the seventeen tensors inference reads, extracted from the authors' TF
checkpoint (see **Weights**; the file is not in the repository).

## State

| | |
|---|---|
| CASF-2016 scoring | **R 0.696**, RMSE 1.608, n=285 — paper 0.78 / 1.42; the gap is pocket preparation, see below |
| embedding | 200d, **native** — the layer the head reads |
| CASF-2016 reproducibility | 285/285 complexes, max difference **0.0** between tiers |
| invariance | **rotation: none** — an axis-aligned voxel grid resamples when the complex turns |
| training | `train` and `finetune`, verified end to end |

## Build

```bash
podman build --format=docker -f port/Containerfile -t pafnucy-torch:latest .
podman build --format=docker -f port/Containerfile.reference -t pafnucy-ref:latest .  # TF, for deriving the weights
```

## Weights

The weights are derived, not downloaded, and **`port/weights.npz` (49 MB) is not in this
repository**: `*.npz` is gitignored, and so is the authors' TensorFlow checkpoint data file
(146 MB, two thirds of it Adam moments, over GitHub's file limit). Every `predict` and `embed`
command needs the npz, so regenerate it once from the authors' checkpoint. The authors' README
(`README.upstream.md`) gives their repository as `https://gitlab.com/cheminfIBB/pafnucy`, which
carries `results/batch5-2017-06-05T07:58:47-best.*`:

```bash
git remote add upstream https://gitlab.com/cheminfIBB/pafnucy
git fetch upstream && git checkout upstream/master -- results/      # brings back the .data file

podman build --format=docker -f port/Containerfile.reference -t pafnucy-ref:latest .
mkdir -p derived
podman run --rm --network=none \
    -v "$PWD/port:/work/port:ro" -v "$PWD/results:/ckpt:ro" -v "$PWD/derived:/out:rw,U" \
    pafnucy-ref:latest \
    python /work/port/extract_reference.py --checkpoint "/ckpt/batch5-2017-06-05T07:58:47-best" \
        --out /out --skip-graph
cp derived/weights.npz port/weights.npz
```

Run this before `podman build -f port/Containerfile`, which copies the working tree into the
image. The route was re-run for this README and gave all seventeen tensors identical to the
file the reported numbers came from. (`*.npz` and `results/*.data-*` are ignored, so none of
this shows up in `git status`.)

## Run it

These are the exact commands used to produce the numbers above. They need
`port/weights.npz`, which a fresh clone does not contain: see **Weights** first, then build.
Every one runs with `--network=none` and a read-only root filesystem.

Input is one directory per complex:

    <complexes>/<id>/<id>_protein.pdb
    <complexes>/<id>/<id>_ligand.sdf      # or .mol2; several models try both

Voxelises the complex onto a 21x21x21 grid of 19 channels with Open Babel. Reads `_pocket.mol2` if present, otherwise `_pocket.pdb`, whose partial charges Open Babel then assigns itself (Gasteiger). The authors' pockets carried Chimera's AMBER charges instead — see the notes.

```bash
# predict
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    localhost/pafnucy-torch:latest \
    sh -c 'cd /work/port && python predict_complexes.py --complexes /data --weights /work/port/weights.npz --out /outputs --device cpu'

# embed
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    localhost/pafnucy-torch:latest \
    sh -c 'cd /work/port && python predict_complexes.py --complexes /data --weights /work/port/weights.npz --out /outputs --device cpu --embed'

# train
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v /path/to/splits:/splits:ro \
    -v /path/to/cache:/cache:rw,U \
    --shm-size 4g \
    localhost/pafnucy-torch:latest \
    sh -c 'cd /work/port && python train.py --complexes /data --labels /splits/train.csv --out /outputs --seed 0 --device cpu --val-labels /splits/val.csv --cache /cache --epochs 30'

# finetune  (encoder frozen; drop --freeze-encoder to tune all of it)
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v /path/to/splits:/splits:ro \
    -v /path/to/cache:/cache:rw,U \
    --shm-size 4g \
    localhost/pafnucy-torch:latest \
    sh -c 'cd /work/port && python train.py --complexes /data --labels /splits/train.csv --out /outputs --seed 0 --device cpu --val-labels /splits/val.csv --cache /cache --epochs 30 --init-encoder /ckpt/encoder.pt --freeze-encoder'
```

## What comes out

| file | holds |
|---|---|
| `predictions.csv` | `complex_id,y_pred` |
| `embeddings.npz` | `ids` and `vectors`, 200-dim — the last hidden layer, which the authors' own `features()` returns. The convolutions have already pooled the grid, so nothing is invented |

## Before you trust the numbers

**Why this scores R 0.696 where the paper reports 0.78 — and why that is not the port.** The authors committed their own test-set predictions in 2017 (`results/batch5-2017-06-05T07:58:47-predictions.csv`), and those give exactly the paper's number: R 0.774, RMSE 1.423 over 290 complexes. On the 284 complexes shared with CASF-2016 as held here, their predictions give 0.769 and ours 0.696, with labels agreeing on all but two — so the whole gap is in the predictions. Fed the authors' own prepared pocket files (`tests/data/complexes/`), this port reproduces their 2017 predictions to within 0.14 pK on all five core-set complexes they ship; fed ours, it misses by up to 1.18. The difference is the pocket preparation. The authors ran every `_pocket.pdb` through UCSF Chimera — `addh`, `addcharge`, AMBER charges (`pdbbind_data.ipynb`, cell 17) — and partial charge is one of the 19 input channels; we read the PDB with Open Babel, which assigns Gasteiger charges. **Chimera is not free software, so that step is not reproduced**, and the gap stays, documented. `port/compare_with_authors.py` re-runs the whole decomposition.

**This model is not rotation invariant, and that is correct.** It bins atoms onto a 21x21x21 grid of 1 Å voxels *aligned to the coordinate axes*, so rotating the complex resamples it — rotating the input files measured 0.79 pK over five complexes. The authors knew: they augment training with the 24 axis-aligned rotations for exactly this reason. Translation is exact on the lattice (a whole number of ångströms gives 0.000e+00, every atom keeping its voxel) and bounded off it (0.056 pK for a sub-voxel shift). If you evaluate it on a re-oriented structure, that gap is the grid, not the chemistry.

## Porting notes

### What this fork changes in training

`--rotation-schedule` defaults to `author24`: 24 full shuffled passes per nominal epoch, one
for each of the 24 proper cube orientations, which is what the authors' `training.py:281-290`
does. That is **24 times** the optimiser steps of a single pass at the same dataset and batch
sizes. `single-none` restores the one-pass budget, `single-random` is continuous random
augmentation (`--rotate` is its legacy alias). Validation stays unrotated. Summaries record
the schedule, sample presentations and optimiser updates, so budgets can be compared
explicitly. `PATCHES.md` has the detail.

- The rotations are proper (determinant +1), never reflections, built from integer signed
  permutations so there is no floating-point drift at half-voxel boundaries.
- Rotation ids travel with the sample indices through the loader queues. Mutating a dataset
  attribute instead is unsafe with worker prefetch. Partial final batches are kept for every
  pass.
- The authors' `tfbio` package is unavailable, so its numeric rotation ids and index order
  could not be checked, and TensorFlow's L2/Adam trajectory is not claimed to be reproduced.
  Neither does any of this close the charge/pocket gap described above.

### Weights, layout and dependencies

- TF-to-torch weight conversion needs NDHWC flattening and SAME pooling. Do not simplify the
  layouts in `pafnucy_torch.py` without comparing against the original graph
  (`verify_port.py` does that against `golden.npz` from `extract_reference.py`).
- The torch image does not install h5py. Training from a complex directory works; the HDF5
  entry point (`--train-hdf`) needs h5py added.
- Open Babel's charge and pocket preparation differs from the authors' AMBER pipeline, so a
  port that passes the network-level comparison has not closed the preprocessing gap.
- `featurizer.py` is a from-scratch reimplementation of `tfbio`'s 19-feature grid, settled by
  reproducing the authors' own featurised arrays in `tests/data/dataset/` column for column
  (`compare_featurizer.py`).
- `reference/casf2016.csv` holds this port's CASF-2016 predictions (`complex_id,y_pred`, 285
  rows); `port/compare_with_authors.py` takes it as `--ours`.
