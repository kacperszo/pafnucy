# Pafnucy — a 3-D convolutional network on a voxelised complex

> A fork maintained for [gnn-benchmark](../../README.md). The authors' own README is
> kept as [README.upstream.md](README.upstream.md) for attribution and for their
> description of the method — **its build and run instructions are not current for
> this fork.**

## What it is

The complex is voxelised onto a 21x21x21 grid of 19 channels centred on the ligand,
and three convolutional blocks feed three dense layers. `features()` returns the 200-dim last hidden
layer and the head is a single `Linear(200, 1)`, which makes the encoder/head boundary the authors'
own and unusually clean.

Ported from TensorFlow 1.2 to torch 2.5.1. The weights are derived rather than downloaded:
`port/weights.npz` holds the seventeen tensors inference reads, extracted from the authors' TF
checkpoint.

## State

| | |
|---|---|
| CASF-2016 scoring | **R 0.696**, RMSE 1.608, n=285 — paper 0.78 / 1.42; the gap is pocket preparation, see below |
| embedding | 200d, **native** — the layer the head reads |
| `gnnb verify` | 285/285, **0.0** |
| invariance | **rotation: none** — an axis-aligned voxel grid resamples when the complex turns |
| training | `train` and `finetune`, verified end to end |

## Build

```bash
podman build --format=docker -f port/Containerfile -t pafnucy-torch:latest .
podman build --format=docker -f port/Containerfile.reference -t pafnucy-ref:latest .  # TF, for deriving the weights
```

## Run it, without the harness

Generated from this model's adapter by `gnnb howto`, so these are the exact commands
the benchmark issues — regenerate with `python tools/sync_model_readmes.py`. Every one
runs with `--network=none` and a read-only root filesystem.

Input is one directory per complex:

    <complexes>/<id>/<id>_protein.pdb
    <complexes>/<id>/<id>_ligand.sdf      # or .mol2; several models try both

Voxelises the complex onto a 21x21x21 grid of 19 channels with Open Babel. Reads `_pocket.mol2` if present, otherwise `_pocket.pdb`, whose partial charges Open Babel then assigns itself (Gasteiger). The authors' pockets carried Chimera's AMBER charges instead — see the notes.

```bash
# pafnucy.torch — localhost/pafnucy-torch:latest
# source: models/pafnucy

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

**This is the one model in the roster that is not rotation invariant, and that is correct.** It bins atoms onto a 21x21x21 grid of 1 Å voxels *aligned to the coordinate axes*, so rotating the complex resamples it — `gnnb invariance` measures 0.79 pK over five complexes. The authors knew: they augment training with the 24 axis-aligned rotations for exactly this reason. Translation is exact on the lattice (a whole number of ångströms gives 0.000e+00, every atom keeping its voxel) and bounded off it (0.056 pK for a sub-voxel shift). If you compare this model against a graph model on a re-oriented structure, that gap is the grid, not the chemistry.

**The weights are derived, not downloaded.** `port/weights.npz` holds the seventeen tensors inference reads, extracted from the authors' TensorFlow-1 checkpoint by `port/extract_reference.py` running under the reference image. The checkpoint itself is not in this fork — it is 146 MB of mostly Adam moments and over GitHub's file limit — so `git fetch upstream` brings it back when the derivation needs rechecking.

## Maintainer notes

`CLAUDE.md` in this directory holds what breaks if it is changed back.
