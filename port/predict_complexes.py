"""Predict affinity for a directory of complexes, with the ported network.

Replaces the authors' `predict.py`, which reads a prepared HDF5 and runs TensorFlow. This
takes the canonical complex layout the rest of the benchmark uses and does the featurising
itself, so nothing has to be staged first.

usage:
    python predict_complexes.py --complexes /data --weights /ckpt/weights.npz --out /outputs
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from featurizer import featurize, make_grid  # noqa: E402
from pafnucy_torch import CHARGE_SCALER, Pafnucy, grid_to_torch, load_tf_weights  # noqa: E402

CHARGE_COLUMN = 12


def read_complex(complex_dir: Path, cid: str) -> tuple[np.ndarray, np.ndarray]:
    """One complex -> (coords centred on the ligand, features), before voxelisation.

    Split out of `build_grid` so `train.py` can reach it. Training and inference have to
    featurise through the same code or the model is fitted on one distribution and scored on
    another — which would look like a modest, plausible loss of accuracy rather than a bug.
    The grid itself is built separately because training may rotate the coordinates first.
    """
    from openbabel import pybel

    def read(suffixes):
        for suffix in suffixes:
            path = complex_dir / f"{cid}{suffix}"
            if path.exists():
                return next(pybel.readfile(path.suffix.lstrip("."), str(path)))
        raise FileNotFoundError(f"no ligand/pocket file for {cid} among {suffixes}")

    ligand = read(["_ligand.mol2", "_ligand.sdf"])
    # The pocket, and only the whole protein as a last resort. This is not just about the
    # grid — that clips at max_dist anyway — but about the featuriser: Open Babel assigns
    # partial charges over the whole molecule it is given, so every atom gets a different
    # charge when the input is a full protein rather than the prepared pocket the authors
    # trained on. PDBbind ships `_pocket.pdb`; Pafnucy's own preparation used the equivalent.
    pocket = read(["_pocket.mol2", "_pocket.pdb", "_protein.pdb"])

    lig_coords, lig_feats = featurize(ligand, molcode=1.0)
    poc_coords, poc_feats = featurize(pocket, molcode=-1.0)
    centroid = lig_coords.mean(axis=0)
    coords = np.vstack([lig_coords - centroid, poc_coords - centroid])
    feats = np.vstack([lig_feats, poc_feats])
    return coords, feats


def build_grid(complex_dir: Path, cid: str, max_dist: float, spacing: float) -> np.ndarray:
    """One complex -> one grid, centred on the ligand as the authors' preparation does."""
    coords, feats = read_complex(complex_dir, cid)
    grid = make_grid(coords, feats, max_dist=max_dist, grid_resolution=spacing)
    # predict.py scales the partial-charge channel before the grid enters the network
    grid[..., CHARGE_COLUMN] /= CHARGE_SCALER
    return grid


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--complexes", required=True)
    p.add_argument("--weights", required=True, help="weights.npz from extract_reference.py")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-dist", type=float, default=10.0)
    p.add_argument("--grid-spacing", type=float, default=1.0)
    p.add_argument("--embed", action="store_true",
                   help="also write the 200-dim pre-head representation")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    model = load_tf_weights(Pafnucy(), args.weights).to(device).eval()
    print(f"device: {device}")

    root = Path(args.complexes)
    ids, preds, embeddings, failed = [], [], [], []
    for d in sorted(x for x in root.iterdir() if x.is_dir()):
        try:
            grid = build_grid(d, d.name, args.max_dist, args.grid_spacing)
            x = grid_to_torch(grid).to(device)
            with torch.no_grad():
                h = model.features(x)
                y = torch.relu(model.output(h))
            ids.append(d.name)
            preds.append(float(y.flatten()[0]))
            if args.embed:
                embeddings.append(h.cpu().numpy()[0])
        except Exception as e:
            failed.append(f"{d.name}: {type(e).__name__}: {e}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["complex_id", "y_pred"])
        w.writerows(zip(ids, preds))
    if args.embed and embeddings:
        np.savez(out / "embeddings.npz", ids=np.array(ids), vectors=np.stack(embeddings))

    if failed:
        print(f"{len(failed)} complexes failed:")
        for line in failed[:10]:
            print("  ", line)
    print(f"{len(ids)} predictions -> {out / 'predictions.csv'}")
    return 0 if ids else 1


if __name__ == "__main__":
    raise SystemExit(main())
