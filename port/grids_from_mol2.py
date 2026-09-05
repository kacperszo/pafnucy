"""Build Pafnucy input grids two ways, so the featuriser's effect on predictions is measurable.

Writes both the grid built from our featuriser and the grid built from the authors' own
featurised arrays in the test HDF5, for the same complexes. Feeding both through the network
turns a "0.8% of atom-features differ" statement into the number that actually matters: how
far apart the predictions end up.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from featurizer import featurize, make_grid  # noqa: E402

CHARGE_COLUMN = 12
CHARGE_SCALER = 0.425896


def scaled(grid: np.ndarray) -> np.ndarray:
    grid = grid.copy()
    grid[..., CHARGE_COLUMN] /= CHARGE_SCALER
    return grid


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--complexes", required=True)
    p.add_argument("--hdf", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import h5py
    from openbabel import pybel

    root = Path(args.complexes)
    ours, theirs, ids = [], [], []
    with h5py.File(args.hdf, "r") as f:
        for cid in list(f):
            d = root / cid
            if not d.is_dir():
                continue
            ref = f[cid][:]
            theirs.append(scaled(make_grid(ref[:, :3], ref[:, 3:])))

            ligand = next(pybel.readfile("mol2", str(d / f"{cid}_ligand.mol2")))
            pocket = next(pybel.readfile("mol2", str(d / f"{cid}_pocket.mol2")))
            lc, lf = featurize(ligand, molcode=1.0)
            pc, pf = featurize(pocket, molcode=-1.0)
            centroid = lc.mean(axis=0)
            ours.append(scaled(make_grid(np.vstack([lc - centroid, pc - centroid]),
                                         np.vstack([lf, pf]))))
            ids.append(cid)

    np.savez(args.out, ids=np.array(ids),
             ours=np.concatenate(ours), theirs=np.concatenate(theirs))
    print(f"{len(ids)} complexes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
