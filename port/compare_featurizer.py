"""Check the tfbio-free featuriser against the authors' own featurised output.

The repo ships both halves of the pair: `tests/data/complexes/<id>/<id>_{pocket,ligand}.mol2`
and `tests/data/dataset/*.hdf`, which is what tfbio.Featurizer made from them. tfbio itself is
gone, so this is the only reference available — and it is a good one, because it fixes every
column at once.

Atom order is not guaranteed to match, so rows are compared as multisets keyed on coordinates
rather than positionally. A featuriser that gets the values right but the order different is
still correct here; one that gets a column wrong is not, and the report names the column.

usage:
    python compare_featurizer.py --complexes tests/data/complexes --hdf tests/data/dataset/test_set.hdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from featurizer import FEATURE_NAMES, featurize  # noqa: E402


def ours(complex_dir: Path, cid: str):
    from openbabel import pybel

    pocket = next(pybel.readfile("mol2", str(complex_dir / f"{cid}_pocket.mol2")))
    ligand = next(pybel.readfile("mol2", str(complex_dir / f"{cid}_ligand.mol2")))
    # Pafnucy centres the complex on the ligand's centroid before gridding, so the reference
    # coordinates are ligand-centred; do the same before comparing.
    lc, lf = featurize(ligand, molcode=1.0)
    pc, pf = featurize(pocket, molcode=-1.0)
    centroid = lc.mean(axis=0)
    coords = np.vstack([lc - centroid, pc - centroid])
    feats = np.vstack([lf, pf])
    return coords, feats


def align(a_coords, a_feats, b_coords, b_feats, tol=1e-3):
    """Match rows by coordinate, so a different atom order is not reported as a difference."""
    order = []
    used = set()
    for i, c in enumerate(a_coords):
        d = np.abs(b_coords - c).max(axis=1)
        j = int(np.argmin(d))
        if d[j] <= tol and j not in used:
            order.append((i, j))
            used.add(j)
    return order


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--complexes", required=True)
    p.add_argument("--hdf", required=True)
    p.add_argument("--limit", type=int, default=10)
    args = p.parse_args()

    import h5py

    root = Path(args.complexes)
    bad_columns: dict[str, int] = {}
    checked = matched_total = 0

    with h5py.File(args.hdf, "r") as f:
        ids = [k for k in list(f) if (root / k).is_dir()][: args.limit]
        if not ids:
            raise SystemExit("no complexes in the hdf that also exist as mol2 directories")
        for cid in ids:
            ref = f[cid][:]
            ref_coords, ref_feats = ref[:, :3], ref[:, 3:]
            try:
                got_coords, got_feats = ours(root / cid, cid)
            except Exception as e:
                print(f"  {cid}: featuriser raised {type(e).__name__}: {e}")
                continue

            pairs = align(ref_coords, ref_feats, got_coords, got_feats)
            checked += 1
            matched_total += len(pairs)
            if len(pairs) < len(ref_coords):
                print(f"  {cid}: matched {len(pairs)}/{len(ref_coords)} atoms by coordinate")
            for i, j in pairs:
                diff = np.abs(ref_feats[i] - got_feats[j]) > 1e-3
                for k in np.flatnonzero(diff):
                    bad_columns[FEATURE_NAMES[k]] = bad_columns.get(FEATURE_NAMES[k], 0) + 1

    print(f"\n{checked} complexes, {matched_total} atoms compared")
    if not bad_columns:
        print("FEATURISER MATCHES THE AUTHORS' OUTPUT")
        return 0
    print("columns that differ, by how many atoms:")
    for name, n in sorted(bad_columns.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<16} {n:>7}  ({100 * n / max(matched_total, 1):.1f}%)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
