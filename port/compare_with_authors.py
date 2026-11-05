"""Where our Pafnucy numbers part from the paper's, and why.

The authors committed their own predictions to the repository in 2017
(`results/batch5-2017-06-05T07:58:47-predictions.csv`, 290 test complexes) and their own
prepared pocket files for a handful of core-set complexes (`tests/data/complexes/`). Together
they let the gap be taken apart instead of guessed at.

Part 1 needs nothing but three csv files and runs anywhere:

    python port/compare_with_authors.py --ours reference/casf2016.csv \
        --truth <csv with complex_id,y_true for CASF-2016>

Part 2 scores the authors' own pocket files with this port. It needs the image, and a directory
of complexes staged from `tests/data/complexes/` at the upstream commit:

    python port/compare_with_authors.py --ours ... --truth ... \
        --their-inputs /data --weights /work/port/weights.npz

Measured on 2026-09-24:

    their predictions, all 290                    R 0.7740  RMSE 1.423   (paper: 0.78, 1.42)
    their predictions, the 284 shared with ours   R 0.7693  RMSE 1.432
    labels that differ from ours                  2 of 284, max 0.04
    OUR predictions, the same 284                 R 0.6960  RMSE 1.611
    ours vs theirs, same weights and complexes    r 0.891, median |d| 0.48 pK, max 1.85

    this port on THEIR pocket files, 5 core complexes    max |d| vs their 2017 predictions 0.14
    this port on PDBbind's ligand + their pocket         max |d| 0.27
    this port on our own preparation                     max |d| 1.18

So the paper's number is exactly what their published predictions give, and this port
reproduces those predictions from their inputs. The gap is the pocket preparation. The authors
ran every `_pocket.pdb` through UCSF Chimera (`addh`, `addcharge` — AMBER charges) before
featurising (`pdbbind_data.ipynb`, cell 17); we read `_pocket.pdb` with Open Babel, which
assigns Gasteiger charges, and partial charge is one of Pafnucy's 19 input channels. Chimera is
not free software, so that step is not reproduced here and the gap stays, documented.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AUTHORS_PREDICTIONS = HERE.parent / "results" / "batch5-2017-06-05T07:58:47-predictions.csv"


def stats(pred, true) -> tuple[float, float]:
    p, t = np.asarray(pred, dtype=float), np.asarray(true, dtype=float)
    return float(np.corrcoef(p, t)[0, 1]), float(np.sqrt(((p - t) ** 2).mean()))


def read_column(path: Path, key: str, value: str) -> dict[str, float]:
    with open(path, newline="") as f:
        return {r[key]: float(r[value]) for r in csv.DictReader(f)}


def main() -> int:
    p = argparse.ArgumentParser(description="Decompose the gap between our Pafnucy and the paper")
    p.add_argument("--ours", required=True, type=Path, help="our predictions.csv on CASF-2016")
    p.add_argument("--truth", required=True, type=Path, help="csv with complex_id,y_true")
    p.add_argument("--their-inputs", type=Path, default=None,
                   help="directory of complexes staged from the authors' tests/data/complexes")
    p.add_argument("--weights", type=Path, default=HERE / "weights.npz")
    args = p.parse_args()

    with open(AUTHORS_PREDICTIONS, newline="") as f:
        theirs = {r["pdbid"]: (float(r["predicted"]), float(r["real"]))
                  for r in csv.DictReader(f) if r["set"] == "test"}
    ours = read_column(args.ours, "complex_id", "y_pred")
    truth = read_column(args.truth, "complex_id", "y_true")

    everything = sorted(theirs)
    r, e = stats([theirs[i][0] for i in everything], [theirs[i][1] for i in everything])
    print(f"their predictions, all {len(everything)}:{'':22}R {r:.4f}  RMSE {e:.3f}")

    shared = sorted(set(theirs) & set(ours) & set(truth))
    r, e = stats([theirs[i][0] for i in shared], [theirs[i][1] for i in shared])
    print(f"their predictions, the {len(shared)} shared with ours:{'':5}R {r:.4f}  RMSE {e:.3f}")

    labels = np.array([theirs[i][1] - truth[i] for i in shared])
    print(f"labels that differ from ours: {int((np.abs(labels) > 1e-6).sum())} of {len(shared)}, "
          f"max {np.abs(labels).max():.2f}")

    r, e = stats([ours[i] for i in shared], [truth[i] for i in shared])
    print(f"OUR predictions, the same {len(shared)}:{'':17}R {r:.4f}  RMSE {e:.3f}")

    d = np.array([ours[i] - theirs[i][0] for i in shared])
    r_between = float(np.corrcoef([ours[i] for i in shared], [theirs[i][0] for i in shared])[0, 1])
    print(f"ours vs theirs, same weights and complexes:  r {r_between:.3f}, "
          f"median |d| {np.median(np.abs(d)):.2f} pK, max {np.abs(d).max():.2f}")

    if args.their_inputs is None:
        return 0

    with tempfile.TemporaryDirectory() as out:
        subprocess.run([sys.executable, str(HERE / "predict_complexes.py"),
                        "--complexes", str(args.their_inputs), "--weights", str(args.weights),
                        "--out", out], check=True, capture_output=True)
        found = list(Path(out).rglob("predictions.csv"))
        mine = read_column(found[0], "complex_id", "y_pred")

    print(f"\nthis port on the authors' own pocket files ({len(mine)} complexes):")
    for cid in sorted(mine):
        print(f"  {cid}  theirs {theirs[cid][0]:.3f}   this port {mine[cid]:.3f}   "
              f"ours as usual {ours.get(cid, float('nan')):.3f}")
    print(f"  max |this port - theirs| = {max(abs(mine[c] - theirs[c][0]) for c in mine):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
