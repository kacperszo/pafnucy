"""Check the PyTorch port against golden outputs from the original TensorFlow graph.

`extract_reference.py` runs the authors' restored graph on a fixed synthetic grid and saves
both the input and what the network returned. This feeds the same input through the port and
requires the same answer. A synthetic grid keeps the featuriser out of the comparison, so any
difference here is the network, not the input pipeline.

usage:
    python verify_port.py --weights /ref/weights.npz --golden /ref/golden.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pafnucy_torch import Pafnucy, grid_to_torch, load_tf_weights  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--golden", required=True)
    p.add_argument("--tolerance", type=float, default=1e-4)
    args = p.parse_args()

    model = load_tf_weights(Pafnucy(), args.weights)
    # eval(): the graph was run with keep_prob=1.0, so dropout must be off here too
    model.eval()

    data = np.load(args.golden)
    x, want = data["x"], data["y"]
    with torch.no_grad():
        got = model(grid_to_torch(x)).numpy()

    print(f"input {x.shape} -> reference {want.shape}, port {got.shape}")
    if want.shape != got.shape:
        print("SHAPE MISMATCH")
        return 1

    delta = np.abs(want - got)
    scale = max(float(np.abs(want).max()), 1e-12)
    print(f"reference {want.flatten()[:4].round(5)} ...")
    print(f"port      {got.flatten()[:4].round(5)} ...")
    print(f"max|Δ| = {delta.max():.3e}   relative = {delta.max() / scale:.3e}")

    ok = delta.max() / scale < args.tolerance
    print("\n" + ("PORT MATCHES THE ORIGINAL" if ok else "PORT DIVERGES"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
