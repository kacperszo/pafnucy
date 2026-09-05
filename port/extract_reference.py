"""Read Pafnucy's TensorFlow-1 checkpoint, and optionally run it, from modern TensorFlow.

Pafnucy was published in 2017 against TensorFlow 1.2, python 3.6, CUDA 8.0 and cuDNN 5.1.
None of that runs on a current GPU, so the original cannot be executed the way the authors
did. It does not need to be: TF 2.x still reads TF-1 checkpoints and can restore and run a
TF-1 meta graph through `tf.compat.v1`. That gives the two things a port needs —

  * the trained weights, as plain arrays, and
  * golden activations to compare a PyTorch reimplementation against

without ever installing TensorFlow 1.

The architecture itself is not documented anywhere in the repo: `training.py` builds the
network by calling `tfbio.net.make_SB_network`, and `tfbio` is the authors' own package,
absent from this checkout. The graph in the `.meta` file is therefore the only authoritative
description of the model, which is why this dumps it rather than trusting the README.

usage (inside a container with tensorflow>=2):
    python extract_reference.py --checkpoint results/<name>-best --out /outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def dump_weights(checkpoint: str, out: Path) -> dict:
    """Every variable in the checkpoint, by name, with its shape and values."""
    import tensorflow as tf

    reader = tf.train.load_checkpoint(checkpoint)
    shapes = reader.get_variable_to_shape_map()
    arrays, manifest = {}, {}
    for name in sorted(shapes):
        value = reader.get_tensor(name)
        arrays[name] = value
        manifest[name] = {"shape": list(np.shape(value)), "dtype": str(np.asarray(value).dtype)}
    np.savez(out / "weights.npz", **arrays)
    (out / "weights.json").write_text(json.dumps(manifest, indent=2))
    for name, meta in manifest.items():
        print(f"  {name:<40} {meta['shape']}")
    return manifest


def dump_graph(checkpoint: str, out: Path) -> None:
    """The op list, so the port is written against the graph rather than the paper."""
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()
    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.import_meta_graph(f"{checkpoint}.meta")
        saver.restore(sess, checkpoint)
        graph = tf.compat.v1.get_default_graph()
        lines = []
        for op in graph.get_operations():
            if op.type in ("Const", "Identity", "Assign", "NoOp", "VariableV2",
                           "RestoreV2", "SaveV2"):
                continue
            outs = ",".join(str(o.shape) for o in op.outputs)
            lines.append(f"{op.type:<16} {op.name:<48} -> {outs}")
        (out / "graph_ops.txt").write_text("\n".join(lines))
        print(f"\n{len(lines)} ops -> {out / 'graph_ops.txt'}")

        # the placeholders are the contract with the featuriser: input shape and any
        # training-time switches such as the dropout keep probability
        print("\nplaceholders:")
        for op in graph.get_operations():
            if op.type == "Placeholder":
                print(f"  {op.name:<40} {op.outputs[0].shape}")


def dump_golden(checkpoint: str, out: Path, n: int, grid: int, channels: int) -> None:
    """Run the original network on a fixed synthetic input and save what it returns.

    This is the reference the PyTorch port has to reproduce. A synthetic grid rather than real
    complexes on purpose: it removes the featuriser from the comparison entirely, so a
    difference can only come from the network. The featuriser is checked separately, against
    the same 19-feature grid construction already verified in MD_DL_BA.

    `predict.py` reaches the graph through named collections rather than tensor names —
    `input`, `output`, and `kp`, the dropout keep probability which inference pins to 1.0 — so
    those are used here too instead of guessing.
    """
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()
    rng = np.random.RandomState(0)
    # sparse and non-negative, like a real occupancy grid, so activations land in a realistic
    # range rather than saturating
    x = (rng.rand(n, grid, grid, grid, channels) < 0.05).astype(np.float32)
    x *= rng.rand(*x.shape).astype(np.float32)

    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.import_meta_graph(f"{checkpoint}.meta", clear_devices=True)
        saver.restore(sess, checkpoint)
        inp = tf.compat.v1.get_collection("input")[0]
        prediction = tf.compat.v1.get_collection("output")[0]  # not `out`: that is the path
        kp = tf.compat.v1.get_collection("kp")[0]
        print(f"input {inp.shape}  output {prediction.shape}")
        y = sess.run(prediction, feed_dict={inp: x, kp: 1.0})

    np.savez(out_path := out / "golden.npz", x=x, y=np.asarray(y))
    print(f"golden: {x.shape} -> {np.asarray(y).shape}  range "
          f"{float(np.min(y)):.4f}..{float(np.max(y)):.4f}  -> {out_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="path without the .index/.meta suffix")
    p.add_argument("--out", required=True)
    p.add_argument("--skip-graph", action="store_true")
    p.add_argument("--golden-n", type=int, default=8, help="synthetic complexes for the golden run")
    p.add_argument("--grid", type=int, default=21, help="grid side; 2*max_dist/spacing + 1")
    p.add_argument("--channels", type=int, default=19)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("variables in the checkpoint:")
    dump_weights(args.checkpoint, out)
    if not args.skip_graph:
        dump_graph(args.checkpoint, out)
        dump_golden(args.checkpoint, out, args.golden_n, args.grid, args.channels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
