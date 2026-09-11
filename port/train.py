"""Train Pafnucy, or fine-tune it on a transferred encoder.

The authors' `training.py` builds the graph through `tfbio.net`, logs to TensorBoard, and
drives TensorFlow 1 sessions; none of that survives the port. Their hyperparameters do, and
are the defaults here: Adam at 1e-5, batch 20, 20 epochs, dropout 0.5, L2 1e-3, and the model
kept only when validation error improves.

The part that matters beyond reproduction is `--init-encoder` with `--freeze-encoder`: load a
trained representation, attach a fresh head, and keep training. Pafnucy's encoder/head
boundary is the one the authors drew in their own graph, between the `fully_connected` and
`output` scopes — `features()` returns the 200-dim vector, `output` is a single linear layer.

Two ways in. `--train-hdf` takes the HDF5 the authors' preparation produces — one dataset per
complex, shape (N_atoms, 3 + 19), affinity in an attribute. `--complexes` with `--labels` takes a
directory of complexes and a csv instead, which is what the harness hands a trainer, and
featurises through `predict_complexes.read_complex` so training and inference cannot diverge.

Outputs follow the contract in `harness/training.py`: `model.pt`, `history.csv` and
`summary.json`.

usage:
    python train.py --complexes /data --labels /splits/train.csv \
        --val-labels /splits/val.csv --out /outputs --cache /cache
    python train.py --train-hdf training_set.hdf --val-hdf validation_set.hdf --out /outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from featurizer import make_grid  # noqa: E402
from pafnucy_torch import CHARGE_SCALER, Pafnucy, grid_to_torch, load_tf_weights  # noqa: E402
from predict_complexes import read_complex  # noqa: E402

CHARGE_COLUMN = 12


class HDFGrids(torch.utils.data.Dataset):
    """Complexes from a prepared HDF5, voxelised on access.

    Grids are built per item rather than cached: one is 21^3 x 19 floats (~340 kB), so a few
    thousand of them would not fit comfortably, and voxelising is cheap next to the forward
    pass.
    """

    def __init__(self, path: str, max_dist: float = 10.0, spacing: float = 1.0,
                 rotate: bool = False):
        import h5py

        self.path, self.max_dist, self.spacing, self.rotate = path, max_dist, spacing, rotate
        with h5py.File(path, "r") as f:
            self.ids = list(f)
            self.affinities = np.array([float(f[k].attrs["affinity"]) for k in self.ids],
                                       dtype=np.float32)
        self._file = None

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        import h5py

        # opened lazily so the handle is created inside each worker process
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        row = self._file[self.ids[i]][:]
        coords, feats = row[:, :3], row[:, 3:]
        if self.rotate:
            # the authors augment with 24 axis-aligned orientations; a random rotation matrix
            # from that group keeps the grid on its axes
            q, _ = np.linalg.qr(np.random.randn(3, 3))
            coords = coords @ (q * np.sign(np.linalg.det(q)))
        grid = make_grid(coords, feats, self.max_dist, self.spacing)
        grid[..., CHARGE_COLUMN] /= CHARGE_SCALER
        return grid_to_torch(grid)[0], torch.tensor([self.affinities[i]])


class ComplexGrids(torch.utils.data.Dataset):
    """Complexes from a directory and a label csv, voxelised on access.

    The same shape as `HDFGrids` and the same grid, reached from the input the harness actually
    provides. Featurisation goes through `predict_complexes.read_complex`, which is also what
    inference uses — Open Babel assigns partial charges over whatever molecule it is handed, so a
    second implementation here would give every atom a different charge and the divergence would
    read as a modest loss of accuracy rather than as a bug.

    Open Babel is the slow part, so the `(coords, features)` arrays are cached. The grid is not:
    one is 21^3 x 19 floats and rebuilding it is cheap next to the forward pass, which is also
    what makes `--rotate` free.
    """

    def __init__(self, complexes: str, labels: str, cache: str | None = None,
                 max_dist: float = 10.0, spacing: float = 1.0, rotate: bool = False):
        self.max_dist, self.spacing, self.rotate = max_dist, spacing, rotate
        with open(labels) as f:
            wanted = {r["complex_id"]: float(r["y_true"]) for r in csv.DictReader(f)}

        path = None
        if cache:
            path = Path(cache) / f"pafnucy_{Path(labels).stem}.npz"
            if path.exists():
                # written by this class, in this container: a dict of arrays, so it opens with
                # allow_pickle=False and needs no trust beyond that
                with np.load(path, allow_pickle=False) as stored:
                    self.ids = [str(k) for k in stored["ids"]]
                    self.affinities = stored["affinities"]
                    self.coords = [stored[f"c{i}"] for i in range(len(self.ids))]
                    self.features = [stored[f"f{i}"] for i in range(len(self.ids))]
                print(f"reusing {path}: {len(self.ids)} complexes")
                return

        self.ids, self.affinities, self.coords, self.features = [], [], [], []
        failed = []
        for cid in sorted(wanted):
            directory = Path(complexes) / cid
            if not directory.is_dir():
                failed.append(f"{cid}: no directory")
                continue
            try:
                coords, feats = read_complex(directory, cid)
            except Exception as e:
                failed.append(f"{cid}: {type(e).__name__}: {e}")
                continue
            self.ids.append(cid)
            self.affinities.append(wanted[cid])
            self.coords.append(coords.astype(np.float32))
            self.features.append(feats.astype(np.float32))
        self.affinities = np.asarray(self.affinities, dtype=np.float32)

        if failed:
            print(f"{len(failed)} complexes failed to featurise:")
            for line in failed[:10]:
                print("  ", line)
        if not self.ids:
            raise SystemExit(f"nothing featurised for {labels}")

        if path is not None:
            payload = {"ids": np.array(self.ids), "affinities": self.affinities}
            payload.update({f"c{i}": c for i, c in enumerate(self.coords)})
            payload.update({f"f{i}": f for i, f in enumerate(self.features)})
            np.savez(path, **payload)
            print(f"cached {len(self.ids)} complexes -> {path}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        coords = self.coords[i]
        if self.rotate:
            q, _ = np.linalg.qr(np.random.randn(3, 3))
            coords = coords @ (q * np.sign(np.linalg.det(q)))
        grid = make_grid(coords, self.features[i], self.max_dist, self.spacing)
        grid[..., CHARGE_COLUMN] /= CHARGE_SCALER
        return grid_to_torch(grid)[0], torch.tensor([self.affinities[i]])


def transfer_encoder(model: Pafnucy, path: str, freeze: bool) -> None:
    """Load an encoder into a fresh model, leaving the head untouched.

    Overlaid onto the model's own state dict and loaded strictly, rather than with
    `strict=False`: a partial load that silently moved nothing looks identical to success.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    target = model.state_dict()
    encoder = {k: v for k, v in state.items() if not k.startswith("output.")}
    unknown = [k for k in encoder if k not in target]
    mismatched = [k for k, v in encoder.items() if target[k].shape != v.shape]
    if unknown or mismatched:
        raise SystemExit(f"cannot transfer: unknown {unknown[:3]}, mismatched {mismatched[:3]}")
    model.load_state_dict({**target, **encoder}, strict=True)
    moved = sum(v.numel() for v in encoder.values())
    print(f"transferred {len(encoder)} tensors / {moved:,} params from {path}")
    if freeze:
        frozen = 0
        for name, param in model.named_parameters():
            if not name.startswith("output."):
                param.requires_grad_(False)
                frozen += param.numel()
        print(f"froze {frozen:,} encoder params; the head stays trainable")


def epoch(model, loader, device, criterion, optimizer) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total, n, ys, ps = 0.0, 0, [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += float(loss) * len(y)
            n += len(y)
            ys.append(y.detach().cpu().numpy().ravel())
            ps.append(out.detach().cpu().numpy().ravel())
    y, p = np.concatenate(ys), np.concatenate(ps)
    r = float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 and y.std() and p.std() else float("nan")
    return total / max(n, 1), r


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train-hdf", default=None,
                   help="the authors' prepared HDF5; or use --complexes with --labels")
    p.add_argument("--val-hdf", default=None)
    p.add_argument("--complexes", default=None,
                   help="directory of <id>/ complexes, as the harness binds at /data")
    p.add_argument("--labels", default=None, help="csv with complex_id,y_true")
    p.add_argument("--val-labels", dest="val_labels", default=None)
    p.add_argument("--cache", default=None,
                   help="directory kept between runs for the featurised arrays; Open Babel is "
                        "the slow part and it gives the same answer every time")
    p.add_argument("--out", required=True,
                   help="output directory: model.pt, history.csv and summary.json go in it")
    p.add_argument("--init-weights", default=None,
                   help="weights.npz from the TF checkpoint, to start from the published model")
    p.add_argument("--init-encoder", default=None,
                   help="a .pt state dict to transfer the encoder from")
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--rotate", action="store_true", help="augment with random rotations")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = Pafnucy(dropout=args.dropout)
    if args.init_weights:
        load_tf_weights(model, args.init_weights)
        print(f"started from the published weights: {args.init_weights}")
    if args.init_encoder:
        transfer_encoder(model, args.init_encoder, args.freeze_encoder)

    device = torch.device(args.device)
    print(f"device: {device}")
    model.to(device)

    # Either source, never both: two featurisation paths silently disagreeing about which
    # complexes are in the training set is not a failure anyone would notice from the loss curve.
    if bool(args.train_hdf) == bool(args.complexes):
        raise SystemExit("pass exactly one of --train-hdf and --complexes")
    if args.complexes and not args.labels:
        raise SystemExit("--complexes needs --labels to know what to train on")

    if args.complexes:
        train_ds = ComplexGrids(args.complexes, args.labels, cache=args.cache,
                                rotate=args.rotate)
        val_ds = (ComplexGrids(args.complexes, args.val_labels, cache=args.cache)
                  if args.val_labels else None)
    else:
        train_ds = HDFGrids(args.train_hdf, rotate=args.rotate)
        val_ds = HDFGrids(args.val_hdf) if args.val_hdf else None
    print(f"{len(train_ds)} train"
          + (f", {len(val_ds)} val complexes" if val_ds else " complexes, no validation split"))
    common = dict(batch_size=args.batch_size, num_workers=args.workers,
                  pin_memory=(device.type == "cuda"))
    train_dl = torch.utils.data.DataLoader(train_ds, shuffle=True, **common)
    val_dl = (torch.utils.data.DataLoader(val_ds, shuffle=False, **common) if val_ds else None)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam([q for q in model.parameters() if q.requires_grad],
                                 lr=args.lr, weight_decay=args.weight_decay)

    out = Path(args.out)
    out_dir = out.parent if out.suffix == ".pt" else out
    model_path = out if out.suffix == ".pt" else out / "model.pt"
    out_dir.mkdir(parents=True, exist_ok=True)

    history, best, best_epoch = [], float("inf"), -1
    started = time.time()
    for e in range(1, args.epochs + 1):
        tr_loss, tr_r = epoch(model, train_dl, device, criterion, optimizer)
        # The first five keys are the harness's standard history columns. Pafnucy's loss *is*
        # MSE, so train_loss and train_mse would be the same column; only the standard name is
        # written rather than both.
        row = {"epoch": e, "train_loss": tr_loss, "train_r": tr_r,
               "val_loss": "", "val_r": ""}
        flag = ""
        line = f"epoch {e:>3}  train mse {tr_loss:7.4f} r {tr_r:+.3f}"
        if val_dl:
            va_loss, va_r = epoch(model, val_dl, device, criterion, None)
            row.update(val_loss=va_loss, val_r=va_r)
            line += f"   val mse {va_loss:7.4f} r {va_r:+.3f}"
            score = va_loss
        else:
            score = tr_loss
        history.append(row)
        if score < best:
            best, best_epoch, flag = score, e, "  <- best"
            # plain tensors, so this reloads under weights_only=True
            torch.save(model.state_dict(), model_path)
        print(line + flag, flush=True)

    with open(out_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    (out_dir / "summary.json").write_text(json.dumps({
        "model": "pafnucy",
        "source": "complexes" if args.complexes else "hdf5",
        "epochs": len(history), "best_epoch": best_epoch, "best_score": best,
        "selected_on": "val_mse" if val_dl else "train_mse",
        "train_complexes": len(train_ds), "val_complexes": len(val_ds) if val_ds else 0,
        "seed": args.seed, "rotate": bool(args.rotate),
        "lr": args.lr, "batch_size": args.batch_size, "weight_decay": args.weight_decay,
        "init_encoder": args.init_encoder, "freeze_encoder": bool(args.freeze_encoder),
        "seconds": round(time.time() - started, 1),
    }, indent=2) + "\n")
    print(f"\nbest {best:.4f} at epoch {best_epoch} -> {model_path}  "
          f"({time.time() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
