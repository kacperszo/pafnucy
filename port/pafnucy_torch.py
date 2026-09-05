"""Pafnucy in PyTorch, weight-compatible with the published TensorFlow-1 checkpoint.

The original is TensorFlow 1.2 on CUDA 8 (2017), a stack that will not run on any current
GPU. The architecture is not written down anywhere in the repo either — `training.py` builds
it by calling `tfbio.net.make_SB_network`, and `tfbio` is the authors' own package from a
conda channel that no longer resolves. So this port was written against the graph itself,
recovered from the checkpoint's `.meta` file by `extract_reference.py`:

    input  (N, 21, 21, 21, 19)      grid 21^3 = 2*max_dist/spacing + 1, 19 features
    conv0  5^3  19->64   SAME  ReLU  maxpool/2  -> 11^3
    conv1  5^3  64->128  SAME  ReLU  maxpool/2  ->  6^3
    conv2  5^3 128->256  SAME  ReLU  maxpool/2  ->  3^3
    flatten -> 6912
    fc0 6912->1000 ReLU   fc1 1000->500 ReLU   fc2 500->200 ReLU
    output 200->1 ReLU

Three things have to be right or the weights load cleanly and the numbers come out wrong:

* **Layout.** TensorFlow is channels-last (NDHWC), PyTorch is channels-first (NCDHW), and
  conv kernels are stored as (kD,kH,kW,in,out) against PyTorch's (out,in,kD,kH,kW).
* **Flatten order.** `fc0` expects the 6912 values in TF's channels-last order. Flattening a
  channels-first tensor gives the same 6912 numbers permuted, which is invisible to every
  shape check and silently wrong.
* **Pooling.** TF's SAME pooling takes 21 -> 11 -> 6 -> 3; PyTorch reproduces that with
  `ceil_mode=True`, not with the default floor behaviour, which would give 10 -> 5 -> 2.

Verified against golden outputs from the original graph — see `verify_port.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

GRID = 21
CHANNELS = 19
# tfbio's Featurizer, in the order the grid stores them. Identical to the feature set used by
# Timenucy/Videonucy, which is where this project already has a verified reimplementation.
FEATURE_NAMES = ["B", "C", "N", "O", "P", "S", "Se", "halogen", "metal", "hyb",
                 "heavyvalence", "heterovalence", "partialcharge", "molcode",
                 "hydrophobic", "aromatic", "acceptor", "donor", "ring"]
# predict.py divides the partial-charge column by this before the grid enters the network
CHARGE_SCALER = 0.425896


class Pafnucy(nn.Module):
    """The 3-D CNN, with the head separable for transfer.

    `features` is the encoder: everything up to and including the last hidden layer. `output`
    is the affinity head — one linear layer and a ReLU. Keeping them apart is what lets the
    representation be reused; the boundary is the same one the authors' graph draws between
    the `fully_connected` and `output` scopes.
    """

    def __init__(self, dropout: float = 0.5) -> None:
        super().__init__()
        self.conv = nn.ModuleList([
            nn.Conv3d(CHANNELS, 64, kernel_size=5, padding=2),
            nn.Conv3d(64, 128, kernel_size=5, padding=2),
            nn.Conv3d(128, 256, kernel_size=5, padding=2),
        ])
        # ceil_mode reproduces TF's SAME pooling: 21 -> 11 -> 6 -> 3
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True)
        self.fc = nn.ModuleList([
            nn.Linear(256 * 3 * 3 * 3, 1000),
            nn.Linear(1000, 500),
            nn.Linear(500, 200),
        ])
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(200, 1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 19, 21, 21, 21) -> (N, 200). The representation, head removed."""
        for conv in self.conv:
            x = self.pool(torch.relu(conv(x)))
        # Back to channels-last before flattening: fc0's weights were trained against TF's
        # NDHWC ordering, and a channels-first flatten permutes the same values.
        x = x.permute(0, 2, 3, 4, 1).reshape(x.shape[0], -1)
        for layer in self.fc:
            x = self.dropout(torch.relu(layer(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.output(self.features(x)))


def load_tf_weights(model: Pafnucy, path: str | Path) -> Pafnucy:
    """Load the weights `extract_reference.py` dumped from the TF checkpoint."""
    w = np.load(path)
    with torch.no_grad():
        for i, conv in enumerate(model.conv):
            # (kD, kH, kW, in, out) -> (out, in, kD, kH, kW)
            kernel = w[f"convolution/conv{i}/w"].transpose(4, 3, 0, 1, 2)
            conv.weight.copy_(torch.from_numpy(np.ascontiguousarray(kernel)))
            conv.bias.copy_(torch.from_numpy(w[f"convolution/conv{i}/b"]))
        for i, layer in enumerate(model.fc):
            # TF stores (in, out); torch.nn.Linear stores (out, in)
            layer.weight.copy_(torch.from_numpy(w[f"fully_connected/fc{i}/w"].T.copy()))
            layer.bias.copy_(torch.from_numpy(w[f"fully_connected/fc{i}/b"]))
        model.output.weight.copy_(torch.from_numpy(w["output/w"].T.copy()))
        model.output.bias.copy_(torch.from_numpy(w["output/b"]))
    return model


def grid_to_torch(grid: np.ndarray) -> torch.Tensor:
    """(N, D, H, W, C) as TF and the featuriser produce it -> (N, C, D, H, W) for torch."""
    return torch.from_numpy(np.ascontiguousarray(grid.transpose(0, 4, 1, 2, 3))).float()
