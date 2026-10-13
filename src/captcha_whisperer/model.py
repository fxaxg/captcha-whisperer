"""PyTorch definition of the captcha CRNN.

This mirrors the exported ONNX graph and is kept for inspection, retraining and
re-export. Everyday inference uses ONNX Runtime and does not import this module.

Architecture
------------
- A DenseNet-style fully convolutional trunk (input 3x60x160) that adds a
  downsampling stage from 64 to 128 channels and produces a 512x8x20 feature
  map.
- The feature map is reshaped into a sequence of 20 steps of 512*8 features.
- Two LSTMs (100 units each) process the sequence forwards and backwards; their
  outputs are concatenated into 200 features per step.
- A linear layer projects each step to 37 classes; training uses CTC loss.

Note on BatchNorm statistics: they are folded into a per-channel affine
(scale, shift) computed from the stored running sums at load time, so the
forward pass is a plain ``x * scale + shift`` per layer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import NUM_CLASSES

EPS = 1e-5


class _LSTM(nn.Module):
    """One LSTM direction, evaluated one time step at a time.

    The stored weights keep the gates in ``(input, forget, output, cell)`` order,
    while ``nn.LSTMCell`` expects ``(input, forget, cell, output)``; the gate
    blocks are reordered once at construction time so the forward loop is a plain
    cell call.
    """

    #: How our gate blocks map onto PyTorch's `(i, f, g, o)` blocks.
    _GATE_ORDER = (0, 1, 3, 2)

    def __init__(self, W: np.ndarray, b: np.ndarray, Wh: np.ndarray):
        super().__init__()
        hidden = b.shape[0] // 4
        input_dim = W.shape[0] // (4 * hidden)
        weight_ih = torch.from_numpy(np.array(W, dtype=np.float32)).reshape(4 * hidden, input_dim)
        weight_hh = torch.from_numpy(np.array(Wh, dtype=np.float32)).reshape(4 * hidden, hidden)
        bias = torch.from_numpy(np.array(b, dtype=np.float32))
        index = torch.cat([torch.arange(g * hidden, (g + 1) * hidden) for g in self._GATE_ORDER])
        self.cell = nn.LSTMCell(input_dim, hidden)
        with torch.no_grad():
            self.cell.weight_ih.copy_(weight_ih[index])
            self.cell.weight_hh.copy_(weight_hh[index])
            self.cell.bias_ih.copy_(bias[index])
            self.cell.bias_hh.zero_()

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Run the cell over ``seq`` of shape (time, batch, features)."""
        state = (torch.zeros(seq.shape[1], self.cell.hidden_size),) * 2
        out = []
        for t in range(seq.shape[0]):
            state = self.cell(seq[t], state)
            out.append(state[0])
        return torch.stack(out, 0)


class CaptchaCRNN(nn.Module):
    """The full model. Weights are loaded from the bundled ``.npz`` file."""

    def __init__(self, weights_file: str | Path):
        super().__init__()
        self._weights = dict(np.load(weights_file))

        def w(name: str) -> torch.Tensor:
            return torch.from_numpy(np.array(self._weights[name], dtype=np.float32))

        def conv(prefix: str, out_c: int, in_c: int, k: int, pad: int, stride: int) -> nn.Conv2d:
            m = nn.Conv2d(in_c, out_c, k, stride=stride, padding=pad, bias=False)
            with torch.no_grad():
                m.weight.copy_(w(prefix + ".0").reshape(out_c, in_c, k, k))
            return m

        def bn_scale(bn: str, sc: str) -> tuple[torch.Tensor, torch.Tensor]:
            mean = w(bn + ".0") / w(bn + ".2")
            var = w(bn + ".1") / w(bn + ".2")
            s = w(sc + ".0")
            b = w(sc + ".1")
            inv = 1.0 / torch.sqrt(var + EPS)
            scale = (s * inv).view(1, -1, 1, 1)
            shift = (b - s * inv * mean).view(1, -1, 1, 1)
            return scale, shift

        self._bn: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._conv: nn.ModuleDict = nn.ModuleDict()

        self._bn["data"] = bn_scale("data_bn", "data_scale")
        self._conv["conv1"] = conv("conv1", 64, 3, 7, 3, 2)
        self._bn["conv1"] = bn_scale("conv1_bn", "conv1_scale")

        # (block name, input channels, conv1x1 out, conv3x3 out, expand out, stride)
        blocks = [
            ("layer_64_1", 64, 64, 64, 256, 1),
            ("layer_64_2", 256, 64, 64, 256, 1),
            ("layer_64_3", 256, 64, 64, 256, 1),
            ("layer_128_1", 256, 128, 128, 512, 2),
            ("layer_128_2", 512, 128, 128, 512, 1),
            ("layer_128_3", 512, 128, 128, 512, 1),
            ("layer_128_4", 512, 128, 128, 512, 1),
        ]
        for name, in_c, c1, c3, expand, stride in blocks:
            self._conv[name + "_conv1"] = conv(name + "_conv1", c1, in_c, 1, 0, 1)
            self._conv[name + "_conv2"] = conv(name + "_conv2", c3, c1, 3, 1, stride)
            self._conv[name + "_conv3"] = conv(name + "_conv3", expand, c3, 1, 0, 1)
            self._bn[name + "_bn2"] = bn_scale(name + "_bn2", name + "_scale2")
            self._bn[name + "_bn3"] = bn_scale(name + "_bn3", name + "_scale3")
            # The very first block has no leading bn1/relu: it is fed by conv1_pool.
            if name != "layer_64_1":
                self._bn[name + "_bn1"] = bn_scale(name + "_bn1", name + "_scale1")
        self._conv["layer_128_1_conv_expand"] = conv("layer_128_1_conv_expand", 512, 256, 1, 0, 2)
        self._conv["layer_64_1_conv_expand"] = conv("layer_64_1_conv_expand", 256, 64, 1, 0, 1)
        self._bn["last"] = bn_scale("last_bn", "last_scale")

        self.lstm_fwd = _LSTM(self._weights["lstm1x.0"], self._weights["lstm1x.1"], self._weights["lstm1x.2"])
        self.lstm_bwd = _LSTM(self._weights["lstm2x.0"], self._weights["lstm2x.1"], self._weights["lstm2x.2"])
        self.fc_w = w("fc1x.0").reshape(NUM_CLASSES, 200)
        self.fc_b = w("fc1x.1")

    def _affine(self, x: torch.Tensor, key: str) -> torch.Tensor:
        scale, shift = self._bn[key]
        return x * scale + shift

    def _block(self, name: str, x: torch.Tensor) -> torch.Tensor:
        p = F.relu(self._affine(x, name + "_bn1"))
        d = F.relu(self._affine(self._conv[name + "_conv1"](p), name + "_bn2"))
        d = F.relu(self._affine(self._conv[name + "_conv2"](d), name + "_bn3"))
        return self._conv[name + "_conv3"](d) + x

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        x = self._affine(x, "data")
        x = F.relu(self._affine(self._conv["conv1"](x), "conv1"))
        x = F.max_pool2d(x, 3, stride=2, ceil_mode=True)

        # First block taps the ReLU'd conv1 output through its expand branch.
        p = F.relu(self._affine(self._conv["layer_64_1_conv1"](x), "layer_64_1_bn2"))
        d = F.relu(self._affine(self._conv["layer_64_1_conv2"](p), "layer_64_1_bn3"))
        x = self._conv["layer_64_1_conv3"](d) + self._conv["layer_64_1_conv_expand"](p)

        # Two more 64-channel dense blocks.
        x = self._block("layer_64_2", x)
        x = self._block("layer_64_3", x)

        # The 128-channel stage downsamples through its own expand branch.
        p = F.relu(self._affine(x, "layer_128_1_bn1"))
        d = F.relu(self._affine(self._conv["layer_128_1_conv1"](p), "layer_128_1_bn2"))
        d = F.relu(self._affine(self._conv["layer_128_1_conv2"](d), "layer_128_1_bn3"))
        x = self._conv["layer_128_1_conv3"](d) + self._conv["layer_128_1_conv_expand"](p)

        x = self._block("layer_128_2", x)
        x = self._block("layer_128_3", x)
        x = self._block("layer_128_4", x)
        return F.relu(self._affine(x, "last"))

    @staticmethod
    def _run_direction(lstm: _LSTM, seq: torch.Tensor, reverse: bool) -> torch.Tensor:
        """Run one LSTM over the sequence, optionally in reverse time order."""
        if reverse:
            return torch.flip(lstm(torch.flip(seq, [0])), [0])
        return lstm(seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.trunk(x)
        n, c, h, wd = feat.shape
        seq = feat.permute(3, 0, 1, 2).reshape(wd, n, c * h)
        fwd = self._run_direction(self.lstm_fwd, seq, reverse=False)
        bwd = self._run_direction(self.lstm_bwd, seq, reverse=True)
        merged = torch.cat([fwd, bwd], dim=2)
        return merged @ self.fc_w.t() + self.fc_b


def export_onnx(weights_file: str | Path, out_file: str | Path, opset: int = 17) -> None:
    """Build the model from ``weights_file`` and write an ONNX graph to ``out_file``."""
    model = CaptchaCRNN(weights_file).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dummy = torch.zeros(1, 3, 60, 160)
    torch.onnx.export(
        model,
        dummy,
        str(out_file),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {1: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
