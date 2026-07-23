"""
Export cyto3 (Cellpose CPnet) weights for WebGPU inference.

- Loads the cyto3 model via cellpose (v3.x).
- Folds each BatchNorm (which sits BEFORE its ReLU+Conv) into a per-channel
  scale/shift applied to the layer input:  y = scale * x + shift, with
      scale = gamma / sqrt(var + eps)
      shift = beta - gamma * mean / sqrt(var + eps)   (eps = 1e-5)
  (BN cannot be merged through the ReLU into the conv, so we keep it as a cheap
   elementwise op that the shader applies while reading the conv input.)
- Writes:
    cyto3_weights.bin      concatenated little-endian float32 tensors
    cyto3_manifest.json    { name -> {offset, length, shape}, plus meta }
- Includes an independent `reference_forward` (no BatchNorm modules, only the
  folded params) and checks it matches the real net to < 1e-3 max-abs.
  This reference IS the spec that the WGSL/JS forward pass must reproduce.

Run:  .venv312/bin/python export_weights.py
"""
import os
import json
import struct
import numpy as np
import torch
import torch.nn.functional as F
from cellpose import models

EPS = 1e-5
_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights", "cellpose-cyto3")
OUT_BIN = os.path.join(_OUTDIR, "weights.bin")
OUT_MANIFEST = os.path.join(_OUTDIR, "manifest.json")


# --------------------------------------------------------------------------
# 1. Load model + fold BatchNorm
# --------------------------------------------------------------------------
def fold_bn(sd, bn_prefix):
    """Return (scale, shift) float32 arrays for a BatchNorm at <bn_prefix>.*"""
    g = sd[bn_prefix + ".weight"].numpy().astype(np.float64)
    b = sd[bn_prefix + ".bias"].numpy().astype(np.float64)
    m = sd[bn_prefix + ".running_mean"].numpy().astype(np.float64)
    v = sd[bn_prefix + ".running_var"].numpy().astype(np.float64)
    inv = 1.0 / np.sqrt(v + EPS)
    scale = (g * inv).astype(np.float32)
    shift = (b - g * m * inv).astype(np.float32)
    return scale, shift


def build_folded(sd):
    """Extract every layer into a flat dict of named float32 arrays."""
    W = {}

    def batchconv(dst, bn, conv):
        # bn: prefix of BatchNorm (…​.0), conv: prefix of Conv (…​.2)
        scale, shift = fold_bn(sd, bn)
        W[dst + ".scale"] = scale
        W[dst + ".shift"] = shift
        W[dst + ".w"] = sd[conv + ".weight"].numpy().astype(np.float32)
        W[dst + ".b"] = sd[conv + ".bias"].numpy().astype(np.float32)

    def linear(dst, pre):
        W[dst + ".w"] = sd[pre + ".weight"].numpy().astype(np.float32)
        W[dst + ".b"] = sd[pre + ".bias"].numpy().astype(np.float32)

    # ---- encoder: downsample.down.res_down_{n} ----
    for n in range(4):
        p = f"downsample.down.res_down_{n}"
        d = f"down.{n}"
        # proj = batchconv0 (BN + conv1x1, no relu)
        batchconv(d + ".proj", f"{p}.proj.0", f"{p}.proj.1")
        for t in range(4):
            batchconv(f"{d}.conv{t}", f"{p}.conv.conv_{t}.0", f"{p}.conv.conv_{t}.2")

    # ---- decoder: upsample.up.res_up_{n} ----
    for n in range(4):
        p = f"upsample.up.res_up_{n}"
        d = f"up.{n}"
        batchconv(d + ".proj", f"{p}.proj.0", f"{p}.proj.1")
        # conv_0 is a plain batchconv
        batchconv(f"{d}.conv0", f"{p}.conv.conv_0.0", f"{p}.conv.conv_0.2")
        # conv_1..3 are batchconvstyle: batchconv + a style linear ("full")
        for t in (1, 2, 3):
            batchconv(f"{d}.conv{t}", f"{p}.conv.conv_{t}.conv.0",
                      f"{p}.conv.conv_{t}.conv.2")
            linear(f"{d}.conv{t}.full", f"{p}.conv.conv_{t}.full")

    # ---- output: batchconv (BN + relu + conv1x1) ----
    batchconv("output", "output.0", "output.2")
    return W


# --------------------------------------------------------------------------
# 2. Reference forward (folded params only) — the port spec
# --------------------------------------------------------------------------
def t(W, name):
    return torch.from_numpy(W[name])


def bc(W, dst, x, relu=True, add=None, style=None):
    """One batchconv: (optional add) -> BN(scale,shift) -> (relu) -> conv.
    If style is given, its per-channel projection is added before BN."""
    if add is not None:
        x = x + add
    if style is not None:
        feat = F.linear(style, t(W, dst + ".full.w"), t(W, dst + ".full.b"))
        x = x + feat.view(1, -1, 1, 1)
    x = x * t(W, dst + ".scale").view(1, -1, 1, 1) + t(W, dst + ".shift").view(1, -1, 1, 1)
    if relu:
        x = F.relu(x)
    w = t(W, dst + ".w")
    pad = w.shape[-1] // 2
    return F.conv2d(x, w, t(W, dst + ".b"), padding=pad)


def resdown(W, n, x):
    d = f"down.{n}"
    a1 = bc(W, f"{d}.conv1", bc(W, f"{d}.conv0", x))
    x = bc(W, f"{d}.proj", x, relu=False) + a1
    a3 = bc(W, f"{d}.conv3", bc(W, f"{d}.conv2", x))
    return x + a3


def resup(W, n, x, y, style):
    d = f"up.{n}"
    a0 = bc(W, f"{d}.conv0", x)
    a1 = bc(W, f"{d}.conv1", a0, add=y, style=style)
    x = bc(W, f"{d}.proj", x, relu=False) + a1
    a2 = bc(W, f"{d}.conv2", x, style=style)
    a3 = bc(W, f"{d}.conv3", a2, style=style)
    return x + a3


def reference_forward(W, x):
    # encoder
    xd = []
    y = x
    for n in range(4):
        if n > 0:
            y = F.max_pool2d(xd[n - 1], 2, 2)
        xd.append(resdown(W, n, y))
    # style
    s = F.avg_pool2d(xd[3], kernel_size=xd[3].shape[2:])
    s = s.flatten(1)
    s = s / (s.pow(2).sum(1, keepdim=True).sqrt())
    # decoder (nbaseup indexing: up.3 first, then upsample + up.2,1,0)
    x = resup(W, 3, xd[3], xd[3], s)
    for n in (2, 1, 0):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = resup(W, n, x, xd[n], s)
    # output head
    x = bc(W, "output", x)
    return x, s


# --------------------------------------------------------------------------
# 3. Serialize + validate
# --------------------------------------------------------------------------
def main():
    print("loading cyto3 …")
    m = models.CellposeModel(gpu=False, model_type="cyto3")
    net = m.net.eval()
    sd = net.state_dict()

    W = build_folded(sd)
    print(f"folded into {len(W)} tensors")

    # serialize (sorted names for a stable layout)
    manifest = {"tensors": {}, "nbase": net.nbase, "sz": net.sz,
                "nout": net.nout, "diam_mean": float(net.diam_mean.item())}
    blob = bytearray()
    offset = 0
    for name in sorted(W):
        arr = np.ascontiguousarray(W[name], dtype="<f4")
        b = arr.tobytes()
        manifest["tensors"][name] = {"offset": offset, "length": arr.size,
                                     "shape": list(arr.shape)}
        blob += b
        offset += arr.size
    with open(OUT_BIN, "wb") as f:
        f.write(blob)
    with open(OUT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {OUT_BIN} ({len(blob)/1e6:.2f} MB), {OUT_MANIFEST} "
          f"({len(W)} tensors, {offset} floats)")

    # validate reference vs real net
    torch.manual_seed(0)
    for tag, x in [("random", torch.randn(1, 2, 128, 128)),
                   ("random2", torch.randn(1, 2, 256, 224))]:
        with torch.no_grad():
            ref, sref = reference_forward(W, x)
            out, sout, _ = net(x)
        d = (ref - out).abs().max().item()
        ds = (sref - sout).abs().max().item()
        print(f"[{tag}] max|ref-net| output={d:.3e}  style={ds:.3e}  "
              f"{'OK' if d < 1e-3 else 'FAIL'}")


if __name__ == "__main__":
    main()
