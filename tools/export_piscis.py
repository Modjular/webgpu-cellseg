"""
export_piscis.py — export the pretrained Piscis `20251212` spot-detection
weights + reference data for the pure-NumPy port (`mini_piscis.py`).

This is the *only* step that needs PyTorch/`piscis`. It:
  1. Loads `Piscis(model_name='20251212')` (an EfficientNetV2 encoder + a
     multi-scale FPN decoder with style (FiLM) conditioning — see `SpotsModel`
     in `piscis/models/spots.py`).
  2. Flattens every weight to a flat float32 blob + a JSON manifest. Two kinds
     of unit get exported, matching the two conv conventions the model mixes:
       - encoder units are post-activation `conv -> BN -> act` with no conv
         bias, so BN folds into the conv (`.w`/`.b`, one op).
       - decoder units are *pre*-activation `BN -> act -> conv` with a conv
         bias, so BN cannot fold into the following conv; it's stored as a
         per-channel affine (`.scale`/`.shift`) applied to the conv's input,
         exactly as `mini_cellpose.py`'s `batchconv` does.
  3. Dumps, per sample: the standardized input, the raw network output
     ([3,H,W] = pooled label + 2 subpixel deltas), and Piscis's own spot
     coordinates, so the port can validate preprocessing, the forward pass,
     and spot detection independently.
  4. Self-checks a NumPy reference forward against the torch model (< 1e-4).

Outputs (per sample `<name>`):
  refdata/ps_<name>.input.bin    float32 [256,256]  (standardized, resized)
  refdata/ps_<name>.output.bin   float32 [3,256,256] (pooled label + deltas)
  refdata/ps_<name>.coords.npy   float32 [N,2]        (Piscis's own spot coords)
  refdata/ps_<name>.meta.json    shapes + threshold + spot count
  piscis_weights.bin, piscis_manifest.json  (written next to this script's
  sibling `../reference/`, where `mini_piscis.py` expects them)

Run:  .venv-piscis/bin/python tools/export_piscis.py
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import glob
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REFDIR = os.path.join(HERE, "refdata")
IMGDIR = os.path.join(HERE, "images_piscis")
WEIGHTS = os.path.join(HERE, "..", "reference", "piscis_weights.bin")
MANIFEST = os.path.join(HERE, "..", "reference", "piscis_manifest.json")
BN_EPS = 1e-5
MODEL_NAME = "20251212"


# ======================================================================
# 1. Weight export
# ======================================================================
def fold(sd, prefix):
    """Fold a post-activation `conv(bias=False) -> BN` unit (encoder) into one
    conv.  y = act(BN(conv(x))), so BN folds into the conv's weight/bias."""
    w = sd[f"{prefix}.conv.0.weight"].numpy().astype(np.float64)   # [Cout,Cin/groups,K,K]
    g = sd[f"{prefix}.conv.1.weight"].numpy().astype(np.float64)
    beta = sd[f"{prefix}.conv.1.bias"].numpy().astype(np.float64)
    mean = sd[f"{prefix}.conv.1.running_mean"].numpy().astype(np.float64)
    var = sd[f"{prefix}.conv.1.running_var"].numpy().astype(np.float64)
    scale = g / np.sqrt(var + BN_EPS)
    wf = w * scale[:, None, None, None]
    bf = beta - mean * scale
    return wf.astype("<f4"), bf.astype("<f4")


def bnstyle(sd, prefix, conv_idx):
    """A pre-activation `BN -> act -> conv(bias=True)` unit (decoder). BN sits
    on the conv's *input*, so it can't fold into the conv weight — store it as
    a separate per-channel affine (scale, shift), applied before the conv."""
    g = sd[f"{prefix}.conv.0.weight"].numpy().astype(np.float64)
    beta = sd[f"{prefix}.conv.0.bias"].numpy().astype(np.float64)
    mean = sd[f"{prefix}.conv.0.running_mean"].numpy().astype(np.float64)
    var = sd[f"{prefix}.conv.0.running_var"].numpy().astype(np.float64)
    scale = g / np.sqrt(var + BN_EPS)
    shift = beta - mean * scale
    w = sd[f"{prefix}.conv.{conv_idx}.weight"].numpy().astype("<f4")
    b = sd[f"{prefix}.conv.{conv_idx}.bias"].numpy().astype("<f4")
    return scale.astype("<f4"), shift.astype("<f4"), w, b


def export_weights(model):

    sd = {k: v.cpu() for k, v in model.model.state_dict().items()}
    tensors, blob, off = {}, [], 0

    def add(tag, arr):
        nonlocal off
        flat = np.ascontiguousarray(arr).ravel().astype("<f4")
        tensors[tag] = {"offset": off, "length": int(flat.size), "shape": list(arr.shape)}
        blob.append(flat)
        off += flat.size

    def add_fold(tag, prefix):
        w, b = fold(sd, prefix)
        add(f"{tag}.w", w)
        add(f"{tag}.b", b)

    def add_bnstyle(tag, prefix, conv_idx, dense_prefix=None):
        scale, shift, w, b = bnstyle(sd, prefix, conv_idx)
        add(f"{tag}.scale", scale)
        add(f"{tag}.shift", shift)
        add(f"{tag}.w", w)
        add(f"{tag}.b", b)
        if dense_prefix is not None:
            add(f"{tag}.dense_w", sd[f"{dense_prefix}.weight"].numpy().astype("<f4"))
            add(f"{tag}.dense_b", sd[f"{dense_prefix}.bias"].numpy().astype("<f4"))

    # ---- stem: conv(1->32, k3) + BN + SiLU, full resolution ----
    add_fold("enc.stem", "fpn.encoder.stem")

    # ---- stage 0: 4x FusedMBConv(expand=1), no SE, in=out=32, full res ----
    for i in range(4):
        add_fold(f"enc.s0.{i}", f"fpn.encoder.blocks.0.{i}.block.0")

    # ---- stage 1: maxpool, then 4x FusedMBConv(expand=2), no SE, ->64ch, 1/2 res ----
    # blocks.1.0 is the (parameter-free) MaxPool2d; the 4 repeats are blocks.1.1..1.4.
    for i in range(4):
        p = f"fpn.encoder.blocks.1.{i + 1}"
        add_fold(f"enc.s1.{i}.c0", f"{p}.block.0")   # expand: 3x3, in-> filters
        add_fold(f"enc.s1.{i}.c1", f"{p}.block.1")   # output: 1x1, filters->out, no act (folded w/o relu; port applies none)

    # ---- stage 2 & 3: maxpool, then 4x MBConv(expand=4, se=0.25), ->128/256ch ----
    for s, in_stage in ((2, 64), (3, 128)):
        for i in range(4):
            p = f"fpn.encoder.blocks.{s}.{i + 1}"
            add_fold(f"enc.s{s}.{i}.c0", f"{p}.block.0")          # expand: 1x1
            add_fold(f"enc.s{s}.{i}.dw", f"{p}.block.1")          # depthwise: 3x3, groups=filters
            add(f"enc.s{s}.{i}.se_reduce.w", sd[f"{p}.block.2.reduce.weight"].numpy().astype("<f4"))
            add(f"enc.s{s}.{i}.se_reduce.b", sd[f"{p}.block.2.reduce.bias"].numpy().astype("<f4"))
            add(f"enc.s{s}.{i}.se_expand.w", sd[f"{p}.block.2.expand.weight"].numpy().astype("<f4"))
            add(f"enc.s{s}.{i}.se_expand.b", sd[f"{p}.block.2.expand.bias"].numpy().astype("<f4"))
            add_fold(f"enc.s{s}.{i}.c3", f"{p}.block.3")          # output: 1x1, no act

    # ---- decoder: 4 UpConv "up_blocks" (one per encoder level) ----
    def add_upconv(tag, prefix):
        add_bnstyle(f"{tag}.proj", f"{prefix}.proj", conv_idx=1)                       # BatchConv: bn,conv (no act)
        add_bnstyle(f"{tag}.conv", f"{prefix}.conv", conv_idx=2)                       # BatchActConv: bn,act,conv
        for k in (0, 1, 2):
            add_bnstyle(f"{tag}.s{k}", f"{prefix}.convs_{k}.conv", conv_idx=2,
                        dense_prefix=f"{prefix}.convs_{k}.dense")

    for level in range(4):
        add_upconv(f"dec.up{level}", f"fpn.decoder.up_blocks.{level}")

    # ---- decoder: the upsample-refine chains that bring levels 3,2,1 to full res ----
    for i in range(3):
        for j in range(3 - i):
            add_upconv(f"dec.rz{i}.{j}", f"fpn.decoder.resize_up_blocks.{i}.{j}")

    # ---- output head: BatchActConv, 32->3, 1x1 ----
    add_bnstyle("dec.out", "fpn.output", conv_idx=2)

    np.concatenate(blob).astype("<f4").tofile(WEIGHTS)
    meta = {
        "input_size": list(model.input_size),
        "channels": model.channels,
        "adjustment": model.adjustment,
        "dilation_iterations": model.dilation_iterations,
        "model_name": MODEL_NAME,
    }
    json.dump({"tensors": tensors, "meta": meta}, open(MANIFEST, "w"), indent=1)
    print(f"wrote {WEIGHTS} ({off * 4 / 1e6:.1f} MB) + {MANIFEST} ({len(tensors)} tensors)")
    return load_weights()


def load_weights():
    meta = json.load(open(MANIFEST))
    blob = np.fromfile(WEIGHTS, dtype="<f4")
    W = {n: blob[t["offset"]:t["offset"] + t["length"]].reshape(t["shape"])
         for n, t in meta["tensors"].items()}
    return W, meta["meta"]


# ======================================================================
# 2. NumPy reference forward (self-check against torch)
# ======================================================================
def sigmoid(x):
    # Piecewise to avoid overflow in exp() for large |x|.
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def silu(x):
    return x * sigmoid(x)


def conv2d(x, w, b):
    """'same'-padded conv. x:[Cin,H,W] w:[Cout,Cin,K,K] b:[Cout]."""
    Cin, H, Wd = x.shape
    Cout, _, K, _ = w.shape
    p = K // 2
    xp = np.pad(x, ((0, 0), (p, p), (p, p))) if p else x
    out = np.zeros((Cout, H, Wd), np.float32)
    for ky in range(K):
        for kx in range(K):
            out += np.tensordot(w[:, :, ky, kx], xp[:, ky:ky + H, kx:kx + Wd], axes=([1], [0]))
    return out + b[:, None, None]


def dwconv2d(x, w, b):
    """Depthwise 'same'-padded conv. x:[C,H,W] w:[C,1,K,K] b:[C]."""
    C, H, Wd = x.shape
    K = w.shape[-1]
    p = K // 2
    xp = np.pad(x, ((0, 0), (p, p), (p, p))) if p else x
    out = np.zeros((C, H, Wd), np.float32)
    for ky in range(K):
        for kx in range(K):
            out += w[:, 0, ky, kx][:, None, None] * xp[:, ky:ky + H, kx:kx + Wd]
    return out + b[:, None, None]


def maxpool2(x):
    C, H, Wd = x.shape
    x = x[:, :H // 2 * 2, :Wd // 2 * 2]
    return x.reshape(C, H // 2, 2, Wd // 2, 2).max(axis=(2, 4))


def upsample2(x):
    return np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)


def cbA(W, tag, x, act=True):
    y = conv2d(x, W[f"{tag}.w"], W[f"{tag}.b"])
    return silu(y) if act else y


def se(W, tag, x):
    s = x.mean(axis=(1, 2))
    s = W[f"{tag}.se_reduce.w"][:, :, 0, 0] @ s + W[f"{tag}.se_reduce.b"]
    s = silu(s)
    s = W[f"{tag}.se_expand.w"][:, :, 0, 0] @ s + W[f"{tag}.se_expand.b"]
    s = sigmoid(s)
    return x * s[:, None, None]


def encoder_forward(W, x):
    x = cbA(W, "enc.stem", x)
    xd = []

    for i in range(4):
        x = x + cbA(W, f"enc.s0.{i}", x)
    xd.append(x)

    x = maxpool2(x)
    for i in range(4):
        y = cbA(W, f"enc.s1.{i}.c0", x)
        y = cbA(W, f"enc.s1.{i}.c1", y, act=False)
        x = x + y if x.shape[0] == y.shape[0] else y
    xd.append(x)

    for s_idx in (2, 3):
        x = maxpool2(x)
        for i in range(4):
            y = cbA(W, f"enc.s{s_idx}.{i}.c0", x)
            y = dwconv2d(y, W[f"enc.s{s_idx}.{i}.dw.w"], W[f"enc.s{s_idx}.{i}.dw.b"])
            y = silu(y)
            y = se(W, f"enc.s{s_idx}.{i}", y)
            y = cbA(W, f"enc.s{s_idx}.{i}.c3", y, act=False)
            x = x + y if x.shape[0] == y.shape[0] else y
        xd.append(x)

    return xd   # [level0 (32ch), level1 (64ch), level2 (128ch), level3 (256ch)]


def batchconv(W, tag, x, act=True, add=None, style=None):
    if add is not None:
        x = x + add
    if style is not None:
        feat = W[f"{tag}.dense_w"] @ style + W[f"{tag}.dense_b"]
        x = x + feat[:, None, None]
    x = x * W[f"{tag}.scale"][:, None, None] + W[f"{tag}.shift"][:, None, None]
    if act:
        x = silu(x)
    return conv2d(x, W[f"{tag}.w"], W[f"{tag}.b"])


def upconv(W, tag, x, y, style):
    proj = batchconv(W, f"{tag}.proj", x, act=False)
    a0 = batchconv(W, f"{tag}.conv", x)
    a1 = batchconv(W, f"{tag}.s0", a0, add=y, style=style)
    x2 = proj + a1
    a2 = batchconv(W, f"{tag}.s1", x2, style=style)
    a3 = batchconv(W, f"{tag}.s2", a2, style=style)
    return x2 + a3


def decoder_forward(W, style, xd):
    f = xd[3]
    feature_maps = None
    levels = [3, 2, 1, 0]
    for i in range(4):
        f = upconv(W, f"dec.up{i}", f, xd[levels[i]], style)
        f_up = f
        if i < 3:
            for j in range(3 - i):
                f_up = upsample2(f_up)
                f_up = upconv(W, f"dec.rz{i}.{j}", f_up, None, style)
        feature_maps = f_up if feature_maps is None else feature_maps + f_up
        f = upsample2(f)
    return feature_maps


def deformable_max_pool(labels, deltas, k):
    """labels:[H,W] deltas:[2,H,W] -> pooled_labels:[H,W].  Every pixel votes
    for round(index + delta); a pixel's pooled value is the max label over all
    pixels (in a k x k window) whose vote lands exactly on it."""
    H, W = labels.shape
    ph, pw = k // 2, k // 2
    ii, jj = np.mgrid[0:H, 0:W]
    ci = np.round(deltas[0] + ii).astype(np.int32)
    cj = np.round(deltas[1] + jj).astype(np.int32)
    ci_p = np.pad(ci, ((ph, ph), (pw, pw)))     # zero padding, matching torch Unfold
    cj_p = np.pad(cj, ((ph, ph), (pw, pw)))
    lab_p = np.pad(labels, ((ph, ph), (pw, pw)))
    pooled = np.zeros((H, W), dtype=labels.dtype)
    for dy in range(-ph, ph + 1):
        for dx in range(-pw, pw + 1):
            sub_ci = ci_p[ph + dy:ph + dy + H, pw + dx:pw + dx + W]
            sub_cj = cj_p[ph + dy:ph + dy + H, pw + dx:pw + dx + W]
            match = (sub_ci == ii) & (sub_cj == jj)
            sub_lab = lab_p[ph + dy:ph + dy + H, pw + dx:pw + dx + W]
            pooled = np.maximum(pooled, np.where(match, sub_lab, 0))
    return pooled


def net_forward(W, x, kernel_size):
    """x:[1,H,W] -> (pooled_labels[H,W], deltas[2,H,W])."""
    xd = encoder_forward(W, x)
    style = xd[3].mean(axis=(1, 2))
    style = style / np.sqrt((style ** 2).sum())
    feat = decoder_forward(W, style, xd)
    out = batchconv(W, "dec.out", feat)             # [3,H,W]
    labels = sigmoid(out[0])
    deltas = out[1:]
    pooled = deformable_max_pool(labels, deltas, kernel_size)
    return pooled, deltas


# ======================================================================
# 3. Preprocessing + spot coordinates (for refdata / self-check only —
#    mini_piscis.py has its own copy of peak_local_max etc.)
# ======================================================================
def standardize(x):
    return (x - x.mean()) / (x.std() + 1e-7)


# -------------------------------------------------------------------------- main
def main():
    import torch
    from piscis import Piscis

    os.makedirs(REFDIR, exist_ok=True)
    model = Piscis(model_name=MODEL_NAME, device="cpu")
    print(f"metadata: adjustment={model.adjustment} input_size={model.input_size} "
          f"channels={model.channels} dilation_iterations={model.dilation_iterations}")

    W, meta = export_weights(model)
    kernel_size = 2 * meta["dilation_iterations"] + 1
    Hin, Win = meta["input_size"]

    import tifffile
    for path in sorted(glob.glob(os.path.join(IMGDIR, "*.tif"))):
        name = os.path.splitext(os.path.basename(path))[0]
        raw = tifffile.imread(path).astype(np.float32)
        H, Wd = raw.shape
        assert (H, Wd) == (Hin, Win), f"{name}: {(H, Wd)} != model input_size {(Hin, Win)}"
        norm = standardize(raw)

        # torch forward
        with torch.no_grad():
            xt = torch.from_numpy(norm[None, None])
            labels_t, deltas_t = model.model(xt)
        labels_t = labels_t[0].numpy()
        deltas_t = deltas_t[0].numpy()

        # numpy reference forward
        labels_n, deltas_n = net_forward(W, norm[None], kernel_size)
        dlab = np.abs(labels_n - labels_t).max()
        ddelta = np.abs(deltas_n - deltas_t).max()

        # Piscis's own spot coordinates (reference for the port's postprocessing)
        coords = model.predict(raw, threshold=0.5)
        n_spots = len(coords)

        out = np.concatenate([labels_n[None], deltas_n], axis=0)   # [3,H,W]
        norm.astype("<f4").tofile(f"{REFDIR}/ps_{name}.input.bin")
        out.astype("<f4").tofile(f"{REFDIR}/ps_{name}.output.bin")
        np.save(f"{REFDIR}/ps_{name}.coords.npy", coords.astype(np.float32))
        json.dump({"name": name, "H": H, "W": Wd, "threshold": 0.5, "min_distance": 1,
                   "n_spots": n_spots},
                  open(f"{REFDIR}/ps_{name}.meta.json", "w"), indent=1)

        print(f"  {name}: {Wd}x{H}  fwd max|Δlabel|={dlab:.1e} max|Δdelta|={ddelta:.1e}  "
              f"spots={n_spots}  {'OK' if dlab < 1e-4 and ddelta < 1e-3 else 'CHECK'}")


if __name__ == "__main__":
    main()
