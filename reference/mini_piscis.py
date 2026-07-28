"""
mini_piscis.py — Piscis (`20251212`) fluorescence spot detection in ~one file
of NumPy.

In the spirit of nanoGPT/micrograd, and a sibling to `mini_cellpose.py` /
`mini_stardist.py` / `mini_instanseg.py`: no PyTorch, no CUDA, no `piscis`
package at inference time — just NumPy — so you can read the whole thing and
see exactly what Piscis does. It loads the pretrained `20251212` weights
(exported once by `export_piscis.py`, the only step that needs PyTorch) and
runs the full pipeline end to end: RNA-FISH-style spot detection, not cell
segmentation — the output is a list of (row, col) subpixel coordinates, one
per spot, rather than an instance-label image.

Piscis in three ideas
----------------------
1. A CNN looks at the image and predicts, at every pixel: a *label*
   confidence (does a spot sit near here?) and a 2-D *displacement* pointing
   from that pixel to the sub-pixel center of the nearest spot. This is the
   same "let every pixel vote for where the real center is" idea Cellpose
   uses for whole cells, just with a much smaller target — one bright blob a
   few pixels wide instead of an entire cytoplasm.
2. **Deformable max pooling.** Every pixel's displacement is rounded to the
   nearest pixel it "votes" for; each destination pixel's pooled confidence
   is the max label value among everyone who voted for it. Unlike a plain
   max-pool (fixed window, no notion of *where* a signal is coming from),
   this collapses a blurry blob of confident pixels down to one sharp peak
   exactly at the position the network agrees is the center — without any
   hand-tuned NMS kernel radius.
3. Ordinary local-maximum peak finding (matching `skimage.feature.
   peak_local_max`) then picks the surviving peaks above a confidence
   threshold and at least `min_distance` apart. Each peak's own predicted
   displacement — already baked into the pooled confidence map's geometry —
   is added back to give the final subpixel spot coordinate.

The network
-----------
An EfficientNetV2-style encoder (4 stages, `FusedMBConv`/`MBConv` blocks with
squeeze-excite in the last two stages, downsampling 8x total via max-pools
between stages — the stem itself does *not* downsample) feeds a multi-scale
FPN decoder. The decoder is "style"-conditioned exactly like Cellpose's
CPnet: a global-average-pooled, L2-normalized vector from the coarsest
encoder features is projected and added into every decoder block (see
`batchconv`/`upconv` below — they're structurally identical to
`mini_cellpose.py`'s `batchconv`/`res_up`). What's new relative to Cellpose is
the decoder's *shape*: each of the 4 encoder levels gets its own refined
feature map, which is then upsampled the rest of the way to full resolution
through a short chain of extra blocks, and all 4 full-resolution results are
summed — a multi-scale merge, rather than Cellpose's single top-down path.
Output = 1 label-logit channel + 2 displacement channels.

Two conv conventions are mixed in this checkpoint, and the exporter handles
each differently: the encoder is post-activation `conv -> BN -> act` with no
conv bias, so BatchNorm folds into the conv weights at export time; the
decoder is *pre*-activation `BN -> act -> conv` with a conv bias (so BN sits
on the conv's *input* and can't be folded), stored instead as a per-channel
affine `.scale`/`.shift` applied before the conv — see `batchconv` below.

Usage
-----
    python mini_piscis.py ../tools/images_piscis/02.tif
    python mini_piscis.py ../tools/images_piscis/02.tif --threshold 0.5 --out coords.npy
    python mini_piscis.py ../tools/images_piscis/02.tif --validate

Weights (`piscis_weights.bin` + `piscis_manifest.json`) must sit next to this
file; regenerate them (from the repo root) with
`.venv-piscis/bin/python tools/export_piscis.py`.

Caveat: Piscis normally tiles images larger than its 256x256 input via
DeepTile and stitches per-tile detections back together; this port instead
resizes the whole image to 256x256 in one shot (bilinear — not the upstream
anti-aliased `skimage.transform.resize`), which is exact for natively-256x256
crops (like `tools/images_piscis/*.tif`) and an approximation otherwise.
"""

import os
import sys
import json
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# ======================================================================
# 0. Weights
# ======================================================================
def load_weights(bin_path, manifest_path):
    """Return (W, meta): W maps tensor-name -> np.float32 array of its shape."""
    manifest = json.load(open(manifest_path))
    blob = np.fromfile(bin_path, dtype="<f4")
    W = {name: blob[t["offset"]: t["offset"] + t["length"]].reshape(t["shape"])
         for name, t in manifest["tensors"].items()}
    return W, manifest["meta"]


# ======================================================================
# 1. NumPy primitives (tensors are [C,H,W], no batch dim)
# ======================================================================
def sigmoid(x):
    # Piecewise, to avoid overflow in exp() for large |x|.
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def silu(x):
    return x * sigmoid(x)


def conv2d(x, w, b):
    """'Same'-padded conv. x:[Cin,H,W] w:[Cout,Cin,K,K] b:[Cout]."""
    Cin, H, Wd = x.shape
    Cout, _, K, _ = w.shape
    p = K // 2
    xp = np.pad(x, ((0, 0), (p, p), (p, p))) if p else x
    out = np.zeros((Cout, H, Wd), np.float32)
    for ky in range(K):
        for kx in range(K):
            patch = xp[:, ky:ky + H, kx:kx + Wd]
            out += np.tensordot(w[:, :, ky, kx], patch, axes=([1], [0]))
    return out + b[:, None, None]


def dwconv2d(x, w, b):
    """Depthwise 'same'-padded conv (one filter per channel, no mixing).
    x:[C,H,W] w:[C,1,K,K] b:[C]."""
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
    """2x2 max pool, stride 2. x:[C,H,W] -> [C,H//2,W//2]."""
    C, H, Wd = x.shape
    x = x[:, :H // 2 * 2, :Wd // 2 * 2]
    return x.reshape(C, H // 2, 2, Wd // 2, 2).max(axis=(2, 4))


def upsample2(x):
    """Nearest-neighbor 2x upsample."""
    return np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)


def resize_bilinear(src, dh, dw):
    """Bilinear resize with pixel-center sampling. src:[H,W] or [C,H,W]."""
    if src.ndim == 2:
        src = src[None]
        squeeze = True
    else:
        squeeze = False
    C, sh, sw = src.shape
    if sh == dh and sw == dw:
        return src[0] if squeeze else src
    fy = (np.arange(dh) + 0.5) * (sh / dh) - 0.5
    fx = (np.arange(dw) + 0.5) * (sw / dw) - 0.5
    fy = np.clip(fy, 0, sh - 1)
    fx = np.clip(fx, 0, sw - 1)
    y0 = np.floor(fy).astype(int)
    x0 = np.floor(fx).astype(int)
    y1 = np.minimum(y0 + 1, sh - 1)
    x1 = np.minimum(x0 + 1, sw - 1)
    wy = (fy - y0)[:, None]
    wx = (fx - x0)[None, :]
    out = (src[:, y0][:, :, x0] * (1 - wy)[None] * (1 - wx)[None] +
           src[:, y0][:, :, x1] * (1 - wy)[None] * wx[None] +
           src[:, y1][:, :, x0] * wy[None] * (1 - wx)[None] +
           src[:, y1][:, :, x1] * wy[None] * wx[None]).astype(np.float32)
    return out[0] if squeeze else out


# ======================================================================
# 2. Encoder — EfficientNetV2-style: FusedMBConv (stages 0-1), MBConv with
#    squeeze-excite (stages 2-3). Stem doesn't downsample; a 2x2 max-pool
#    sits between each stage, so levels 0..3 are at [1, 1/2, 1/4, 1/8] of the
#    input resolution with [32, 64, 128, 256] channels respectively.
# ======================================================================
def cbA(W, tag, x, act=True):
    """conv -> (BN, already folded in) -> optional SiLU."""
    y = conv2d(x, W[f"{tag}.w"], W[f"{tag}.b"])
    return silu(y) if act else y


def squeeze_excite(W, tag, x):
    """Global-context gate: squeeze to a per-channel scalar, two 1x1 "convs"
    (really just matmuls, since the spatial size is 1x1), sigmoid, rescale."""
    s = x.mean(axis=(1, 2))
    s = W[f"{tag}.se_reduce.w"][:, :, 0, 0] @ s + W[f"{tag}.se_reduce.b"]
    s = silu(s)
    s = W[f"{tag}.se_expand.w"][:, :, 0, 0] @ s + W[f"{tag}.se_expand.b"]
    s = sigmoid(s)
    return x * s[:, None, None]


def encoder_forward(W, x):
    """x:[1,H,W] -> [level0 (32ch,H), level1 (64ch,H/2), level2 (128ch,H/4),
    level3 (256ch,H/8)]."""
    x = cbA(W, "enc.stem", x)
    xd = []

    # Stage 0: 4x FusedMBConv(expand=1) — no expansion, no SE, just a plain
    # residual 3x3 conv block at full resolution.
    for i in range(4):
        x = x + cbA(W, f"enc.s0.{i}", x)
    xd.append(x)

    # Stage 1: maxpool, then 4x FusedMBConv(expand=2) — a fused 3x3
    # expand+depthwise conv, then a 1x1 projection (no activation). Residual
    # only once in==out (i.e. every repeat after the first, which changes
    # the channel count 32->64).
    x = maxpool2(x)
    for i in range(4):
        y = cbA(W, f"enc.s1.{i}.c0", x)
        y = cbA(W, f"enc.s1.{i}.c1", y, act=False)
        x = x + y if x.shape[0] == y.shape[0] else y
    xd.append(x)

    # Stages 2 & 3: maxpool, then 4x MBConv(expand=4, se=0.25) — 1x1 expand,
    # 3x3 depthwise, squeeze-excite, 1x1 project (no activation).
    for s in (2, 3):
        x = maxpool2(x)
        for i in range(4):
            y = cbA(W, f"enc.s{s}.{i}.c0", x)
            y = silu(dwconv2d(y, W[f"enc.s{s}.{i}.dw.w"], W[f"enc.s{s}.{i}.dw.b"]))
            y = squeeze_excite(W, f"enc.s{s}.{i}", y)
            y = cbA(W, f"enc.s{s}.{i}.c3", y, act=False)
            x = x + y if x.shape[0] == y.shape[0] else y
        xd.append(x)

    return xd


# ======================================================================
# 3. Decoder — FPN with style (FiLM) conditioning. `batchconv`/`upconv` are
#    structurally identical to mini_cellpose.py's `batchconv`/`res_up`: BN is
#    pre-activation here (applied to the conv's *input*), so it's stored as a
#    scale/shift pair rather than folded into the conv weight.
# ======================================================================
def batchconv(W, tag, x, act=True, add=None, style=None):
    """(+add) -> (+style projection) -> BN affine -> (SiLU) -> conv."""
    if add is not None:
        x = x + add
    if style is not None:
        feat = W[f"{tag}.dense_w"] @ style + W[f"{tag}.dense_b"]
        x = x + feat[:, None, None]
    x = x * W[f"{tag}.scale"][:, None, None] + W[f"{tag}.shift"][:, None, None]
    if act:
        x = silu(x)
    return conv2d(x, W[f"{tag}.w"], W[f"{tag}.b"])


def upconv(W, tag, x, skip, style):
    """One decoder block: refine `x`, merge a projected skip, refine twice
    more with style conditioning. `skip` may be None (the resize-refine
    chains below have no encoder skip to merge, only style)."""
    proj = batchconv(W, f"{tag}.proj", x, act=False)
    a0 = batchconv(W, f"{tag}.conv", x)
    a1 = batchconv(W, f"{tag}.s0", a0, add=skip, style=style)
    x2 = proj + a1
    a2 = batchconv(W, f"{tag}.s1", x2, style=style)
    a3 = batchconv(W, f"{tag}.s2", a2, style=style)
    return x2 + a3


def decoder_forward(W, style, xd):
    """Refine each encoder level with its own `upconv`, then upsample+refine
    every level except the finest the rest of the way to full resolution,
    and sum all 4 full-resolution contributions — a multi-scale merge,
    unlike Cellpose's single top-down decoder path."""
    f = xd[3]                      # start at the coarsest (bottleneck) level
    feature_maps = None
    levels = [3, 2, 1, 0]           # xd[-i-1] for i in 0..3
    for i in range(4):
        f = upconv(W, f"dec.up{i}", f, xd[levels[i]], style)
        f_up = f
        if i < 3:                  # levels 3,2,1 need 3,2,1 more upsamples to reach full res
            for j in range(3 - i):
                f_up = upsample2(f_up)
                f_up = upconv(W, f"dec.rz{i}.{j}", f_up, None, style)
        feature_maps = f_up if feature_maps is None else feature_maps + f_up
        f = upsample2(f)
    return feature_maps


def make_style(x):
    """Global-average-pooled, L2-normalized style vector from the bottleneck
    — the same recipe as mini_cellpose's CPnet style vector."""
    s = x.mean(axis=(1, 2))
    return s / np.sqrt((s ** 2).sum())


def net_forward(W, x, kernel_size):
    """x:[1,H,W] -> (pooled_labels[H,W] in [0,1], deltas[2,H,W])."""
    xd = encoder_forward(W, x)
    style = make_style(xd[3])
    feat = decoder_forward(W, style, xd)
    out = batchconv(W, "dec.out", feat)             # [3,H,W]: label logit + 2 deltas
    labels = sigmoid(out[0])
    deltas = out[1:]
    pooled = deformable_max_pool(labels, deltas, kernel_size)
    return pooled, deltas


# ======================================================================
# 4. Deformable max pooling
# ======================================================================
def deformable_max_pool(labels, deltas, k):
    """labels:[H,W] deltas:[2,H,W] -> pooled_labels:[H,W].

    Every pixel casts one vote for round(its own position + its predicted
    displacement). A pixel's pooled value is the max label confidence among
    all pixels in a k x k neighborhood whose vote lands exactly on it — so a
    blurry cloud of confident pixels around a spot collapses onto whichever
    single pixel the network's displacement field agrees is the center.
    """
    H, W = labels.shape
    ph = pw = k // 2
    ii, jj = np.mgrid[0:H, 0:W]
    ci = np.round(deltas[0] + ii).astype(np.int32)
    cj = np.round(deltas[1] + jj).astype(np.int32)
    ci_p = np.pad(ci, ((ph, ph), (pw, pw)))     # zero-padded: out-of-bounds votes never match
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


# ======================================================================
# 5. Preprocessing
# ======================================================================
def standardize(x):
    return (x - x.mean()) / (x.std() + 1e-7)


def normalize(x):
    x_min, x_max = x.min(), x.max()
    return (x - x_min) / (x_max - x_min + 1e-7)


def adjust(x, kind):
    return standardize(x) if kind == "standardize" else normalize(x)


# ======================================================================
# 6. Peak finding — a from-scratch equivalent of
#    `skimage.feature.peak_local_max(image, min_distance, threshold_abs,
#    exclude_border=False)` (the default `p_norm=inf`, i.e. Chebyshev
#    distance, NMS pass). No skimage/scipy: everything below is NumPy.
# ======================================================================
def _maximum_filter_square(img, radius):
    """Square max filter of size (2r+1), edge-replicate boundary — separable
    into two 1-D passes (equivalent to scipy.ndimage.maximum_filter(...,
    mode='nearest'))."""
    def max1d(a, axis):
        pad_width = [(0, 0)] * a.ndim
        pad_width[axis] = (radius, radius)
        ap = np.pad(a, pad_width, mode="edge")
        out = None
        for k in range(2 * radius + 1):
            sl = [slice(None)] * a.ndim
            sl[axis] = slice(k, k + a.shape[axis])
            cur = ap[tuple(sl)]
            out = cur if out is None else np.maximum(out, cur)
        return out
    return max1d(max1d(img, axis=0), axis=1)


def _ensure_spacing(coords, spacing):
    """Greedy NMS on a list of coordinates already sorted by descending
    intensity: keep a point, reject every other point within `spacing`
    (Chebyshev distance, strict '<'), repeat. Matches skimage's
    `_ensure_spacing` with the default `p_norm=inf`."""
    n = len(coords)
    if n == 0:
        return coords
    rejected = np.zeros(n, dtype=bool)
    keep = np.zeros(n, dtype=bool)
    for i in range(n):
        if rejected[i]:
            continue
        keep[i] = True
        d = np.abs(coords[i] - coords).max(axis=1)     # Chebyshev distance to all points
        close = d < spacing
        close[i] = False
        rejected |= close
    return coords[keep]


def peak_local_max(image, min_distance=1, threshold_abs=0.0):
    """(row, col) integer coordinates of local maxima, at least `min_distance`
    apart, above `threshold_abs`. `exclude_border=False` always (Piscis's
    default)."""
    img_max = _maximum_filter_square(image, min_distance)
    mask = image == img_max
    if mask.all():
        # A perfectly flat image has no well-defined peak — skimage returns none.
        return np.empty((0, 2), dtype=int)
    mask &= image > threshold_abs
    coords = np.argwhere(mask)
    if coords.shape[0] == 0:
        return coords.astype(int)
    intensities = image[mask]
    order = np.argsort(-intensities, kind="stable")
    coords = coords[order]
    return _ensure_spacing(coords, spacing=min_distance)


def compute_spot_coordinates(labels, deltas, threshold, min_distance):
    """labels:[H,W] (pooled confidence) deltas:[2,H,W] -> coords:[N,2] float,
    (row, col) with the network's own subpixel displacement applied."""
    peaks = peak_local_max(labels, min_distance=min_distance, threshold_abs=threshold)
    if len(peaks) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return peaks + deltas[:, peaks[:, 0], peaks[:, 1]].T


# ======================================================================
# 7. Top-level pipeline
# ======================================================================
def detect_spots(W, meta, gray, threshold=0.5, min_distance=1, verbose=True):
    """Full Piscis: grayscale image -> [N,2] float (row, col) spot coordinates."""
    log = print if verbose else (lambda *a, **k: None)
    H, Wd = gray.shape
    Hin, Win = meta["input_size"]
    kernel_size = 2 * meta["dilation_iterations"] + 1

    x = adjust(gray.astype(np.float32), meta["adjustment"])
    if (H, Wd) != (Hin, Win):
        x = resize_bilinear(x, Hin, Win)
        log(f"input {Wd}x{H} -> resized {Win}x{Hin} (approximate: bilinear, not anti-aliased)")
    scale_y, scale_x = Hin / H, Win / Wd

    labels, deltas = net_forward(W, x[None], kernel_size)
    log(f"pooled label map: max={labels.max():.3f}  mean={labels.mean():.4f}")

    coords = compute_spot_coordinates(labels, deltas, threshold, min_distance)
    if (H, Wd) != (Hin, Win):
        coords = coords / np.array([scale_y, scale_x])
    log(f"==> {len(coords)} spots")
    return coords


# ======================================================================
# 8. CLI + optional self-validation against refdata/
# ======================================================================
def load_gray(path):
    if path.lower().endswith((".tif", ".tiff")):
        import tifffile
        a = tifffile.imread(path).astype(np.float32)
    else:
        from PIL import Image
        a = np.asarray(Image.open(path)).astype(np.float32)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    return a


def main():
    ap = argparse.ArgumentParser(description="Piscis 20251212 spot detection in pure NumPy.")
    ap.add_argument("image", help="path to an image (tif/png/...)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min_distance", type=int, default=1)
    ap.add_argument("--weights", default=os.path.join(HERE, "piscis_weights.bin"))
    ap.add_argument("--manifest", default=os.path.join(HERE, "piscis_manifest.json"))
    ap.add_argument("--out", default=None, help="save coordinates as .npy")
    ap.add_argument("--validate", action="store_true",
                    help="compare forward pass + spots against tools/refdata/ps_<name>.*")
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        sys.exit(f"missing {args.weights} — run export_piscis.py first (needs PyTorch/piscis).")
    W, meta = load_weights(args.weights, args.manifest)
    print(f"loaded {len(W)} weight tensors  (input_size={meta['input_size']} "
          f"adjustment={meta['adjustment']})")

    gray = load_gray(args.image)
    coords = detect_spots(W, meta, gray, threshold=args.threshold, min_distance=args.min_distance)

    if args.validate:
        name = os.path.splitext(os.path.basename(args.image))[0]
        refdir = os.path.join(HERE, "..", "tools", "refdata")
        m = json.load(open(os.path.join(refdir, f"ps_{name}.meta.json")))
        kernel_size = 2 * meta["dilation_iterations"] + 1
        x = adjust(gray.astype(np.float32), meta["adjustment"])
        labels, deltas = net_forward(W, x[None], kernel_size)
        ref_out = np.fromfile(os.path.join(refdir, f"ps_{name}.output.bin"), "<f4").reshape(3, m["H"], m["W"])
        dlab = np.abs(labels - ref_out[0]).max()
        ddelta = np.abs(deltas - ref_out[1:]).max()
        ref_coords = np.load(os.path.join(refdir, f"ps_{name}.coords.npy"))
        print(f"  forward: max|Δlabel|={dlab:.1e}  max|Δdelta|={ddelta:.1e}")
        print(f"  spots: {len(coords)} vs ref {len(ref_coords)} "
              f"(ref computed by the actual torch model at threshold={m['threshold']})")

    if args.out:
        np.save(args.out, coords)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
