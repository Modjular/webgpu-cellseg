"""
mini_cellpose.py — Cellpose (cyto3) cell segmentation in ~one file of NumPy.

In the spirit of nanoGPT/micrograd: no PyTorch, no CUDA, no `cellpose` package
at inference time — just NumPy — so you can read the whole thing and see exactly
what Cellpose does. It loads the pretrained cyto3 weights (exported once by
`export_weights.py`, which is the only step that needs torch) and runs the full
pipeline end to end.

Cellpose in three ideas
-----------------------
1. A U-Net looks at the image and predicts, at every pixel, a 2-D *flow vector*
   pointing toward the center of the cell that pixel belongs to, plus a scalar
   *cell probability*.
2. "Dynamics": every pixel is a particle that follows the flow field for a fixed
   number of steps. All the pixels of one cell drift to the same point (the
   cell's center), so they pile up together.
3. Wherever many particles converge is a cell. Give each pile a label, then paint
   that label back onto the pixels that ended up there. A quality-control pass
   deletes masks whose shape disagrees with the predicted flow.

The network (CPnet)
-------------------
A residual U-Net, `nbase = [2, 32, 64, 128, 256]`, 3x3 convs, 4 down / 4 up
levels. A global-average-pooled, L2-normalized "style" vector from the
bottleneck is injected into every decoder block. BatchNorm sits *before* its
ReLU+Conv, so `export_weights.py` folds each BN into a per-channel affine
(scale, shift) applied to the conv input — that's why you see `.scale`/`.shift`
here instead of BN modules. Output = 2 flow channels + 1 cell-probability.

Usage
-----
    python mini_cellpose.py images/cellpose_img_075.png
    python mini_cellpose.py images/cellpose_img_075.png --diameter 30 --out masks.npy

Weights (`cyto3_weights.bin` + `cyto3_manifest.json`) must sit next to this file;
regenerate them with `.venv312/bin/python export_weights.py` if missing.
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
    meta = json.load(open(manifest_path))
    blob = np.fromfile(bin_path, dtype="<f4")
    W = {}
    for name, t in meta["tensors"].items():
        W[name] = blob[t["offset"]: t["offset"] + t["length"]].reshape(t["shape"])
    return W, meta


# ======================================================================
# 1. NumPy neural-net primitives (all tensors are [C, H, W], no batch dim)
# ======================================================================
def relu(x):
    return np.maximum(x, 0.0)


def conv2d(x, w, b, pad):
    """2-D convolution, 'same' padding.  x:[Cin,H,W] w:[Cout,Cin,K,K] b:[Cout].

    Implemented as the textbook definition of convolution: a weighted sum over
    the K*K kernel taps of the input shifted under each tap. That keeps memory
    small (no big im2col matrix) and reads like the math — each tap contributes
    `w[:, :, ky, kx] @ shifted_input`.
    """
    Cin, H, Wd = x.shape
    Cout, _, K, _ = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    out = np.zeros((Cout, H, Wd), dtype=np.float32)
    for ky in range(K):
        for kx in range(K):
            patch = xp[:, ky:ky + H, kx:kx + Wd]            # [Cin, H, W]
            out += np.tensordot(w[:, :, ky, kx], patch, axes=([1], [0]))  # [Cout,H,W]
    return out + b[:, None, None]


def maxpool2(x):
    """2x2 max pool, stride 2.  x:[C,H,W] -> [C,H//2,W//2]."""
    C, H, Wd = x.shape
    x = x[:, :H // 2 * 2, :Wd // 2 * 2]
    return x.reshape(C, H // 2, 2, Wd // 2, 2).max(axis=(2, 4))


def upsample2(x):
    """Nearest-neighbor 2x upsample (matches F.interpolate mode='nearest')."""
    return np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)


def linear(w, b, v):
    """Dense layer for the style projection.  w:[Cout,Cin] b:[Cout] v:[Cin]."""
    return w @ v + b


def resize_bilinear(src, dh, dw):
    """Bilinear resize with pixel-center sampling (matches cv2 INTER_LINEAR,
    what Cellpose uses to rescale by diameter and to un-resize the flows)."""
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
    fy = np.clip(fy, 0, sh - 1); fx = np.clip(fx, 0, sw - 1)
    y0 = np.floor(fy).astype(int); x0 = np.floor(fx).astype(int)
    y1 = np.minimum(y0 + 1, sh - 1); x1 = np.minimum(x0 + 1, sw - 1)
    wy = (fy - y0)[:, None]; wx = (fx - x0)[None, :]
    out = (src[:, y0][:, :, x0] * (1 - wy)[None] * (1 - wx)[None] +
           src[:, y0][:, :, x1] * (1 - wy)[None] * wx[None] +
           src[:, y1][:, :, x0] * wy[None] * (1 - wx)[None] +
           src[:, y1][:, :, x1] * wy[None] * wx[None]).astype(np.float32)
    return out[0] if squeeze else out


# ======================================================================
# 2. CPnet forward pass (mirrors export_weights.reference_forward)
# ======================================================================
def batchconv(W, name, x, do_relu=True, add=None, style=None):
    """One CPnet unit:  (+add) -> (+style projection) -> BN affine -> (ReLU) -> conv.

    BN is folded to `scale`/`shift`. `add` is a residual/skip merged into the
    input; `style` is the global style vector, projected per-channel and added
    to the input before BN (this is how the decoder is conditioned on the whole
    image, not just the local receptive field).
    """
    if add is not None:
        x = x + add
    if style is not None:
        feat = linear(W[name + ".full.w"], W[name + ".full.b"], style)  # [Cin]
        x = x + feat[:, None, None]
    x = x * W[name + ".scale"][:, None, None] + W[name + ".shift"][:, None, None]
    if do_relu:
        x = relu(x)
    w = W[name + ".w"]
    return conv2d(x, w, W[name + ".b"], pad=w.shape[-1] // 2)


def res_down(W, n, x):
    """Encoder residual block: two residual conv pairs + a 1x1 projection skip."""
    d = f"down.{n}"
    a1 = batchconv(W, f"{d}.conv1", batchconv(W, f"{d}.conv0", x))
    x = batchconv(W, f"{d}.proj", x, do_relu=False) + a1     # projected skip
    a3 = batchconv(W, f"{d}.conv3", batchconv(W, f"{d}.conv2", x))
    return x + a3


def res_up(W, n, x, skip, style):
    """Decoder residual block: like res_down but style-conditioned, and it merges
    the encoder skip connection (`skip`) into the first styled conv."""
    d = f"up.{n}"
    a0 = batchconv(W, f"{d}.conv0", x)
    a1 = batchconv(W, f"{d}.conv1", a0, add=skip, style=style)
    x = batchconv(W, f"{d}.proj", x, do_relu=False) + a1
    a2 = batchconv(W, f"{d}.conv2", x, style=style)
    a3 = batchconv(W, f"{d}.conv3", a2, style=style)
    return x + a3


def cpnet_forward(W, x):
    """x:[2,H,W] -> out:[3,H,W]  (2 flow channels + 1 cell-probability)."""
    # ---- encoder: 4 levels, max-pool between them, keep each level for skips
    xd = []
    y = x
    for n in range(4):
        if n > 0:
            y = maxpool2(xd[n - 1])
        xd.append(res_down(W, n, y))
    # ---- style vector: global average pool of the bottleneck, L2-normalized
    s = xd[3].mean(axis=(1, 2))
    s = s / np.sqrt((s ** 2).sum())
    # ---- decoder: start at the bottleneck, then upsample + merge skips
    x = res_up(W, 3, xd[3], xd[3], s)
    for n in (2, 1, 0):
        x = upsample2(x)
        x = res_up(W, n, x, xd[n], s)
    # ---- output head: 1x1 conv -> 3 channels
    return batchconv(W, "output", x)


# ======================================================================
# 3. Preprocessing
# ======================================================================
def normalize99(img, lower=1, upper=99):
    """Contrast-stretch to the 1st..99th percentile — Cellpose's default norm."""
    p1, p99 = np.percentile(img, lower), np.percentile(img, upper)
    return (img - p1) / (p99 - p1 + 1e-10)


def pad_to(x, div=16):
    """Reflect-pad a [H,W] channel so both dims are multiples of `div`
    (the U-Net down/up-samples 4 times, i.e. by 16)."""
    H, W = x.shape
    Hp, Wp = -(-H // div) * div, -(-W // div) * div
    out = np.zeros((Hp, Wp), np.float32)
    out[:H, :W] = x
    if Hp > H:
        out[H:Hp, :W] = x[H - 1::-1][:Hp - H]               # reflect rows
    if Wp > W:
        out[:, W:Wp] = out[:, W - 1::-1][:, :Wp - W]         # reflect cols
    return out, (H, W)


# ======================================================================
# 4. Dynamics — follow the flow field to convergence points
# ======================================================================
def follow_flows(dy, dx, ys, xs, niter):
    """Euler-integrate each seed pixel (ys,xs) along the flow field (dy,dx) for
    `niter` steps, sampling the field with bilinear interpolation and clamping to
    the image. Returns the final (py, px) positions."""
    H, W = dy.shape
    py = ys.astype(np.float32).copy()
    px = xs.astype(np.float32).copy()
    Hm, Wm = H - 1, W - 1
    for _ in range(niter):
        yf = np.clip(np.floor(py).astype(int), 0, Hm)
        xf = np.clip(np.floor(px).astype(int), 0, Wm)
        yf1 = np.minimum(yf + 1, Hm); xf1 = np.minimum(xf + 1, Wm)
        ty = py - np.floor(py); tx = px - np.floor(px)
        w00 = (1 - ty) * (1 - tx); w01 = (1 - ty) * tx
        w10 = ty * (1 - tx);       w11 = ty * tx
        sdy = dy[yf, xf] * w00 + dy[yf, xf1] * w01 + dy[yf1, xf] * w10 + dy[yf1, xf1] * w11
        sdx = dx[yf, xf] * w00 + dx[yf, xf1] * w01 + dx[yf1, xf] * w10 + dx[yf1, xf1] * w11
        py = np.clip(py + sdy, 0, Hm)
        px = np.clip(px + sdx, 0, Wm)
    return py, px


def get_masks(py, px, ys, xs, H, W, rpad=20):
    """Turn convergence points into labels: histogram the final positions, seed a
    mask at every local maximum with enough hits, grow it over the dense region,
    then paint that label back onto each pixel via its convergence bucket."""
    Hh, Ww = H + 2 * rpad, W + 2 * rpad
    pty = np.clip(py.astype(int) + rpad, 0, H + rpad - 1)
    ptx = np.clip(px.astype(int) + rpad, 0, W + rpad - 1)

    hist = np.zeros((Hh, Ww), np.int32)                     # convergence histogram
    np.add.at(hist, (pty, ptx), 1)

    # seeds = 5x5 local maxima with > 10 hits
    seeds = []
    ys_s, xs_s = np.nonzero(hist > 10)
    for y, x in zip(ys_s, xs_s):
        win = hist[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
        if hist[y, x] >= win.max():
            seeds.append((y, x, hist[y, x]))
    if not seeds:
        return np.zeros((H, W), np.int32)
    seeds.sort(key=lambda s: s[2])                          # ascending; big wins ties

    # grow each seed within an 11x11 window: 5 rounds of dilate & (hist > 2)
    M = np.zeros((Hh, Ww), np.int32)
    for k, (sy, sx, _) in enumerate(seeds, start=1):
        y0, x0 = sy - 5, sx - 5
        hslc = np.zeros((11, 11), np.int32)
        ya, yb = max(0, y0), min(Hh, y0 + 11)
        xa, xb = max(0, x0), min(Ww, x0 + 11)
        hslc[ya - y0:yb - y0, xa - x0:xb - x0] = hist[ya:yb, xa:xb]
        cur = np.zeros((11, 11), bool); cur[5, 5] = True
        for _ in range(5):
            grown = cur.copy()
            grown[1:] |= cur[:-1]; grown[:-1] |= cur[1:]
            grown[:, 1:] |= cur[:, :-1]; grown[:, :-1] |= cur[:, 1:]
            cur = grown & (hslc > 2)
        yy, xx = np.nonzero(cur)
        gy, gx = yy + y0, xx + x0
        ok = (gy >= 0) & (gy < Hh) & (gx >= 0) & (gx < Ww)
        M[gy[ok], gx[ok]] = k

    labels = np.zeros((H, W), np.int32)
    labels[ys, xs] = M[pty, ptx]
    return labels


# ======================================================================
# 5. Quality control — masks_to_flows + flow_error + remove_bad_flow_masks
# ======================================================================
def masks_to_flows(labels, H, W):
    """Reconstruct a flow field *from* the label masks: run heat diffusion from
    each mask's center over its own pixels, then take the gradient. If a mask is a
    real cell, this reconstructed flow should match the network's predicted flow.
    Returns unit-normalized (dy_hat, dx_hat)."""
    dyh = np.zeros((H, W), np.float64)
    dxh = np.zeros((H, W), np.float64)
    maxL = int(labels.max())
    for l in range(1, maxL + 1):
        ys, xs = np.nonzero(labels == l)
        if ys.size == 0:
            continue
        minY, minX = ys.min(), xs.min()
        ly = (ys.max() - minY + 1) + 2                      # +2 for a 1px border
        lx = (xs.max() - minX + 1) + 2
        Y = ys - minY + 1; X = xs - minX + 1                # local, padded coords
        # center = mask pixel closest to the centroid
        imin = np.argmin((X - X.mean()) ** 2 + (Y - Y.mean()) ** 2)
        cidx = Y[imin] * lx + X[imin]
        idx = Y * lx + X
        neigh = [idx, idx - lx, idx + lx, idx - 1, idx + 1,
                 idx - lx - 1, idx - lx + 1, idx + lx - 1, idx + lx + 1]
        T = np.zeros(ly * lx, np.float64)
        for _ in range(2 * (ly + lx)):                      # diffuse
            T[cidx] += 1
            T[idx] = sum(T[nb] for nb in neigh) / 9.0       # Jacobi (uses old T)
        dyh[ys, xs] = T[idx + lx] - T[idx - lx]             # gradient = flow
        dxh[ys, xs] = T[idx + 1] - T[idx - 1]
    mag = np.sqrt(dyh ** 2 + dxh ** 2)
    return dyh / (mag + 1e-60), dxh / (mag + 1e-60)


def remove_bad_flow_masks(labels, dy, dx, H, W, threshold):
    """Delete masks whose reconstructed flow (from their shape) disagrees with the
    network's predicted flow (dy,dx already = dP/5). This is Cellpose's main QC:
    it removes spurious masks that noise seeded in the background."""
    maxL = int(labels.max())
    if maxL == 0:
        return labels
    dyh, dxh = masks_to_flows(labels, H, W)
    err = (dyh - dy) ** 2 + (dxh - dx) ** 2
    esum = np.zeros(maxL + 1); ecnt = np.zeros(maxL + 1)
    flat = labels.ravel()
    np.add.at(esum, flat, err.ravel())
    np.add.at(ecnt, flat, 1)
    merr = np.where(ecnt > 0, esum / np.maximum(ecnt, 1), 0.0)
    bad = np.nonzero(merr[1:] > threshold)[0] + 1
    if bad.size:
        labels = labels.copy()
        labels[np.isin(labels, bad)] = 0
    return labels


def filter_and_renumber(labels, H, W, min_size):
    """Drop empty/tiny masks, compact the label numbering to 1..N."""
    maxL = int(labels.max())
    if maxL == 0:
        return labels
    counts = np.bincount(labels.ravel(), minlength=maxL + 1)
    remap = np.zeros(maxL + 1, np.int32)
    nxt = 1
    for l in range(1, maxL + 1):
        if counts[l] >= min_size:
            remap[l] = nxt; nxt += 1
    return remap[labels]


# ======================================================================
# 6. Top-level pipeline
# ======================================================================
def segment(W, gray, diameter=30.0, cellprob_threshold=0.0,
            flow_threshold=0.4, min_size=15, niter=200, verbose=True):
    """Full Cellpose: grayscale image -> integer label mask [H,W]."""
    H, W_ = gray.shape
    log = print if verbose else (lambda *a, **k: None)

    # --- preprocess: normalize, optionally rescale so cells are ~30px, pad ---
    rescale = 30.0 / diameter if diameter and diameter > 0 else 1.0
    ch0 = normalize99(gray).astype(np.float32)
    H2, W2 = H, W_
    if abs(rescale - 1) > 1e-6:
        H2, W2 = max(1, round(H * rescale)), max(1, round(W_ * rescale))
        ch0 = resize_bilinear(ch0, H2, W2)
    padded, (Hc, Wc) = pad_to(ch0, 16)
    net_in = np.stack([padded, np.zeros_like(padded)], 0)   # 2nd channel is 0
    log(f"input {W_}x{H} -> work {W2}x{H2} -> padded {padded.shape[1]}x{padded.shape[0]}")

    # --- network forward ---
    out = cpnet_forward(W, net_in)                          # [3, Hp, Wp]
    dP = out[:2, :H2, :W2]                                  # flow (dy, dx)
    cellprob = out[2, :H2, :W2]                             # cell probability

    # --- resize flows/prob back to full resolution; dynamics run at full res ---
    if (H2, W2) != (H, W_):
        dP = resize_bilinear(dP, H, W_)
        cellprob = resize_bilinear(cellprob, H, W_)
        niter = max(1, round(niter / rescale))              # more px to cross now
    dy, dx = dP[0] / 5.0, dP[1] / 5.0                       # /5 = per-step size

    # --- dynamics: follow flows from every cell pixel to its convergence point ---
    ys, xs = np.nonzero(cellprob > cellprob_threshold)
    if ys.size == 0:
        log("no cell pixels found")
        return np.zeros((H, W_), np.int32)
    mdy = np.where(cellprob > cellprob_threshold, dy, 0.0)
    mdx = np.where(cellprob > cellprob_threshold, dx, 0.0)
    py, px = follow_flows(mdy, mdx, ys, xs, niter)
    log(f"followed {ys.size} pixels for {niter} steps")

    # --- cluster convergence points into masks, then QC ---
    labels = get_masks(py, px, ys, xs, H, W_)
    big = 0.4 * H * W_
    counts = np.bincount(labels.ravel())
    for l in np.nonzero(counts > big)[0]:                   # drop oversized masks
        if l > 0:
            labels[labels == l] = 0
    log(f"{int(labels.max())} candidate masks")
    if flow_threshold and flow_threshold > 0:
        labels = remove_bad_flow_masks(labels, dy, dx, H, W_, flow_threshold)
        log(f"{len(np.unique(labels)) - 1} masks after flow QC")
    labels = filter_and_renumber(labels, H, W_, min_size)
    log(f"{int(labels.max())} final masks")
    return labels


# ======================================================================
# 7. CLI
# ======================================================================
def load_gray(path):
    """Load an image as a float32 grayscale array (mean of RGB)."""
    from PIL import Image
    a = np.asarray(Image.open(path)).astype(np.float32)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    return a


def main():
    ap = argparse.ArgumentParser(description="Cellpose cyto3 in pure NumPy.")
    ap.add_argument("image", help="path to an image (png/tif/...)")
    ap.add_argument("--diameter", type=float, default=30.0)
    ap.add_argument("--cellprob_threshold", type=float, default=0.0)
    ap.add_argument("--flow_threshold", type=float, default=0.4)
    ap.add_argument("--min_size", type=int, default=15)
    ap.add_argument("--niter", type=int, default=200)
    ap.add_argument("--weights", default=os.path.join(HERE, "cyto3_weights.bin"))
    ap.add_argument("--manifest", default=os.path.join(HERE, "cyto3_manifest.json"))
    ap.add_argument("--out", default=None, help="save labels as .npy")
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        sys.exit(f"missing {args.weights} — run export_weights.py first (needs torch).")
    W, meta = load_weights(args.weights, args.manifest)
    print(f"loaded {len(W)} weight tensors  (nbase={meta['nbase']})")

    gray = load_gray(args.image)
    labels = segment(W, gray, diameter=args.diameter,
                     cellprob_threshold=args.cellprob_threshold,
                     flow_threshold=args.flow_threshold,
                     min_size=args.min_size, niter=args.niter)
    print(f"==> {int(labels.max())} cells")
    if args.out:
        np.save(args.out, labels)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
