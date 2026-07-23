"""
mini_instanseg.py — InstanSeg (brightfield_nuclei) nucleus segmentation in ~one
file of NumPy.

Third sibling to mini_cellpose.py / mini_stardist.py: no PyTorch, no `instanseg`
package at inference time — just NumPy — so the whole method is readable. It loads
the pretrained `brightfield_nuclei` weights (exported once by export_instanseg.py,
the only step needing torch) and runs the full pipeline end to end.

InstanSeg in three ideas
------------------------
1. A U-Net predicts, at every pixel: a 2-D **coordinate** (where the centre of my
   object is), a 2-D **sigma** (a learned local descriptor), and a scalar **seed**
   map (how centre-like this pixel is). The coordinate is a small learned offset
   added to the pixel's own (x, y).
2. **Seeds**: local maxima of the seed map are object centres. Each seed reads off
   its own predicted coordinate.
3. For each seed, a tiny learned **pixel-classifier** MLP looks at every pixel in a
   window and, from (that pixel's coordinate − the seed's coordinate, and its
   sigma), predicts "do you belong to me?". Threshold that → one instance mask.
   Assign each pixel to its highest-probability seed, then merge seeds that landed
   on the same object (high mask IoU). That's the segmentation — no flow dynamics
   (cf. cellpose), no polygons (cf. stardist); an object is grown from its centre
   by a learned per-pixel classifier.

The network is an InstanSeg_UNet (BatchNorm folded into each conv by
export_instanseg.py — it's post-activation conv→BN→ReLU, so BN folds into the conv
it follows). Output = 2 coord + 2 sigma + 1 seed channels.

Usage
-----
    python mini_instanseg.py images/cellpose_img_003.png --validate

Weights (`instanseg_bf_weights.bin` + `instanseg_bf_manifest.json`) must sit next
to this file; regenerate with `.venv-instanseg/bin/python export_instanseg.py`.
"""
import os
import sys
import json
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N_COORD, N_SIGMA = 2, 2


# ======================================================================
# 0. Weights
# ======================================================================
def load_weights(bin_path, manifest_path):
    meta = json.load(open(manifest_path))
    blob = np.fromfile(bin_path, dtype="<f4")
    W = {n: blob[t["offset"]:t["offset"] + t["length"]].reshape(t["shape"])
         for n, t in meta["tensors"].items()}
    return W, meta


# ======================================================================
# 1. NumPy primitives ([C,H,W], no batch dim)
# ======================================================================
def relu(x):
    return np.maximum(x, 0.0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def conv2d(x, w, b):
    Cin, H, Wd = x.shape
    Cout, _, K, _ = w.shape
    p = K // 2
    xp = np.pad(x, ((0, 0), (p, p), (p, p)))
    out = np.zeros((Cout, H, Wd), np.float32)
    for ky in range(K):
        for kx in range(K):
            out += np.tensordot(w[:, :, ky, kx], xp[:, ky:ky + H, kx:kx + Wd], axes=([1], [0]))
    return out + b[:, None, None]


def maxpool2(x):
    C, H, Wd = x.shape
    x = x[:, :H // 2 * 2, :Wd // 2 * 2]
    return x.reshape(C, H // 2, 2, Wd // 2, 2).max(axis=(2, 4))


def up2(x):
    return np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)


# ======================================================================
# 2. InstanSeg_UNet forward (mirrors export_instanseg.fcn_forward)
# ======================================================================
def cna(W, rf, tag, x):
    """conv (BN folded in) then optional ReLU."""
    y = conv2d(x, W[f"{tag}.w"], W[f"{tag}.b"])
    return relu(y) if rf[tag] else y


def enc_block(W, rf, n, x, pool):
    """Encoder residual block: proj + conv2(conv1) then + conv4(conv3)."""
    if pool:
        x = maxpool2(x)
    proj = cna(W, rf, f"enc{n}.c0", x)
    x = cna(W, rf, f"enc{n}.c1", x)
    x = proj + cna(W, rf, f"enc{n}.c2", x)
    x = x + cna(W, rf, f"enc{n}.c4", cna(W, rf, f"enc{n}.c3", x))
    return x


def dec_block(W, rf, n, x, skip):
    """Decoder block: upsample, merge the 1x1-projected skip, two residual convs."""
    x = up2(x)
    proj = cna(W, rf, f"dec{n}.c0", x)
    x = cna(W, rf, f"dec{n}.c1", x)
    x = proj + cna(W, rf, f"dec{n}.c2", x + cna(W, rf, f"dec{n}.skip", skip))
    x = x + cna(W, rf, f"dec{n}.c4", cna(W, rf, f"dec{n}.c3", x))
    return x


def fcn_forward(W, rf, x):
    """x:[3,H,W] -> [5,H,W]  (2 coord fields + 2 sigma + 1 seed)."""
    e0 = enc_block(W, rf, 0, x, pool=False)
    e1 = enc_block(W, rf, 1, e0, pool=True)
    e2 = enc_block(W, rf, 2, e1, pool=True)
    e3 = enc_block(W, rf, 3, e2, pool=True)
    d = dec_block(W, rf, 0, e3, e2)
    d = dec_block(W, rf, 1, d, e1)
    d = dec_block(W, rf, 2, d, e0)
    return np.concatenate([cna(W, rf, f"head{m}", d) for m in range(3)], axis=0)


# ======================================================================
# 3. Preprocessing
# ======================================================================
def percentile_normalize(img, p=0.1, eps=1e-3):
    """Per-channel (x - p0.1) / max(eps, p99.9 - p0.1) — InstanSeg's normalization."""
    img = img.astype(np.float32).copy()
    for c in range(img.shape[0]):
        lo, hi = np.percentile(img[c], [p, 100 - p])
        img[c] = (img[c] - lo) / max(eps, hi - lo)
    return img


def pad_to(x, div=8):
    """Reflect-pad [C,H,W] so H,W are multiples of div (the net pools 3x -> /8)."""
    C, H, W = x.shape
    Hp, Wp = -(-H // div) * div, -(-W // div) * div
    out = np.zeros((C, Hp, Wp), np.float32)
    out[:, :H, :W] = x
    if Hp > H:
        out[:, H:Hp, :W] = x[:, H - 1::-1][:, :Hp - H]
    if Wp > W:
        out[:, :, W:Wp] = out[:, :, W - 1::-1][:, :, :Wp - W]
    return out, (H, W)


def load_rgb(path):
    from PIL import Image
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
    return np.moveaxis(a, 2, 0)


# ======================================================================
# 4. Decode helpers
# ======================================================================
def coordinate_map(H, W):
    """InstanSeg's linear coordinate grid: xx varies along columns, yy along rows,
    both scaled by 64/256 (so 4 px = 1 coordinate unit)."""
    xx = np.linspace(0, W * 64 / 256, W, dtype=np.float32)[None, :].repeat(H, 0)
    yy = np.linspace(0, H * 64 / 256, H, dtype=np.float32)[:, None].repeat(W, 1)
    return np.stack([xx, yy], 0)                               # [2,H,W]: [col-coord, row-coord]


def peak_local_max(img, neighbourhood, minimum):
    """Seed centres = local maxima: a pixel that equals the max of its
    (2*neighbourhood+1)^2 window and exceeds `minimum`. Mirrors InstanSeg's
    max-pool peak finder (ties are rare on the smooth seed map)."""
    H, W = img.shape
    k = neighbourhood
    pad = np.full((H + 2 * k, W + 2 * k), -np.inf, np.float32)
    pad[k:k + H, k:k + W] = img
    mx = np.full((H, W), -np.inf, np.float32)
    for dy in range(2 * k + 1):
        for dx in range(2 * k + 1):
            mx = np.maximum(mx, pad[dy:dy + H, dx:dx + W])
    ys, xs = np.nonzero((img == mx) & (img > minimum))
    return np.stack([ys, xs], 1)                              # [C,2] (row,col)


def pixel_classifier(W, feat):
    """The learned MLP: 4 -> 5 -> 5 -> 1 (relu, relu, linear).  feat:[N,4] -> [N]."""
    x = relu(feat @ W["pc.fc1.w"].T + W["pc.fc1.b"])
    x = relu(x @ W["pc.fc2.w"].T + W["pc.fc2.b"])
    x = x @ W["pc.fc3.w"].T + W["pc.fc3.b"]
    return x[:, 0]


def flood_fill_keep(binary, sy, sx):
    """Keep only the connected component (4-connectivity) of `binary` containing
    the seed (sy,sx), then fill any interior holes — InstanSeg's per-crop cleanup."""
    H, W = binary.shape
    if not binary[sy, sx]:
        out = np.zeros_like(binary)
        out[sy, sx] = True
        return out
    keep = np.zeros_like(binary)
    stack = [(sy, sx)]
    keep[sy, sx] = True
    while stack:
        y, x = stack.pop()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < H and 0 <= nx < W and binary[ny, nx] and not keep[ny, nx]:
                keep[ny, nx] = True
                stack.append((ny, nx))
    # fill holes: background not reachable from the border is a hole
    bg = ~keep
    reach = np.zeros_like(binary)
    stack = [(y, 0) for y in range(H) if bg[y, 0]] + [(y, W - 1) for y in range(H) if bg[y, W - 1]]
    stack += [(0, x) for x in range(W) if bg[0, x]] + [(H - 1, x) for x in range(W) if bg[H - 1, x]]
    for y, x in stack:
        reach[y, x] = True
    while stack:
        y, x = stack.pop()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < H and 0 <= nx < W and bg[ny, nx] and not reach[ny, nx]:
                reach[ny, nx] = True
                stack.append((ny, nx))
    return keep | (bg & ~reach)


def connected_components(adj):
    """Union-find over an adjacency matrix -> component id per node (for merging
    seeds whose masks overlap heavily)."""
    n = adj.shape[0]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                parent[find(i)] = find(j)
    return np.array([find(i) for i in range(n)])


# ======================================================================
# 5. The InstanSeg decode
# ======================================================================
def decode(W, out, params):
    """[5,H,W] network output -> integer label mask [H,W]."""
    ws = params["window_size"]; mask_thr = params["mask_threshold"]
    H, Wd = out.shape[1:]
    xxyy = coordinate_map(H, Wd)
    fields = (sigmoid(out[0:N_COORD]) - 0.5) * 8 + xxyy        # [2,H,W] absolute coordinate
    sigma = out[N_COORD:N_COORD + N_SIGMA]                     # [2,H,W]
    seed_map = out[N_COORD + N_SIGMA] / 15.0 + 0.5            # [H,W]

    centroids = peak_local_max(seed_map, params["peak_distance"], params["seed_threshold"])
    C = len(centroids)
    if C == 0:
        return np.zeros((H, Wd), np.int32)

    # each seed's own coordinate
    cyx = centroids
    c = fields[:, cyx[:, 0], cyx[:, 1]].T                      # [C,2]

    sz = min(ws, H, Wd) * 2                                    # window is 2*window_size
    half = sz // 2
    best_prob = np.zeros((H, Wd), np.float32)                 # convert(): max-prob label per pixel
    labels = np.zeros((H, Wd), np.int32)
    masks = []                                                 # thresholded crop masks (for merge)

    for i in range(C):
        # window around the (clamped) centroid
        cy = int(np.clip(cyx[i, 0], half, H - half)) if H > sz else half
        cx = int(np.clip(cyx[i, 1], half, Wd - half)) if Wd > sz else half
        y0, x0 = cy - half, cx - half
        ys = np.clip(np.arange(y0, y0 + sz), 0, H - 1)
        xs = np.clip(np.arange(x0, x0 + sz), 0, Wd - 1)
        gy, gx = np.meshgrid(ys, xs, indexing="ij")           # [sz,sz] absolute coords

        # feature = [coord - seed_coord, sigma] at each window pixel
        f0 = fields[0, gy, gx] - c[i, 0]
        f1 = fields[1, gy, gx] - c[i, 1]
        s0 = sigma[0, gy, gx]; s1 = sigma[1, gy, gx]
        feat = np.stack([f0, f1, s0, s1], -1).reshape(-1, 4)
        prob = sigmoid(pixel_classifier(W, feat)).reshape(sz, sz)

        binary = prob >= mask_thr
        if params["cleanup_fragments"]:
            sy = int(np.clip(cyx[i, 0] - y0, 0, sz - 1))
            sx = int(np.clip(cyx[i, 1] - x0, 0, sz - 1))
            filled = flood_fill_keep(binary, sy, sx)
            prob = np.where(filled, np.maximum(prob, mask_thr), 0.0)  # drop fragments, fill holes
            binary = prob >= mask_thr

        # convert(): each pixel takes the label of its highest-probability seed
        pr = prob[binary]
        yy, xx = gy[binary], gx[binary]
        win = pr > best_prob[yy, xx]
        best_prob[yy[win], xx[win]] = pr[win]
        labels[yy[win], xx[win]] = i + 1
        masks.append(set(zip(yy.tolist(), xx.tolist())))

    labels = _merge_and_filter(labels, masks, C, params)
    return labels


def _merge_and_filter(labels, masks, C, params):
    """Merge seeds whose thresholded masks overlap (IoU > overlap_threshold) — the
    same object found twice — then drop instances below min_size and renumber."""
    areas = np.array([len(m) for m in masks])
    adj = np.zeros((C, C), bool)
    for i in range(C):
        if not masks[i]:
            continue
        for j in range(i + 1, C):
            if not masks[j]:
                continue
            inter = len(masks[i] & masks[j])
            if inter and inter / (areas[i] + areas[j] - inter) > params["overlap_threshold"]:
                adj[i, j] = True
    comp = connected_components(adj)                          # component per seed (0-based)
    remap = np.zeros(C + 1, np.int32)
    remap[1:] = comp + 1                                       # label i+1 -> its component
    merged = remap[labels]

    # min_size + compact to 1..N
    maxL = int(merged.max())
    if maxL == 0:
        return merged
    counts = np.bincount(merged.ravel(), minlength=maxL + 1)
    nxt, out_remap = 1, np.zeros(maxL + 1, np.int32)
    for l in range(1, maxL + 1):
        if counts[l] >= params["min_size"]:
            out_remap[l] = nxt; nxt += 1
    return out_remap[merged]


# ======================================================================
# 6. Top-level pipeline
# ======================================================================
def segment(W, rf, rgb, params, verbose=True):
    """RGB image [3,H,W] (raw) -> integer label mask [H,W]."""
    log = print if verbose else (lambda *a, **k: None)
    norm = percentile_normalize(rgb)
    padded, (H, Wd) = pad_to(norm, 8)
    log(f"input {rgb.shape[2]}x{rgb.shape[1]} -> padded {padded.shape[2]}x{padded.shape[1]}")
    out = fcn_forward(W, rf, np.clip(padded, -2, 3))[:, :H, :Wd]
    labels = decode(W, out, params)
    log(f"{int(labels.max())} nuclei")
    return labels


DEFAULTS = {
    "min_size": 10,
    "mask_threshold": 0.53,
    "peak_distance": 5,
    "seed_threshold": 0.7,
    "overlap_threshold": 0.3,
    "window_size": 32,
    "cleanup_fragments": True,
}


# ======================================================================
# 7. CLI + self-validation
# ======================================================================
def average_precision(gt, pred, thr=0.5):
    gt_ids = [i for i in np.unique(gt) if i > 0]
    pr_ids = [i for i in np.unique(pred) if i > 0]
    if not gt_ids and not pr_ids:
        return 1.0, 0, 0, 0
    if not gt_ids or not pr_ids:
        return 0.0, 0, len(pr_ids), len(gt_ids)
    gmap = {v: i for i, v in enumerate(gt_ids)}
    pmap = {v: i for i, v in enumerate(pr_ids)}
    inter = np.zeros((len(gt_ids), len(pr_ids)), np.int64)
    m = (gt > 0) & (pred > 0)
    for g, p in zip(gt[m], pred[m]):
        inter[gmap[g], pmap[p]] += 1
    ga = np.array([np.sum(gt == v) for v in gt_ids])
    pa = np.array([np.sum(pred == v) for v in pr_ids])
    iou = inter / np.maximum(ga[:, None] + pa[None, :] - inter, 1)
    tp, used = 0, set()
    for gi in np.argsort(-iou.max(axis=1)):
        pj = int(np.argmax(iou[gi]))
        if iou[gi, pj] > thr and pj not in used:
            tp += 1; used.add(pj)
    fp, fn = len(pr_ids) - tp, len(gt_ids) - tp
    return tp / (tp + fp + fn), tp, fp, fn


def main():
    ap = argparse.ArgumentParser(description="InstanSeg brightfield_nuclei in pure NumPy.")
    ap.add_argument("image")
    ap.add_argument("--weights", default=os.path.join(HERE, "instanseg_bf_weights.bin"))
    ap.add_argument("--manifest", default=os.path.join(HERE, "instanseg_bf_manifest.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        sys.exit(f"missing {args.weights} — run export_instanseg.py first (needs torch).")
    W, meta = load_weights(args.weights, args.manifest)
    rf = meta["relu"]
    print(f"loaded {len(meta['tensors'])} weight tensors")

    rgb = load_rgb(args.image)
    labels = segment(W, rf, rgb, DEFAULTS)
    print(f"==> {int(labels.max())} nuclei")

    if args.validate:
        name = os.path.splitext(os.path.basename(args.image))[0]
        m = json.load(open(os.path.join(HERE, "refdata", f"is_{name}.meta.json")))
        ref = np.load(os.path.join(HERE, "refdata", f"is_{name}.labels.npy"))
        norm = percentile_normalize(rgb)
        padded, (H, Wd) = pad_to(norm, 8)
        out = fcn_forward(W, rf, np.clip(padded, -2, 3))[:, :H, :Wd]
        rout = np.fromfile(os.path.join(HERE, "refdata", f"is_{name}.output.bin"), "<f4").reshape(5, m["H"], m["W"])
        print(f"  forward: max|Δ|={np.abs(out - rout).max():.1e}")
        apv, tp, fp, fn = average_precision(ref, labels)
        print(f"  masks: {int(labels.max())} vs ref {int(ref.max())}  "
              f"AP@0.5={apv:.3f} (tp{tp}/fp{fp}/fn{fn})")

    if args.out:
        np.save(args.out, labels)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
