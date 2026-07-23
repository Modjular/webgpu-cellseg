"""
mini_stardist.py — StarDist (2D_versatile_fluo) nucleus segmentation in ~one
file of NumPy.

In the spirit of nanoGPT/micrograd, and a sibling to `mini_cellpose.py`: no
TensorFlow, no CUDA, no `stardist` package at inference time — just NumPy — so
you can read the whole thing and see exactly what StarDist does.  It loads the
pretrained `2D_versatile_fluo` weights (exported once by `export_stardist.py`,
the only step that needs Keras) and runs the full pipeline end to end.

StarDist in three ideas
-----------------------
1. A U-Net looks at the image and predicts, at every pixel of a coarse output
   grid, (a) an *object probability* and (b) `n_rays=32` *radial distances* — how
   far it is from that pixel to the object boundary along 32 rays fanning out at
   equal angles.  Probability + 32 distances = one candidate *star-convex polygon*
   centered on that pixel.
2. Every pixel above the probability threshold proposes a polygon, so an object is
   proposed hundreds of times by its neighbouring pixels.  **Non-maximum
   suppression** keeps the highest-probability polygon and throws away every other
   polygon that overlaps it too much — collapsing each cloud of proposals to one.
3. Rasterize the surviving polygons into a label image, highest probability drawn
   last so it wins overlaps.

That's the whole method: no flow field, no iterative dynamics (contrast
`mini_cellpose.py`); the shape is read straight off the 32 distance channels and
the only post-processing is greedy NMS between polygons.

The network (a U-Net)
---------------------
`unet_n_depth=3`, `n_filter_base=32`, 3x3 convs, ReLU, **no BatchNorm**.  An
initial conv+conv+maxpool block downsamples to the `grid=(2,2)` output
resolution, then a standard depth-3 U-Net (encoder 32/64/128, bottleneck 256,
decoder with concat skips) feeds a shared `features` conv and two 1x1 heads:
`prob` (sigmoid) and `dist` (32 linear channels).  See `export_stardist.py` for
the exact layer graph; the port mirrors it in `unet_forward`.

Usage
-----
    python mini_stardist.py images/cellpose_image_020.png
    python mini_stardist.py images/cellpose_image_020.png --out labels.npy

Weights (`stardist_fluo_weights.bin` + `stardist_fluo_manifest.json`) must sit
next to this file; regenerate with `.venv-stardist/bin/python export_stardist.py`.
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
    meta = json.load(open(manifest_path))
    blob = np.fromfile(bin_path, dtype="<f4")
    W = {n: blob[t["offset"]:t["offset"] + t["length"]].reshape(t["shape"])
         for n, t in meta["tensors"].items()}
    return W, meta


# ======================================================================
# 1. NumPy neural-net primitives (tensors are [C,H,W], no batch dim)
# ======================================================================
def relu(x):
    return np.maximum(x, 0.0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def conv2d(x, w, b):
    """'same'-padded conv.  x:[Cin,H,W] w:[Cout,Cin,K,K] b:[Cout].

    The textbook definition: a weighted sum over the KxK kernel taps of the input
    shifted under each tap — small memory, reads like the math (same conv as
    mini_cellpose)."""
    Cin, H, Wd = x.shape
    Cout, _, K, _ = w.shape
    p = K // 2
    xp = np.pad(x, ((0, 0), (p, p), (p, p)))
    out = np.zeros((Cout, H, Wd), np.float32)
    for ky in range(K):
        for kx in range(K):
            patch = xp[:, ky:ky + H, kx:kx + Wd]
            out += np.tensordot(w[:, :, ky, kx], patch, axes=([1], [0]))
    return out + b[:, None, None]


def maxpool2(x):
    """2x2 max pool, stride 2 (Keras MaxPooling2D, 'valid')."""
    C, H, Wd = x.shape
    x = x[:, :H // 2 * 2, :Wd // 2 * 2]
    return x.reshape(C, H // 2, 2, Wd // 2, 2).max(axis=(2, 4))


def upsample2(x):
    """Nearest-neighbor 2x upsample (Keras UpSampling2D, 'nearest')."""
    return np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)


# ======================================================================
# 2. U-Net forward pass (mirrors export_stardist.unet_forward)
# ======================================================================
def unet_forward(W, x):
    """x:[1,H,W] -> (prob[gh,gw], dist[gh,gw,32]) at grid resolution (H/2,W/2)."""
    def cr(name, y):                                     # conv + ReLU
        return relu(conv2d(y, W[name + ".w"], W[name + ".b"]))
    # initial block downsamples to the (2,2) grid, then a depth-3 U-Net
    c1 = cr("conv2d_1", cr("conv2d", x))
    d0 = cr("down_level_0_no_1", cr("down_level_0_no_0", maxpool2(c1)))
    d1 = cr("down_level_1_no_1", cr("down_level_1_no_0", maxpool2(d0)))
    d2 = cr("down_level_2_no_1", cr("down_level_2_no_0", maxpool2(d1)))
    m = cr("middle_2", cr("middle_0", maxpool2(d2)))
    # decoder: upsample, concat encoder skip (upsampled first, then skip), conv x2
    u = np.concatenate([upsample2(m), d2], axis=0)
    u = cr("up_level_2_no_2", cr("up_level_2_no_0", u))
    u = np.concatenate([upsample2(u), d1], axis=0)
    u = cr("up_level_1_no_2", cr("up_level_1_no_0", u))
    u = np.concatenate([upsample2(u), d0], axis=0)
    u = cr("up_level_0_no_2", cr("up_level_0_no_0", u))
    f = cr("features", u)
    prob = sigmoid(conv2d(f, W["prob.w"], W["prob.b"]))[0]         # [gh,gw]
    dist = conv2d(f, W["dist.w"], W["dist.b"]).transpose(1, 2, 0)  # [gh,gw,32]
    return prob, dist


# ======================================================================
# 3. Preprocessing
# ======================================================================
def normalize99(x, lower=1, upper=99.8):
    """csbdeep/StarDist default: percentile contrast-stretch to [pmin,pmax]."""
    p1, p99 = np.percentile(x, lower), np.percentile(x, upper)
    return ((x - p1) / (p99 - p1 + 1e-20)).astype(np.float32)


def pad_to(x, div=16):
    """Reflect-pad [H,W] so both dims are multiples of div (4 pools => /16)."""
    H, W = x.shape
    Hp, Wp = -(-H // div) * div, -(-W // div) * div
    out = np.zeros((Hp, Wp), np.float32)
    out[:H, :W] = x
    if Hp > H:
        out[H:Hp, :W] = x[H - 1::-1][:Hp - H]
    if Wp > W:
        out[:, W:Wp] = out[:, W - 1::-1][:, :Wp - W]
    return out, (H, W)


# ======================================================================
# 4. Geometry — distances -> polygon coordinates
# ======================================================================
def ray_angles(n_rays=32):
    """The 32 ray directions, evenly spaced over the full circle (StarDist's
    convention: linspace(0, 2*pi, n_rays, endpoint=False))."""
    return np.linspace(0, 2 * np.pi, n_rays, endpoint=False)


def dist_to_coord(dist, points):
    """(n,32) distances + (n,2) center points (y,x) -> (n,2,32) polygon vertices.

    Vertex k of polygon i sits `dist[i,k]` away from center `points[i]` along ray
    angle phi_k:  y = cy + d*sin(phi),  x = cx + d*cos(phi)  (StarDist's mapping)."""
    phis = ray_angles(dist.shape[1])
    coord = dist[:, None, :] * np.array([np.sin(phis), np.cos(phis)])   # (n,2,32)
    coord += np.asarray(points)[..., None]
    return coord


def polygon_pixels(ys, xs, H, W):
    """Rasterize one polygon (vertex rows `ys`, cols `xs`) to the integer pixels
    inside it, clipped to [0,H)x[0,W).  Even-odd scanline fill sampling pixel
    centres — the same inside/outside rule skimage.draw.polygon uses, so the
    port's masks match StarDist's rendered labels."""
    n = len(ys)
    y0 = max(0, int(np.ceil(ys.min())))
    y1 = min(H - 1, int(np.floor(ys.max())))
    rr, cc = [], []
    for y in range(y0, y1 + 1):
        xints = []
        for k in range(n):
            ya, xa = ys[k], xs[k]
            yb, xb = ys[(k + 1) % n], xs[(k + 1) % n]
            # edge crosses this scanline if y is in [min,max) of its endpoints
            if (ya <= y < yb) or (yb <= y < ya):
                xints.append(xa + (y - ya) * (xb - xa) / (yb - ya))
        xints.sort()
        for a in range(0, len(xints) - 1, 2):
            xl = max(0, int(np.ceil(xints[a])))
            xhi = min(W - 1, int(np.floor(xints[a + 1])))
            for x in range(xl, xhi + 1):
                rr.append(y); cc.append(x)
    return np.asarray(rr, np.int64), np.asarray(cc, np.int64)


# ======================================================================
# 5. Non-maximum suppression of star-convex polygons
# ======================================================================
def nms_polygons(points, dist, scores, H, W, nms_thresh):
    """Greedy NMS over polygons, StarDist's rule: walking candidates from highest
    score down, a polygon is *suppressed* by an already-kept one if their overlap
    Ainter / min(A1,A2) exceeds `nms_thresh`.  (Note it's min-area overlap, not
    union IoU.)  Returns the kept indices (into the score-sorted arrays).

    Rasterizes each candidate to get exact overlaps.  A candidate is only compared
    against kept polygons whose centre is within the sum of the two bounding radii
    (their bounding circles touch) — the cheap spatial prune that keeps this from
    being O(n^2) full-mask intersections."""
    order = np.argsort(scores)[::-1]
    radii = dist.max(axis=1)                                    # bounding radius per poly
    coords = dist_to_coord(dist, points)                       # (n,2,32)

    kept = []
    kc = np.empty((0, 2))                                       # kept centres (y,x)
    kr = np.empty((0,))                                         # kept radii
    kmask = []                                                  # kept pixel sets (frozenset)
    karea = []                                                  # kept areas

    for i in order:
        cy, cx = points[i]
        ri = radii[i]
        if len(kept):
            d = np.hypot(kc[:, 0] - cy, kc[:, 1] - cx)
            near = np.nonzero(d < ri + kr)[0]                  # bounding circles overlap
        else:
            near = np.empty(0, np.int64)
        rr, ccx = polygon_pixels(coords[i, 0], coords[i, 1], H, W)
        area_i = rr.size
        if area_i == 0:
            continue
        pix_i = set(zip(rr.tolist(), ccx.tolist()))
        suppressed = False
        for j in near:                                         # j indexes kept[]
            inter = len(pix_i & kmask[j])
            if inter and inter / min(area_i, karea[j]) > nms_thresh:
                suppressed = True
                break
        if not suppressed:
            kept.append(i)
            kc = np.vstack([kc, [cy, cx]])
            kr = np.append(kr, ri)
            kmask.append(pix_i)
            karea.append(area_i)
    return np.asarray(kept, np.int64)


def polygons_to_label(points, dist, scores, kept, H, W):
    """Render the kept polygons to a label image, lowest score first so the
    highest-probability polygon is painted last and wins any overlap (matching
    StarDist's polygons_to_label ordering)."""
    coords = dist_to_coord(dist[kept], points[kept])
    order = np.argsort(scores[kept], kind="stable")            # ascending: big drawn last
    lbl = np.zeros((H, W), np.int32)
    for lab, i in enumerate(order, start=1):
        rr, cc = polygon_pixels(coords[i, 0], coords[i, 1], H, W)
        lbl[rr, cc] = lab
    return lbl


# ======================================================================
# 6. Top-level pipeline
# ======================================================================
def segment(W, gray, prob_thresh=0.4791, nms_thresh=0.3, grid=(2, 2),
            border=2, verbose=True):
    """Full StarDist: grayscale image -> integer label mask [H,W]."""
    log = print if verbose else (lambda *a, **k: None)
    H, Wd = gray.shape

    # --- preprocess: normalize + reflect-pad to /16 ---
    norm = normalize99(gray)
    padded, _ = pad_to(norm, 16)
    log(f"input {Wd}x{H} -> padded {padded.shape[1]}x{padded.shape[0]}")

    # --- network forward -> prob + 32 distances on the (2,2) grid ---
    prob, dist = unet_forward(W, padded[None])                 # prob[gh,gw], dist[gh,gw,32]
    gh, gw = prob.shape
    log(f"grid {gw}x{gh}  prob[max]={prob.max():.3f}")

    # --- candidate points: prob above threshold, excluding a `border`-px grid frame ---
    mask = prob > prob_thresh
    if border > 0:
        frame = np.zeros_like(mask)
        frame[border:-border, border:-border] = True
        mask &= frame
    gy, gx = np.nonzero(mask)
    if gy.size == 0:
        log("no candidate polygons")
        return np.zeros((H, Wd), np.int32)
    scores = prob[gy, gx]
    dists = dist[gy, gx]                                        # (n,32)
    points = np.stack([gy * grid[0], gx * grid[1]], axis=1).astype(np.float64)  # -> full res
    log(f"{gy.size} candidate polygons above prob_thresh={prob_thresh}")

    # --- NMS, then rasterize survivors to a label image (at full padded res) ---
    kept = nms_polygons(points, dists, scores, padded.shape[0], padded.shape[1], nms_thresh)
    log(f"{kept.size} polygons after NMS")
    labels = polygons_to_label(points, dists, scores, kept, padded.shape[0], padded.shape[1])
    return labels[:H, :Wd]


# ======================================================================
# 7. CLI + optional self-validation against refdata/
# ======================================================================
def load_gray(path):
    from PIL import Image
    a = np.asarray(Image.open(path)).astype(np.float32)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    return a


def average_precision(gt, pred, thr=0.5):
    """AP@thr = TP / (TP + FP + FN) with 1-1 IoU>thr matching (cellpose's metric)."""
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
    union = ga[:, None] + pa[None, :] - inter
    iou = inter / np.maximum(union, 1)
    tp = 0
    used_p = set()
    for gi in np.argsort(-iou.max(axis=1)):
        pj = np.argmax(iou[gi])
        if iou[gi, pj] > thr and pj not in used_p:
            tp += 1
            used_p.add(pj)
    fp = len(pr_ids) - tp
    fn = len(gt_ids) - tp
    return tp / (tp + fp + fn), tp, fp, fn


def main():
    ap = argparse.ArgumentParser(description="StarDist 2D_versatile_fluo in pure NumPy.")
    ap.add_argument("image", help="path to an image (png/tif/...)")
    ap.add_argument("--prob_thresh", type=float, default=0.4791)
    ap.add_argument("--nms_thresh", type=float, default=0.3)
    ap.add_argument("--weights", default=os.path.join(HERE, "stardist_fluo_weights.bin"))
    ap.add_argument("--manifest", default=os.path.join(HERE, "stardist_fluo_manifest.json"))
    ap.add_argument("--out", default=None, help="save labels as .npy")
    ap.add_argument("--validate", action="store_true",
                    help="compare forward + masks against refdata/sd_<name>.*")
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        sys.exit(f"missing {args.weights} — run export_stardist.py first (needs Keras).")
    W, meta = load_weights(args.weights, args.manifest)
    print(f"loaded {len(meta['tensors'])} weight tensors")

    gray = load_gray(args.image)
    labels = segment(W, gray, prob_thresh=args.prob_thresh, nms_thresh=args.nms_thresh)
    print(f"==> {int(labels.max())} nuclei")

    if args.validate:
        name = os.path.splitext(os.path.basename(args.image))[0]
        m = json.load(open(os.path.join(HERE, "refdata", f"sd_{name}.meta.json")))
        ref = np.load(os.path.join(HERE, "refdata", f"sd_{name}.labels.npy"))
        # forward check against dumped prob/dist
        norm = normalize99(gray)
        padded, _ = pad_to(norm, 16)
        prob, dist = unet_forward(W, padded[None])
        rp = np.fromfile(os.path.join(HERE, "refdata", f"sd_{name}.prob.bin"), "<f4").reshape(m["gh"], m["gw"])
        rd = np.fromfile(os.path.join(HERE, "refdata", f"sd_{name}.dist.bin"), "<f4").reshape(m["gh"], m["gw"], m["n_rays"])
        print(f"  forward: max|Δprob|={np.abs(prob-rp).max():.1e}  max|Δdist|={np.abs(dist-rd).max():.1e}")
        apv, tp, fp, fn = average_precision(ref, labels, 0.5)
        print(f"  masks: {int(labels.max())} vs ref {int(ref.max())}  "
              f"AP@0.5={apv:.3f} (tp{tp}/fp{fp}/fn{fn})")

    if args.out:
        np.save(args.out, labels)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
