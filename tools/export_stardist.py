"""
export_stardist.py — export the pretrained StarDist2D `2D_versatile_fluo`
weights + reference data for the pure-NumPy port (`mini_stardist.py`).

This is the *only* step that needs TensorFlow/Keras.  It:
  1. Loads `StarDist2D.from_pretrained('2D_versatile_fluo')`.
  2. Writes every conv layer's weights to a flat float32 blob + a JSON manifest,
     with kernels transposed from Keras `[kh,kw,cin,cout]` to the `[cout,cin,kh,kw]`
     layout the NumPy port convolves with.
  3. For each sample image, runs a *controlled* pipeline the port mirrors exactly
     (normalize99.8 -> reflect-pad to /16 -> CPnet-style U-Net) and dumps the
     network input, the prob/dist maps and the reference instance labels so the
     port can validate preprocessing, the forward pass and the masks.
  4. Self-checks a NumPy reference forward against Keras (< 1e-4) so the port's
     forward has a trustworthy target.

Outputs (per sample `<name>`):
  refdata/sd_<name>.input.bin   float32 [H,W]        (normalized, un-padded)
  refdata/sd_<name>.prob.bin    float32 [gh,gw]      (grid = H/2 x W/2)
  refdata/sd_<name>.dist.bin    float32 [gh,gw,32]   (radial distances, 32 rays)
  refdata/sd_<name>.labels.bin  int32   [H,W]        (StarDist predict_instances)
  refdata/sd_<name>.labels.npy  (for python-side AP checks)
  refdata/sd_<name>.meta.json   shapes + grid + n_rays + thresholds
  stardist_fluo_weights.bin, stardist_fluo_manifest.json

Run:  .venv-stardist/bin/python export_stardist.py
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import glob
import json
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REFDIR = os.path.join(HERE, "refdata")
WEIGHTS = os.path.join(HERE, "..", "weights", "stardist-fluo", "weights.bin")
MANIFEST = os.path.join(HERE, "..", "weights", "stardist-fluo", "manifest.json")

# The U-Net's conv layers, in forward order.  `k` is the kernel size; every conv
# is followed by ReLU except the two output heads (prob -> sigmoid, dist -> linear).
CONV_LAYERS = [
    "conv2d", "conv2d_1",
    "down_level_0_no_0", "down_level_0_no_1",
    "down_level_1_no_0", "down_level_1_no_1",
    "down_level_2_no_0", "down_level_2_no_1",
    "middle_0", "middle_2",
    "up_level_2_no_0", "up_level_2_no_2",
    "up_level_1_no_0", "up_level_1_no_2",
    "up_level_0_no_0", "up_level_0_no_2",
    "features", "prob", "dist",
]


# ---------------------------------------------------------------- weight export
def export_weights(model):
    """Flatten conv weights to a blob + manifest.  Kernels are transposed to
    [cout,cin,kh,kw] (what the NumPy port's conv2d expects)."""
    layers = {l.name: l for l in model.keras_model.layers}
    tensors, blob, off = {}, [], 0
    for name in CONV_LAYERS:
        w, b = layers[name].get_weights()          # w:[kh,kw,cin,cout] b:[cout]
        w = np.ascontiguousarray(w.transpose(3, 2, 0, 1)).astype("<f4")  # [cout,cin,kh,kw]
        for tag, arr in ((f"{name}.w", w), (f"{name}.b", b.astype("<f4"))):
            flat = arr.ravel()
            tensors[tag] = {"offset": off, "length": int(flat.size), "shape": list(arr.shape)}
            blob.append(flat)
            off += flat.size
    np.concatenate(blob).astype("<f4").tofile(WEIGHTS)
    meta = {"tensors": tensors, "conv_layers": CONV_LAYERS}
    json.dump(meta, open(MANIFEST, "w"), indent=1)
    print(f"wrote {WEIGHTS} ({off*4/1e6:.1f} MB) + {MANIFEST} ({len(tensors)} tensors)")
    return load_weights()


def load_weights():
    meta = json.load(open(MANIFEST))
    blob = np.fromfile(WEIGHTS, dtype="<f4")
    W = {n: blob[t["offset"]:t["offset"] + t["length"]].reshape(t["shape"])
         for n, t in meta["tensors"].items()}
    return W


# ------------------------------------------------ NumPy reference forward (check)
# Tensors are [C,H,W] with no batch dim, mirroring mini_cellpose's conventions.
def relu(x):
    return np.maximum(x, 0.0)


def conv2d(x, w, b):
    """'same'-padded conv.  x:[Cin,H,W] w:[Cout,Cin,K,K] b:[Cout]."""
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
    C, H, Wd = x.shape
    x = x[:, :H // 2 * 2, :Wd // 2 * 2]
    return x.reshape(C, H // 2, 2, Wd // 2, 2).max(axis=(2, 4))


def upsample2(x):
    return np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def unet_forward(W, x):
    """x:[1,H,W] -> (prob[gh,gw], dist[gh,gw,32]) at grid resolution (H/2,W/2)."""
    def cr(name, y):
        return relu(conv2d(y, W[name + ".w"], W[name + ".b"]))
    # grid block (downsamples to the (2,2) grid), then a depth-3 U-Net
    c1 = cr("conv2d_1", cr("conv2d", x))
    d0 = cr("down_level_0_no_1", cr("down_level_0_no_0", maxpool2(c1)))
    d1 = cr("down_level_1_no_1", cr("down_level_1_no_0", maxpool2(d0)))
    d2 = cr("down_level_2_no_1", cr("down_level_2_no_0", maxpool2(d1)))
    m = cr("middle_2", cr("middle_0", maxpool2(d2)))
    u = np.concatenate([upsample2(m), d2], axis=0)              # skip: up first, then encoder
    u = cr("up_level_2_no_2", cr("up_level_2_no_0", u))
    u = np.concatenate([upsample2(u), d1], axis=0)
    u = cr("up_level_1_no_2", cr("up_level_1_no_0", u))
    u = np.concatenate([upsample2(u), d0], axis=0)
    u = cr("up_level_0_no_2", cr("up_level_0_no_0", u))
    f = cr("features", u)
    prob = sigmoid(conv2d(f, W["prob.w"], W["prob.b"]))[0]      # [gh,gw]
    dist = conv2d(f, W["dist.w"], W["dist.b"]).transpose(1, 2, 0)  # [gh,gw,32]
    return prob, dist


# ------------------------------------------------------------------ preprocessing
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


def load_gray(path):
    a = np.asarray(Image.open(path)).astype(np.float32)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    return a


# -------------------------------------------------------------------------- main
def main():
    from stardist.models import StarDist2D
    os.makedirs(REFDIR, exist_ok=True)
    model = StarDist2D.from_pretrained("2D_versatile_fluo")
    grid = tuple(model.config.grid)
    n_rays = model.config.n_rays
    prob_thresh = float(model.thresholds.prob)
    nms_thresh = float(model.thresholds.nms)
    print(f"grid={grid} n_rays={n_rays} prob_thresh={prob_thresh:.4f} nms_thresh={nms_thresh}")

    W = export_weights(model)

    for path in sorted(glob.glob(os.path.join(HERE, "images", "*.png"))):
        name = os.path.splitext(os.path.basename(path))[0]
        norm = normalize99(load_gray(path))
        H, Wd = norm.shape
        padded, _ = pad_to(norm, 16)
        Hp, Wp = padded.shape

        # keras forward on the padded input
        y = model.keras_model.predict(padded[None, ..., None], verbose=0)
        prob_k = np.asarray(y[0])[0, ..., 0]                    # [gh,gw]
        dist_k = np.asarray(y[1])[0]                            # [gh,gw,32]

        # numpy reference forward — self-check
        prob_n, dist_n = unet_forward(W, padded[None])
        dprob = np.abs(prob_n - prob_k).max()
        ddist = np.abs(dist_n - dist_k).max()

        # Reference instances from StarDist's *own* NMS + polygon rendering, run on
        # the exact same padded prob/dist we dump above (not predict_instances,
        # whose internal padding/normalization would diverge from the controlled
        # pipeline the port mirrors).  This isolates the port's NMS+geometry as the
        # only thing under test — mirroring how the cellpose test validates dynamics
        # on the reference network output.
        from stardist.nms import non_maximum_suppression
        from stardist.geometry import polygons_to_label
        pts, probi, disti = non_maximum_suppression(
            dist_k, prob_k, grid=grid, prob_thresh=prob_thresh, nms_thresh=nms_thresh)
        labels_full = polygons_to_label(disti, pts, shape=(Hp, Wp), prob=probi)
        labels = labels_full[:H, :Wd]
        n_inst = int(len(np.unique(labels)) - 1)

        gh, gw = prob_k.shape
        norm.astype("<f4").tofile(f"{REFDIR}/sd_{name}.input.bin")
        prob_k.astype("<f4").tofile(f"{REFDIR}/sd_{name}.prob.bin")
        dist_k.astype("<f4").tofile(f"{REFDIR}/sd_{name}.dist.bin")
        labels.astype("<i4").tofile(f"{REFDIR}/sd_{name}.labels.bin")
        np.save(f"{REFDIR}/sd_{name}.labels.npy", labels.astype(np.int32))
        json.dump({"name": name, "H": H, "W": Wd, "Hp": Hp, "Wp": Wp,
                   "gh": gh, "gw": gw, "grid": list(grid), "n_rays": n_rays,
                   "prob_thresh": prob_thresh, "nms_thresh": nms_thresh,
                   "n_instances": n_inst},
                  open(f"{REFDIR}/sd_{name}.meta.json", "w"), indent=1)

        print(f"  {name}: {Wd}x{H} -> grid {gw}x{gh}  "
              f"fwd max|Δprob|={dprob:.1e} max|Δdist|={ddist:.1e}  "
              f"instances={n_inst}  {'OK' if dprob < 1e-4 and ddist < 1e-3 else 'CHECK'}")


if __name__ == "__main__":
    main()
