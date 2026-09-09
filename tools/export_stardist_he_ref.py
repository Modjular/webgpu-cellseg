"""
export_stardist_he_ref.py — reference dumps for the StarDist2D `2D_versatile_he`
(H&E) checkpoint, so the WebGPU port is validated against the *original* model the
same way `2D_versatile_fluo` already is by export_stardist.py.

Unlike export_stardist_he.py (which only repacks the release `.h5` weights via h5py
and needs neither TensorFlow nor the `stardist` package), this DOES load the real
model — it is the only H&E step that needs TensorFlow/Keras + `stardist` — because a
trustworthy reference has to come from StarDist's own forward + NMS, not from our port.

It:
  1. Loads `StarDist2D.from_pretrained('2D_versatile_he')` (RGB, n_channel_in=3).
  2. Takes StarDist's shipped H&E example (`stardist.data.test_image_he_2d()`,
     the 300x500 histology RGB from the Cancer Imaging Archive) and centre-crops it
     to CROP x CROP (256) — small enough to run on CPU and to commit, per the
     "smallest sample per model" rule the rest of tests/refdata follows.
  3. Runs the *controlled* pipeline the port mirrors exactly (per-channel normalize99.8
     -> reflect-pad to /16 -> RGB CPnet-style U-Net) and dumps the network input, the
     prob/dist maps and the reference instance labels.
  4. Self-checks a NumPy reference forward (using the committed weights/stardist-he
     blob, whose fluo-style tags export_stardist_he.py already produced) against Keras
     (< 1e-4 / < 1e-3), which also confirms that repacked blob matches the real model.

Reuses export_stardist.py's preprocessing + NumPy reference forward — the only H&E
difference is the 3-channel input, which its conv2d already handles.

Outputs:
  refdata/stardist/sd_he_histo.input.bin   float32 [3,H,W]   (normalized, un-padded)
  refdata/stardist/sd_he_histo.prob.bin    float32 [gh,gw]
  refdata/stardist/sd_he_histo.dist.bin    float32 [gh,gw,32]
  refdata/stardist/sd_he_histo.labels.bin  int32   [H,W]
  refdata/stardist/sd_he_histo.labels.npy
  refdata/stardist/sd_he_histo.meta.json

Run:  tools/.venv-stardist/bin/python tools/export_stardist_he.py   # weights first
      tools/.venv-stardist/bin/python tools/export_stardist_he_ref.py
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")     # CPU-only; the crop is small
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")   # deterministic reduction order
import json

import numpy as np

# Preprocessing + the NumPy reference forward are shared with the fluo exporter; the
# only H&E-specific thing is a 3-channel input, which conv2d/unet_forward handle as-is.
from export_stardist import normalize99, pad_to, unet_forward

HERE = os.path.dirname(os.path.abspath(__file__))
# Write straight into the harness's committed refdata tree (tests/refdata/stardist),
# not tools/refdata — export_stardist.py's flat tools/refdata output predates the
# tests/ layout (see tests/README.md's "written against the original dev layout").
REFDIR = os.path.join(HERE, "..", "tests", "refdata", "stardist")
HE_DIR = os.path.join(HERE, "..", "weights", "stardist-he")
WEIGHTS = os.path.join(HE_DIR, "weights.bin")
MANIFEST = os.path.join(HE_DIR, "manifest.json")

NAME = "he_histo"
CROP = 256   # divisible by 16, so reflect-pad is a no-op; keeps the committed dump small


def load_he_weights():
    """The committed stardist-he blob, keyed by export_stardist_he.py's fluo-style tags
    (conv2d, conv2d_1, ...) — the same names unet_forward expects."""
    meta = json.load(open(MANIFEST))
    blob = np.fromfile(WEIGHTS, dtype="<f4")
    return {n: blob[t["offset"]:t["offset"] + t["length"]].reshape(t["shape"])
            for n, t in meta["tensors"].items()}


def centre_crop(img, size):
    H, W = img.shape[:2]
    if H < size or W < size:
        raise SystemExit(f"H&E example is {W}x{H}, smaller than crop {size}")
    y0, x0 = (H - size) // 2, (W - size) // 2
    return img[y0:y0 + size, x0:x0 + size]


def main():
    if not os.path.exists(MANIFEST):
        raise SystemExit("weights/stardist-he not found — run export_stardist_he.py first")

    from stardist.models import StarDist2D
    from stardist.data import test_image_he_2d
    from stardist.nms import non_maximum_suppression
    from stardist.geometry import polygons_to_label

    os.makedirs(REFDIR, exist_ok=True)
    model = StarDist2D.from_pretrained("2D_versatile_he")
    grid = tuple(model.config.grid)
    n_rays = model.config.n_rays
    prob_thresh = float(model.thresholds.prob)
    nms_thresh = float(model.thresholds.nms)
    print(f"grid={grid} n_rays={n_rays} prob_thresh={prob_thresh:.4f} nms_thresh={nms_thresh}")

    W = load_he_weights()
    n_channel = int(model.config.n_channel_in)
    assert n_channel == 3, f"expected RGB H&E model, got n_channel_in={n_channel}"

    img = np.asarray(test_image_he_2d()).astype(np.float32)[..., :3]   # [H0,W0,3] RGB
    img = centre_crop(img, CROP)                                       # [H,W,3]
    H, Wd = img.shape[:2]

    # Per-channel percentile normalize (StarDist's axis_norm=(0,1)) — the same formula
    # src/stardist.js normalize99() applies, so the dumped input matches what the JS
    # harness reconstructs when it re-pads these planes.
    normed = np.stack([normalize99(img[..., c]) for c in range(3)], axis=0)   # [3,H,W]

    # reflect-pad each channel to /16 (a no-op at 256, but kept for parity with fluo)
    padded = np.stack([pad_to(normed[c], 16)[0] for c in range(3)], axis=0)   # [3,Hp,Wp]
    Hp, Wp = padded.shape[1:]

    # keras forward on [1,Hp,Wp,3]
    y = model.keras_model.predict(np.moveaxis(padded, 0, -1)[None], verbose=0)
    prob_k = np.asarray(y[0])[0, ..., 0]        # [gh,gw]
    dist_k = np.asarray(y[1])[0]                # [gh,gw,32]

    # NumPy reference forward on [3,Hp,Wp] — self-check (also validates the repacked blob)
    prob_n, dist_n = unet_forward(W, padded)
    dprob = np.abs(prob_n - prob_k).max()
    ddist = np.abs(dist_n - dist_k).max()

    # Reference instances from StarDist's own NMS + polygon rendering on that same
    # prob/dist (not predict_instances, whose internal norm/pad would diverge from the
    # controlled pipeline) — isolates the port's NMS+geometry, exactly as the fluo path.
    pts, probi, disti = non_maximum_suppression(
        dist_k, prob_k, grid=grid, prob_thresh=prob_thresh, nms_thresh=nms_thresh)
    labels_full = polygons_to_label(disti, pts, shape=(Hp, Wp), prob=probi)
    labels = labels_full[:H, :Wd]
    n_inst = int(len(np.unique(labels)) - 1)

    gh, gw = prob_k.shape
    normed.astype("<f4").tofile(f"{REFDIR}/sd_{NAME}.input.bin")     # [3,H,W]
    prob_k.astype("<f4").tofile(f"{REFDIR}/sd_{NAME}.prob.bin")
    dist_k.astype("<f4").tofile(f"{REFDIR}/sd_{NAME}.dist.bin")
    labels.astype("<i4").tofile(f"{REFDIR}/sd_{NAME}.labels.bin")
    np.save(f"{REFDIR}/sd_{NAME}.labels.npy", labels.astype(np.int32))
    json.dump({"name": NAME, "H": H, "W": Wd, "Hp": Hp, "Wp": Wp,
               "gh": gh, "gw": gw, "grid": list(grid), "n_rays": n_rays,
               "channels": n_channel,
               "prob_thresh": prob_thresh, "nms_thresh": nms_thresh,
               "n_instances": n_inst},
              open(f"{REFDIR}/sd_{NAME}.meta.json", "w"), indent=1)

    ok = dprob < 1e-4 and ddist < 1e-3
    print(f"  {NAME}: {Wd}x{H} RGB -> grid {gw}x{gh}  "
          f"fwd max|Δprob|={dprob:.1e} max|Δdist|={ddist:.1e}  "
          f"instances={n_inst}  {'OK' if ok else 'CHECK'}")
    if not ok:
        raise SystemExit("reference forward self-check exceeded tolerance")


if __name__ == "__main__":
    main()
