"""
export_instanseg.py — export the pretrained InstanSeg `brightfield_nuclei` model
(network weights + the tiny pixel-classifier MLP) + reference data for the pure-
NumPy port (`mini_instanseg.py`).

The only step that needs PyTorch/InstanSeg. It:
  1. Loads InstanSeg("brightfield_nuclei") — an InstanSeg_UNet (`fcn`) that emits,
     per pixel, 2 coordinate fields + 2 sigma channels + 1 seed map, plus a 4→5→5→1
     `pixel_classifier` MLP used in the instance decode.
  2. Folds every conv+BatchNorm (the net is post-activation conv→BN→ReLU, so BN
     folds into the conv that precedes it) and writes a flat weight blob + manifest.
  3. Dumps, per sample: the percentile-normalized input, the [5,H,W] network output,
     and the reference instance labels (InstanSeg's own eval) so the port can
     validate preprocessing, the forward pass and the masks.
  4. Self-checks a NumPy reference forward against the torch fcn (< 1e-3).

Outputs (per sample `<name>`):
  refdata/is_<name>.input.bin   float32 [3,H,W]   (percentile-normalized)
  refdata/is_<name>.output.bin  float32 [5,H,W]   (network output, cropped)
  refdata/is_<name>.labels.bin  int32   [H,W]     (InstanSeg reference instances)
  refdata/is_<name>.labels.npy
  refdata/is_<name>.meta.json   shapes + decode params
  instanseg_bf_weights.bin, instanseg_bf_manifest.json

Run:  .venv-instanseg/bin/python export_instanseg.py
"""
import os
os.environ.setdefault("INSTANSEG_NO_TQDM", "1")
import glob
import json
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REFDIR = os.path.join(HERE, "refdata")
WEIGHTS = os.path.join(HERE, "..", "weights", "instanseg-brightfield", "weights.bin")
MANIFEST = os.path.join(HERE, "..", "weights", "instanseg-brightfield", "manifest.json")
BN_EPS = 1e-5


# ---------------------------------------------------------------- weight export
def fold(sd, name, relu):
    """Fold a conv_norm_act unit (conv `.0` + BatchNorm `.1`) into one conv.
    Post-activation: y = act(BN(conv(x))), so BN folds into this conv's w,b."""
    w = sd[f"{name}.0.weight"].numpy().astype(np.float64)     # [Cout,Cin,K,K]
    b = sd[f"{name}.0.bias"].numpy().astype(np.float64)
    g = sd[f"{name}.1.weight"].numpy().astype(np.float64)     # BN gamma
    beta = sd[f"{name}.1.bias"].numpy().astype(np.float64)
    mean = sd[f"{name}.1.running_mean"].numpy().astype(np.float64)
    var = sd[f"{name}.1.running_var"].numpy().astype(np.float64)
    scale = g / np.sqrt(var + BN_EPS)
    wf = w * scale[:, None, None, None]
    bf = (b - mean) * scale + beta
    return wf.astype("<f4"), bf.astype("<f4"), relu


def unit_names():
    """Every conv_norm_act unit in the fcn, mapped to a short port-side name and a
    relu flag (all True except the three output heads)."""
    out = {}
    for n in range(4):                                        # encoder blocks
        for m in range(5):
            out[f"enc{n}.c{m}"] = (f"encoder.{n}.conv{m}", True)
    for n in range(3):                                        # decoder blocks
        out[f"dec{n}.c0"] = (f"decoders.0.decoder.{n}.conv0", True)
        out[f"dec{n}.skip"] = (f"decoders.0.decoder.{n}.conv_skip", True)
        for m in range(1, 5):
            out[f"dec{n}.c{m}"] = (f"decoders.0.decoder.{n}.conv{m}", True)
    for m, relu in ((0, False), (1, False), (2, False)):     # output heads (no act)
        out[f"head{m}"] = (f"decoders.0.final_block.{m}", relu)
    return out


def export_weights(net):
    sd = {k: v.cpu() for k, v in net.fcn.state_dict().items()}
    tensors, blob, off = {}, [], 0

    def add(tag, arr):
        nonlocal off
        flat = np.ascontiguousarray(arr).ravel().astype("<f4")
        tensors[tag] = {"offset": off, "length": int(flat.size), "shape": list(arr.shape)}
        blob.append(flat); off += flat.size

    relu_flags = {}
    for tag, (name, relu) in unit_names().items():
        wf, bf, r = fold(sd, name, relu)
        add(f"{tag}.w", wf); add(f"{tag}.b", bf); relu_flags[tag] = r
    # pixel-classifier MLP (fc1 4->5, fc2 5->5, fc3 5->1) — plain linear layers
    pc = {k: v.cpu() for k, v in net.pixel_classifier.state_dict().items()}
    for k in ("fc1", "fc2", "fc3"):
        add(f"pc.{k}.w", pc[f"{k}.weight"].numpy().astype("<f4"))
        add(f"pc.{k}.b", pc[f"{k}.bias"].numpy().astype("<f4"))

    np.concatenate(blob).astype("<f4").tofile(WEIGHTS)
    json.dump({"tensors": tensors, "relu": relu_flags}, open(MANIFEST, "w"))
    print(f"wrote {WEIGHTS} ({off*4/1e6:.1f} MB) + {MANIFEST} ({len(tensors)} tensors)")
    return load_weights()


def load_weights():
    meta = json.load(open(MANIFEST))
    blob = np.fromfile(WEIGHTS, dtype="<f4")
    W = {n: blob[t["offset"]:t["offset"] + t["length"]].reshape(t["shape"])
         for n, t in meta["tensors"].items()}
    return W, meta


# ------------------------------------------------ NumPy reference forward (check)
def relu(x):
    return np.maximum(x, 0.0)


def conv(x, w, b):
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


def cna(W, tag, x, relu_flags):
    y = conv(x, W[f"{tag}.w"], W[f"{tag}.b"])
    return relu(y) if relu_flags[tag] else y


def enc_block(W, n, x, rf, pool):
    if pool:
        x = maxpool2(x)
    proj = cna(W, f"enc{n}.c0", x, rf)
    x = cna(W, f"enc{n}.c1", x, rf)
    x = proj + cna(W, f"enc{n}.c2", x, rf)
    x = x + cna(W, f"enc{n}.c4", cna(W, f"enc{n}.c3", x, rf), rf)
    return x


def dec_block(W, n, x, skip, rf):
    x = up2(x)
    proj = cna(W, f"dec{n}.c0", x, rf)
    x = cna(W, f"dec{n}.c1", x, rf)
    x = proj + cna(W, f"dec{n}.c2", x + cna(W, f"dec{n}.skip", skip, rf), rf)
    x = x + cna(W, f"dec{n}.c4", cna(W, f"dec{n}.c3", x, rf), rf)
    return x


def fcn_forward(W, rf, x):
    """x:[3,H,W] -> [5,H,W] (2 coord fields + 2 sigma + 1 seed)."""
    e0 = enc_block(W, 0, x, rf, pool=False)
    e1 = enc_block(W, 1, e0, rf, pool=True)
    e2 = enc_block(W, 2, e1, rf, pool=True)
    e3 = enc_block(W, 3, e2, rf, pool=True)
    d = dec_block(W, 0, e3, e2, rf)
    d = dec_block(W, 1, d, e1, rf)
    d = dec_block(W, 2, d, e0, rf)
    heads = [cna(W, f"head{m}", d, rf) for m in range(3)]      # 32->2, 32->2, 32->1
    return np.concatenate(heads, axis=0)


# ------------------------------------------------------------- preprocessing
def percentile_normalize(img, p=0.1, eps=1e-3):
    """Per-channel (x - p0.1) / max(eps, p99.9 - p0.1) — InstanSeg's normalization."""
    img = img.astype(np.float32).copy()
    for c in range(img.shape[0]):
        lo, hi = np.percentile(img[c], [p, 100 - p])
        img[c] = (img[c] - lo) / max(eps, hi - lo)
    return img


def pad_to(x, div=8):
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
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
    return np.moveaxis(a, 2, 0)                                 # [3,H,W]


# -------------------------------------------------------------------------- main
def main():
    import torch
    from instanseg import InstanSeg
    os.makedirs(REFDIR, exist_ok=True)
    model = InstanSeg("brightfield_nuclei", verbosity=0, device="cpu")
    model.instanseg = model.instanseg.to("cpu")
    net = model.instanseg
    dflt = {k: getattr(net, "default_" + k) for k in
            ["min_size", "mask_threshold", "peak_distance", "seed_threshold",
             "overlap_threshold", "fg_threshold", "window_size", "cleanup_fragments"]}
    print("defaults:", dflt)

    W, meta = export_weights(net)
    rf = meta["relu"]

    for path in sorted(glob.glob(os.path.join(HERE, "images", "*.png"))):
        name = os.path.splitext(os.path.basename(path))[0]
        norm = percentile_normalize(load_rgb(path))            # [3,H,W]
        C, H, Wd = norm.shape
        padded, _ = pad_to(norm, 8)
        Hp, Wp = padded.shape[1:]

        with torch.no_grad():
            t = torch.from_numpy(padded)[None].clamp(-2, 3)
            out_t = net.fcn(t)[0, :, :H, :Wd].numpy()          # [5,H,W]
        out_n = fcn_forward(W, rf, np.clip(padded, -2, 3))[:, :H, :Wd]
        dd = np.abs(out_t - out_n).max()

        # InstanSeg's own instance labels (reference for the decode)
        lab = model.eval_small_image(torch.from_numpy(norm), normalise=False,
                                     return_image_tensor=False)[0]
        labels = np.asarray(lab.squeeze()).astype(np.int32)
        n_inst = int(labels.max())

        norm.astype("<f4").tofile(f"{REFDIR}/is_{name}.input.bin")
        out_t.astype("<f4").tofile(f"{REFDIR}/is_{name}.output.bin")
        labels.astype("<i4").tofile(f"{REFDIR}/is_{name}.labels.bin")
        np.save(f"{REFDIR}/is_{name}.labels.npy", labels)
        json.dump({"name": name, "H": H, "W": Wd, "Hp": Hp, "Wp": Wp,
                   "n_instances": n_inst, **{k: (float(v) if isinstance(v, float) else int(v))
                                             for k, v in dflt.items() if not isinstance(v, bool)},
                   "cleanup_fragments": bool(dflt["cleanup_fragments"])},
                  open(f"{REFDIR}/is_{name}.meta.json", "w"), indent=1)
        print(f"  {name}: {Wd}x{H}  fwd max|Δ|={dd:.1e}  instances={n_inst}  "
              f"{'OK' if dd < 1e-3 else 'CHECK'}")


if __name__ == "__main__":
    main()
