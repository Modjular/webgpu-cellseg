"""
PyTorch cyto3 baseline + reference data for the WebGPU port.

For each job (see `jobs` in main) this:
  1. Runs a *controlled* whole-image pipeline (the exact one the WebGPU demo
     mirrors): select channels -> normalize99 per channel -> pad to /16 -> CPnet
     forward -> cellpose dynamics.  Dumps the raw channels, net input, net output
     and reference masks so the browser can validate preprocessing, the forward
     pass and the masks.
  2. Times the network forward pass on MPS and CPU (warm, median of N).
  3. Also runs canonical `model.eval()` for a secondary fidelity reference.

Every sample gets a grayscale (channels=[0,0]) job.  Samples listed in
TWO_CHANNEL_JOBS additionally get a two-channel (cyto+nuclei) job, dumped under
`<name>_ch<c1><c2>`, since cyto3 is trained on [cytoplasm, nuclear] pairs and the
port needs a reference for that path too.

Outputs:
  refdata/<name>.raw.bin      float32 [2,H,W]    (selected channels, un-normalized)
  refdata/<name>.input.bin    float32 [2,Hp,Wp]  (network input, padded)
  refdata/<name>.output.bin   float32 [3,Hp,Wp]  (network output, padded)
  refdata/<name>.masks.bin    int32   [H,W]      (reference masks, cropped)
  refdata/<name>.meta.json    shapes + params + channels
  refdata/<name>.masks.npy    (for python-side AP checks)
  results.json                appended timing rows for pytorch_mps / pytorch_cpu

Run:  .venv312/bin/python baseline_pytorch.py
"""
import os
import glob
import json
import time
import numpy as np
import torch
from PIL import Image
from cellpose import models, dynamics, metrics

DIAMETER = 30.0          # cyto3 native diameter -> rescale = 30/DIAMETER = 1 (no resize)
CELLPROB_THRESHOLD = 0.0
FLOW_THRESHOLD = 0.4     # cellpose default QC step; browser mirrors this (remove_bad_flow_masks)
MIN_SIZE = 15
NITER = 200
N_TIME = 5               # timed repeats (median)
REFDIR = "refdata"
RESULTS = "results.json"

# Samples whose colour planes carry genuinely distinct cytoplasm/nuclear signal,
# mapped to the cellpose `channels` pair to use for an *extra* two-channel
# reference (on top of the grayscale one every sample gets).
# cellpose_img_075 is the classic green-cytoplasm / blue-nuclei sample; the other
# two samples store the same grayscale replicated across all three planes, so a
# two-channel reference for them would be meaningless.
TWO_CHANNEL_JOBS = {"cellpose_img_075": [2, 3]}   # [cyto=green, nuclei=blue]


def select_channels(im, channels=(0, 0)):
    """
    PIL image + cellpose `channels` -> (chan, chan2) float32 [H,W], raw units.

    Mirrors cellpose transforms.reshape: channels[0] is the channel to segment
    (0=grayscale i.e. mean of colour channels, 1=red, 2=green, 3=blue) and
    channels[1] the optional nuclear channel (0=none).  chan2 is None when there
    is no nuclear channel, which is what makes the net's 2nd plane stay zero.
    """
    a = np.asarray(im).astype(np.float32)
    if a.ndim == 2:
        return a, None
    a = a[..., :3]
    if channels[0] == 0:
        return a.mean(axis=2), None
    chan = a[..., channels[0] - 1]
    chan2 = a[..., channels[1] - 1] if channels[1] > 0 else None
    return chan, chan2


def normalize99(x, lower=1, upper=99):
    """
    Percentile-normalize one channel, mirroring cellpose's two guards:
    transforms.normalize_img skips a channel whose ptp is 0, and
    transforms.normalize99 zeroes one whose 1..99 spread is <= 1e-3.

    Both matter now that a nuclear channel is selectable: a blank or constant
    plane is entirely realistic input (wrong channel picked, or an image with no
    nuclear stain), and dividing by its ~0 spread would turn a dead channel into
    garbage the net would then happily consume.  Neither guard fires for real
    grayscale, so existing references stay byte-identical.  The +1e-10 is kept
    from the original for the same reason.
    """
    x = x.astype(np.float32)
    if np.ptp(x) <= 0:
        return x                                # constant: cellpose leaves as-is
    p1 = np.percentile(x, lower)
    p99 = np.percentile(x, upper)
    if p99 - p1 <= 1e-3:
        return np.zeros_like(x)                 # no dynamic range: cellpose zeroes it
    return (x - p1) / (p99 - p1 + 1e-10)


def pad_to(x, div=16):
    """Pad (C,H,W) with reflect to multiple of div. Returns padded, (H,W)."""
    C, H, W = x.shape
    Hp = ((H + div - 1) // div) * div
    Wp = ((W + div - 1) // div) * div
    out = np.zeros((C, Hp, Wp), np.float32)
    out[:, :H, :W] = x
    if Hp > H:
        out[:, H:Hp, :W] = x[:, H - 1:H, :][:, ::-1, :][:, :Hp - H, :]
    if Wp > W:
        out[:, :, W:Wp] = out[:, :, W - 1:W][:, :, ::-1][:, :, :Wp - W]
    return out, (H, W)


def preprocess(path, channels=(0, 0)):
    """Image path -> (raw [2,H,W], net_input [2,Hp,Wp] float32, (H,W)).

    `raw` is the *selected but un-normalized* channel pair.  It's dumped so the
    JS port can validate its own normalize99 + padding against this reference,
    rather than only the forward pass (which is fed the already-preprocessed
    input.bin and so would never exercise the preprocessing at all).
    """
    chan, chan2 = select_channels(Image.open(path), channels)
    zero = np.zeros_like(chan)
    raw = np.stack([chan, chan2 if chan2 is not None else zero], 0)
    x = np.stack([normalize99(chan),
                  normalize99(chan2) if chan2 is not None else zero], 0)   # (2,H,W)
    padded, HW = pad_to(x, 16)
    return raw.astype(np.float32), padded, HW


def run_net(net, inp, device):
    """inp: numpy (2,Hp,Wp). Returns output numpy (3,Hp,Wp)."""
    t = torch.from_numpy(inp[None]).to(device)
    with torch.no_grad():
        out = net(t)[0]
    return out.cpu().numpy()[0]


def timed(fn, n=N_TIME, warmup=2):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts))


def masks_from_output(out, HW):
    H, W = HW
    dP = out[:2, :H, :W].astype(np.float32)
    cellprob = out[2, :H, :W].astype(np.float32)
    masks = dynamics.compute_masks(
        dP, cellprob, niter=NITER, cellprob_threshold=CELLPROB_THRESHOLD,
        flow_threshold=FLOW_THRESHOLD, interp=True, min_size=MIN_SIZE,
        device=torch.device("cpu"))
    return masks, dP, cellprob


def append_results(rows):
    data = []
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            data = json.load(f)
    keys = {(r["checkpoint"], r["image"]) for r in rows}
    data = [r for r in data if (r.get("checkpoint"), r.get("image")) not in keys]
    data.extend(rows)
    with open(RESULTS, "w") as f:
        json.dump(data, f, indent=1)


def main():
    os.makedirs(REFDIR, exist_ok=True)
    have_mps = torch.backends.mps.is_available()
    print("MPS available:", have_mps)

    m = models.CellposeModel(gpu=False, model_type="cyto3")
    net_cpu = m.net.eval()
    net_mps = None
    if have_mps:
        import copy
        net_mps = copy.deepcopy(m.net).to("mps").eval()

    # One grayscale job per sample (the long-standing references, unchanged),
    # plus a two-channel job for samples that actually have a nuclear plane.
    jobs = []
    for path in sorted(glob.glob("images/*.png")):
        name = os.path.splitext(os.path.basename(path))[0]
        jobs.append((name, path, [0, 0]))
        if name in TWO_CHANNEL_JOBS:
            ch = TWO_CHANNEL_JOBS[name]
            jobs.append((f"{name}_ch{ch[0]}{ch[1]}", path, ch))

    rows = []
    for name, path, channels in jobs:
        raw, inp, HW = preprocess(path, channels)
        H, W = HW
        Hp, Wp = inp.shape[1:]
        print(f"\n{name}: {W}x{H} -> padded {Wp}x{Hp}  channels={channels}")

        # controlled reference forward (CPU) + masks
        out = run_net(net_cpu, inp, "cpu")
        masks, dP, cellprob = masks_from_output(out, HW)
        n_masks = int(masks.max())
        print(f"  reference masks: {n_masks}")

        # dump reference data for the browser
        raw.astype("<f4").tofile(f"{REFDIR}/{name}.raw.bin")
        inp.astype("<f4").tofile(f"{REFDIR}/{name}.input.bin")
        out.astype("<f4").tofile(f"{REFDIR}/{name}.output.bin")
        masks.astype("<i4").tofile(f"{REFDIR}/{name}.masks.bin")
        np.save(f"{REFDIR}/{name}.masks.npy", masks)
        with open(f"{REFDIR}/{name}.meta.json", "w") as f:
            json.dump({"name": name, "H": H, "W": W, "Hp": Hp, "Wp": Wp,
                       "channels": list(channels),
                       "diameter": DIAMETER, "niter": NITER,
                       "cellprob_threshold": CELLPROB_THRESHOLD,
                       "flow_threshold": FLOW_THRESHOLD, "min_size": MIN_SIZE,
                       "n_masks": n_masks}, f, indent=1)

        # timing: network forward only (the part WebGPU replaces)
        cpu_ms = timed(lambda: run_net(net_cpu, inp, "cpu"))
        rows.append({"checkpoint": "pytorch_cpu", "label": "PyTorch CPU (forward)",
                     "device": "cpu", "image": name, "size": [W, H],
                     "forward_ms": cpu_ms, "n_masks": n_masks})
        print(f"  forward CPU: {cpu_ms:.1f} ms")
        if net_mps is not None:
            def mps_fwd():
                t = torch.from_numpy(inp[None]).to("mps")
                with torch.no_grad():
                    o = net_mps(t)[0]
                torch.mps.synchronize()
                return o
            mps_ms = timed(mps_fwd)
            rows.append({"checkpoint": "pytorch_mps", "label": "PyTorch MPS (forward)",
                         "device": "mps", "image": name, "size": [W, H],
                         "forward_ms": mps_ms, "n_masks": n_masks})
            print(f"  forward MPS: {mps_ms:.1f} ms")

        # secondary: canonical cellpose eval (tiled pipeline) for fidelity context
        try:
            cmask, _, _ = m.eval(np.asarray(Image.open(path)), diameter=DIAMETER,
                                 flow_threshold=FLOW_THRESHOLD,
                                 cellprob_threshold=CELLPROB_THRESHOLD,
                                 niter=NITER, channels=list(channels))
            ap = metrics.average_precision(masks.astype(np.int32),
                                           cmask.astype(np.int32),
                                           threshold=[0.5])[0][0]
            print(f"  canonical eval masks: {int(cmask.max())}  "
                  f"AP@0.5(controlled vs canonical)={ap:.3f}")
        except Exception as e:  # eval path differences shouldn't block refs
            print("  canonical eval skipped:", e)

    append_results(rows)
    print("\nwrote refdata/ and appended", len(rows), "rows to", RESULTS)


if __name__ == "__main__":
    main()
