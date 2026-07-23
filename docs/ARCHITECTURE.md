# Architecture

Each model is one self-contained ES module in `src/` exposing a `*WebGPU` class. No build
step, no dependencies. The pattern is the same across all three:

```
raw image (grayscale/RGB Float32)
   → preprocess           (JS: normalize, resize-by-scale, pad)     [CPU]
   → CNN forward           (hand-written WGSL, single command encoder) [GPU]
   → decode/post-process   (JS + optional WGSL)                       [CPU/GPU]
   → integer label image
```

The **CNN forward is WGSL**; the **instance decode is JS** (ported from the reference
implementation) — correctness-first, with the hottest loops moved to WGSL where it paid off.

## Shared building blocks (all three engines)

- **Pooled scratch buffers.** GPU buffers are acquired from a free-list keyed by
  `(usage, size)` and returned to the pool at the end of each forward, not destroyed —
  this removes per-op allocate/zero/free churn (hundreds of MB per call). See `_acquire` /
  `releaseAll` / `mkStorage`.
- **Single-submit forward.** The whole encoder → GPU-style → decoder → output head is
  recorded into one `GPUCommandEncoder` and submitted once; no mid-forward CPU stall for
  readback.
- **Direct tiled conv in WGSL.** A 16×16 workgroup stages an 18×18 haloed input tile per
  input channel into shared memory and reuses it across kernel taps and output channels.
  (This beat an implicit-GEMM conv here — see GOTCHAS #4.)
- **`load(baseURL)` convenience.** `await Model.load()` fetches `manifest.json` +
  `weights.bin` (from `../weights/<model>/` by default, or a URL you pass) and returns a
  ready instance. `manifest.json` maps tensor name → `{offset, length, shape}` into the raw
  little-endian float32 `weights.bin`; the class keeps them URL-agnostic so you can host
  weights anywhere.

## Cellpose (`src/cellpose.js`)

- **Net:** CPnet U-Net (residual down/up blocks) → 3 outputs: `dY`, `dX`, `cellprob`. A
  256-d **style vector** (global-avg-pool of the deepest features, L2-normalized) conditions
  every decoder block.
- **Tiled inference** (`runNet`): exact mirror of cellpose `core.run_net` — zero-pad via
  `get_pad_yx`, overlapping 224² tiles, **per-tile** forward (per-tile style), `_taper_mask`
  weighted blend, crop. This is what makes non-native diameters match desktop cellpose; see
  GOTCHAS #1. Single-tile images take a fast path (blend is a no-op).
- **Flow dynamics:** seed pixels where `cellprob > threshold`, follow the (interpolated,
  clamped) flow field for `niter = 200/rescale` Euler steps, then connected-components +
  size/flow QC. The Euler integration runs as a WGSL kernel; histogram/labeling in JS.

## StarDist (`src/stardist.js`)

- **Net:** U-Net → per-pixel object probability + 32 radial distances (star-convex rays).
- **Decode (JS):** threshold probability, build a star-convex polygon per candidate pixel
  from its 32 rays, non-max-suppress by polygon overlap (`nms_thresh`), rasterize survivors
  to labels.

## InstanSeg (`src/instanseg.js`)

- **Net:** U-Net → learned per-pixel **embeddings** + a seed/foreground field.
- **Decode (JS):** pick seeds above `seed_thresh`, assign pixels to the nearest seed in
  embedding space subject to `mask_thresh`, emit instance labels. Input is RGB, percentile-
  normalized; padded to a multiple of 8.

## Reference ports (`reference/`)

`mini_cellpose.py` / `mini_stardist.py` / `mini_instanseg.py` are single-file, **NumPy-only**
teaching ports (nanoGPT-style) that load the same exported weights and reproduce each model
stage by stage. They are the fidelity oracle the WGSL engines are validated against, and the
`*_walkthrough.ipynb` notebooks plot the output of every layer. `baseline_pytorch.py`
generates the reference dumps under `tests/refdata/`.
