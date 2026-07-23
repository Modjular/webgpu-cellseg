# Gotchas & lessons learned

Every non-obvious thing that cost real debugging time porting Cellpose / StarDist /
InstanSeg to browser WebGPU. If you're re-implementing one of these models — in WGSL or
any other framework — read this first. Roughly ordered by how much time each one cost.

---

## 1. Tiled inference with a *per-tile* style vector (the big one)

**Symptom:** the WGSL forward matched PyTorch to ~1e-5 at the model's *native* diameter,
but at other diameters the flow field diverged badly — `max|diff|` of **8–9** on values
ranging ±5, correlation only 0.97–0.99. Masks were visibly wrong.

**Cause:** Cellpose (`core.run_net`) never runs an image larger than **224 px** (either
dimension, at the *working*/rescaled resolution) through the network in one pass. It:

1. zero-pads the working-res image to a multiple of 16 **plus an extra 8 px per side**
   (`transforms.get_pad_yx`, `div=16, extra=1`), with `mode="constant"` (**zeros**, not
   edge-replicate),
2. cuts it into overlapping **224×224 tiles** (`make_tiles`, `tile_overlap=0.1`),
3. runs the net **per tile** — and the "style vector" that conditions every decoder block
   is computed from **that tile's own local 224×224 view** (`style = make_style(T0[-1])`),
4. blends the per-tile outputs back with a tapered weight mask (`_taper_mask` →
   `average_tiles`), then crops the padding off.

Running the whole image in one pass computes **one global style vector** instead of one
per tile — a structural difference, not a rounding error. It only shows up once the
working resolution exceeds 224 (e.g. a 960×1280 image at diameter 100 rescales to
288×384, which tiles into 6 patches; at diameter 50 → 576×768 → 20 patches).

**Fix:** `src/cellpose.js`'s `runNet()` mirrors `run_net` exactly. After the fix:
`max|diff|` **8–9 → ~1e-4**, correlation **→ 1.000000**, and segmentation **AP@0.5 = 1.0
vs desktop Cellpose** on both passes (cyto diam 100: 190/190 objects; nuclei diam 50:
183/183; 0 FP/FN). See `tests/cellpose_tiling.mjs` for the grid/taper unit tests validated
against cellpose's actual output.

**Why the native-diameter case worked all along:** at diameter 30 (rescale 1) on the small
reference crops, the working resolution is ≤224, so cellpose produces a *single* tile — and
with one tile the taper mask cancels (`yf = y·mask/mask = y`). Single-tile = whole-image,
so global-style and per-tile-style are identical *by construction*. The bug was invisible
until you changed the diameter.

**Two sub-gotchas inside this one:**
- The padding cellpose uses for tiling is **zeros + the extra 8 px/side**, which differs
  from a naive "pad to multiple of 16 with edge-replicate." Both the amount and the mode
  matter; match `get_pad_yx` exactly.
- Tiles/blend operate in **padded coordinates**; you crop the padding off *after* blending.

---

## 2. Preprocessing must be bit-exact — several silent traps

The network is sensitive; small preprocessing differences compound through a deep U-Net.

- **`normalize99` percentile indexing.** Cellpose's percentile is effectively a
  floor-indexed nearest-rank, not numpy's linear-interpolated `np.percentile`. Match the
  indexing cellpose actually uses (and its guards: a constant channel is left as-is, a
  channel with 1–99 spread ≤ 1e-3 is zeroed). On real images the two agree, but a hand-port
  that "looks right" can be off.
- **Resize dimension rounding is truncation, not rounding.** Cellpose uses
  `int(dim * rescale)` (`transforms.resize_image`), i.e. `Math.trunc`, **not** `Math.round`.
  They agree for many diameters and disagree for others — a latent off-by-one that shifts
  the whole working resolution.
- **`niter = 200 / rescale`.** Flow dynamics run at *full* resolution but on flows computed
  at the *rescaled* resolution, so the Euler integration needs `200/rescale` iterations, not
  200. Use the *effective* rescale actually applied.
- **Edge-replicate vs zero pad.** The old single-pass path edge-replicated to a multiple of
  16; real cellpose zero-pads (see #1). For a small image the padded border is a big fraction
  of the receptive field, so this is not negligible.

Bilinear resize (`cv2.INTER_LINEAR`, pixel-center sampling) and 2×2 maxpool / nearest
upsample *did* match to float32 noise — those were ruled out early. Don't assume; dump both
sides and diff stage-by-stage.

---

## 3. Apple-silicon / portable-WGSL hardware limits

Discovered on an M-series Mac (Metal via ANGLE/wgpu). These bit us as silent failures or
crashes, not friendly errors:

- **`maxStorageBufferBindingSize` is often 128 MB**, and several large intermediate buffers
  are live at once during a forward pass. Pushing a single big buffer near the limit crashes
  *bind-group creation* under real memory pressure, not just the allocation. (Tiling fixed
  this for free: no forward pass ever allocates more than one 224² tile's worth of buffers,
  so image size no longer constrains GPU memory — see #1.)
- **65535 max workgroups per dispatch dimension** → dispatch over x/y/channel as a 3D grid
  for large fields, don't linearize into one dimension.
- **32 KB workgroup (shared) memory**, and the backend is **occupancy-bound**: small shared
  footprints win. A conv kernel with a ~1.6 KB shared tile beat larger-footprint variants.

---

## 4. Direct conv beats implicit-GEMM here (perf)

The textbook way to get conv throughput is im2col + GEMM with register tiling. On this
backend it was **1.85–2.5× slower** than a direct high-occupancy tiled conv, across three
tile configs (all numerically correct). Two reasons:

1. **GEMM sacrifices receptive-field reuse** — it re-gathers/re-activates the input once per
   kernel tap (~9× more input traffic), whereas the direct conv stages an 18×18 input tile
   once per input channel and reuses it across all 9 taps × 8 output channels.
2. **Occupancy** — the backend rewards small shared footprints; even 0.6 KB GEMM tiles
   couldn't overcome (1).

Closing the remaining gap to PyTorch-MPS (~22×) would need Apple SIMD-group matrix
intrinsics, which portable WGSL does not expose. Full write-up: the `cp008` experiment notes.
Also: **f16 storage+arithmetic** gave ≤4% speedup but dropped AP@0.5 to 0.78 on a hard image
— not worth it. Keep f32.

---

## 5. Environment: cyto3 needs Cellpose 3.x → Python 3.12

Cellpose 4.x **dropped the cyto3 CNN** (only the Cellpose-SAM/ViT model remains). cyto3
needs cellpose 3.x, which pins `numpy<2.1` → incompatible with Python 3.13. Use Python 3.12
(`uv venv`, `cellpose==3.1.1.x`) for anything that touches cyto3 (weight export, reference
dumps). StarDist and InstanSeg have their own env pins (TensorFlow/Keras and PyTorch
respectively) — keep them in separate venvs.

---

## 6. Testing WebGPU headlessly, and a demo race that looked like a bug

- **Deno runs the exact WGSL** via `deno run --unstable-webgpu` — the fastest correctness
  loop, no browser needed. For full-browser checks, `puppeteer-core` driving installed Chrome
  with `--enable-unsafe-webgpu --use-angle=metal` works.
- **The "undefined buffer" scare.** A headless smoke test of the demo failed with
  `createBindGroup ... buffer is undefined` — looked like a serious engine bug. It was the
  *test harness*: the Segment button had no `disabled` attribute, so it was clickable
  immediately, and the harness clicked it **before the 26 MB weights finished loading** →
  the forward ran with an empty weight table. Real users wait for "ready." Lesson: (a)
  disable the run control until weights load (this repo's demos now start `disabled`), and
  (b) in automated tests, wait for the actual *ready* signal, never just for the button.

---

## 7. Hosting the converted weights — why you must self-host

If you ship an in-browser port, you **cannot** just point at the upstream model endpoint:

1. **Wrong format.** Cellpose fetches PyTorch checkpoints from `www.cellpose.org/models`;
   the browser needs your *converted* `weights.bin` + `manifest.json`. That artifact only
   exists because you made it.
2. **No CORS.** A browser `fetch()` to a third-party host you don't control is blocked unless
   that host sends `Access-Control-Allow-Origin`. Model-download endpoints don't.

So self-host the converted weights (permitted: cyto3/StarDist are BSD-3-Clause, InstanSeg is
Apache-2.0 — redistribution with attribution is fine; see `weights/*/LICENSE` + `NOTICE`).
Practical options:
- **HuggingFace model repo** — free, CORS-enabled, versioned; built for exactly this.
- **jsDelivr on a GitHub tag** — CORS + immutable URLs, but it caps GitHub files at ~20 MB
  (cyto3 is 26 MB → may be rejected; the smaller ones are fine).
- **Committed in the repo** (what this repo does) — simplest and fully self-contained, at the
  cost of clone size. `Model.load(url)` takes a base URL so you can flip to a CDN with no code
  change; keep `tools/export_*.py` + `SHA256SUMS` so anyone can regenerate and verify.
