# Handoff — kernel optimisation work landed from downstream

**Temporary.** Delete this once the contents have been folded into `README.md` /
`docs/ARCHITECTURE.md` or into a PR description.

Everything here is committed on branch `perf/conv-kernel-and-flow-qc` in two commits: the
Cellpose work plus tooling, then the StarDist/InstanSeg port.

This transfers a Cellpose optimisation effort that was carried out in a downstream repo
which vendors these modules. The engine changes are **~10× end-to-end on Cellpose** with
segmentation output unchanged, plus the measurement tooling that produced them.

Nothing about the public API changed. `CellposeWebGPU.load()` → `.segmentImage()` behaves
exactly as before.

---

## What landed

### New files

| file | what |
|---|---|
| `src/device.js` | adapter/device acquisition for all three models — **the raised `maxComputeWorkgroupStorageSize` here is required**, the conv kernel does not compile under WebGPU's 16 KB default |
| `src/conv-kernel.js` | the conv shader, generated per kernel size. Four constants, all measured optima |
| `src/profile/` | measurement library: per-dispatch timing, empirical roofline, analytical cost model, kernel taxonomy, benchmark harness |
| `tools/convbench.mjs` | conv variant benchmark (Deno) |
| `tools/profile.mjs` | end-to-end profile (Deno; refuses to run without trustworthy timestamps — see below) |
| `demo/profile.html` | the same profile in Chrome, which is the only place it is sound |
| `tests/cellpose_flowqc.mjs` | equivalence harness for the GPU flow QC |
| `docs/PHASE0.md`, `docs/PHASE1.md` | the measurement record and the reasoning behind every constant |

### Changed

- `src/cellpose.js` — new conv kernel, GPU flow-consistency QC, stage instrumentation,
  compute-pass labels, a `_mkEncoder` hook for profiling.
- `src/stardist.js`, `src/instanseg.js` — now share the same conv kernel in its "plain"
  form, plus device acquisition, the profiling hook and labelled passes.
  **StarDist forward 103 ms → 34 ms (3.0×); InstanSeg 434 ms → 83 ms (5.2×)**, both with
  AP@0.5 = 1.000 and identical mask counts. Gains are smaller than Cellpose's because
  both nets are shallower and neither had BN work to fuse into the staging load.

---

## The changes, in order of size

Full reasoning in `docs/PHASE1.md`; the short version:

1. **Conv kernel rebuilt — ~11× on the kernel.** The old one accumulated into
   `var acc: array<f32, BLK>` indexed by a loop whose bound came from a uniform. A
   dynamically-indexed private array does not stay in registers; on Metal it spills to
   thread-local memory, which is device-backed. Replacing it with named scalars was
   **5× on its own**. Then compile-time `K` (taps unroll), spatial register blocking, a
   larger `BLK`, and input-channel staging to cut barriers.

2. **Flow-consistency QC moved to the GPU — ~8× on that step.** `remove_bad_flow_masks`
   reconstructs each mask's flow with a 9-point diffusion. It was ~95% of mask-assembly
   time and, once the conv got faster, up to 79% of total wall clock. It now diffuses
   every mask at once over the whole image.

The order matters: the second only became worth doing *because* the first shifted the
bottleneck onto it. Expect the same again — see "what's next".

### The four conv constants are a joint optimum

`BLK`, `RBY`/`RBX`, `CB` in `src/conv-kernel.js` trade against two shared budgets:
registers (`BLK·RBY·RBX` live accumulators) and threadgroup memory (`CB` staged tiles).
The optimum is **interior on every axis** — raising any one of them loses, sometimes by
half (BLK=32 at 2×2 drops to 16% of roof). Do not reason about one in isolation, and
re-run the full sweep after touching any of them.

---

## Correctness — what was actually verified

This is the part to trust, and the bar for anything further.

| check | result |
|---|---|
| `tests/cellpose_forward.mjs` | **AP@0.5 = 1.000**, 162/149 masks, unchanged. `max|Δ|` vs PyTorch 2.25e-5 / 1.99e-5 (baseline was 1.38e-5 / 2.38e-5 — comparable, the two-channel case improved) |
| `tests/cellpose_flowqc.mjs` | GPU vs CPU flow QC: **0 of 72,000 pixels differ**, both channels |
| `tests/stardist.mjs`, `tests/instanseg.mjs` | unchanged, ALL OK |
| downstream end-to-end | 190 cells / 183 nuclei / 173 kept, unchanged through every step |

Two of the three kernel changes are **bit-exact** (the conv rebuild through register
blocking; the flow-QC index precomputation). Two are not, and the distinction was
deliberate:

- **Input-channel staging (`CB=4`)** reorders accumulation over `ci`, giving ~5e-7
  relative difference. That is f32 summation-order noise (f32 epsilon is 1.2e-7 over a
  256-channel reduction), not precision loss.
- **GPU flow QC** runs the diffusion in f32 where the CPU used f64 — but the
  normalisation and per-mask error stay on the CPU in f64 *on purpose*, because that is
  where the threshold comparison deciding how many masks survive happens. Verified by
  diffing label maps, not by a tolerance.

An **f16 shared-memory variant was measured and rejected**: 1.24× faster, but ~2.8e-4
relative error — 500× larger, and the flow field feeds a 200-step Euler integration whose
trajectories decide instance boundaries. It is still in `conv-variants.js` as
`regblk_2x2_f16`. Revisiting it needs IoU agreement against the reference across a real
image set, not a tolerance on one conv.

---

## Timing on Deno is not trustworthy — read before benchmarking

**Deno's WebGPU on Metal writes a zero timestamp for the last compute pass of every
encoder.** Earlier passes are timed correctly. Since a zero delta is indistinguishable
from an unwritten query, the naive failure mode is a profile that silently drops one
dispatch per encoder; a single-pass probe hits exactly the broken case and concludes
timestamps are wholly dead. Larger query sets can also fail allocation outright
("Cannot allocate sample buffer").

`timestampsWork()` in `src/profile/timing.js` probes this properly — several passes, all
must be nonzero — and everything falls back to batched wall clock when it fails.

Consequences:

- **Correctness in Deno: fully trustworthy.** All the tests above run there.
- **Performance in Deno: indicative at best.** Batched wall clock recovers the slow
  baseline accurately (3.5% of roof, matching Chrome) but understates fast kernels, and
  variant *ordering* can invert. Do not tune constants from Deno numbers.
- **Performance in Chrome: sound.** Open `demo/profile.html`. Timestamps there are
  nanosecond-resolution and unquantized.

---

## Running things

```bash
python3 -m http.server 8000          # for the demo pages

# correctness
deno run --allow-read                 tests/cellpose_tiling.mjs
deno run --unstable-webgpu --allow-read tests/cellpose_forward.mjs
deno run --unstable-webgpu --allow-read tests/cellpose_flowqc.mjs
deno run --unstable-webgpu --allow-read tests/stardist.mjs
deno run --unstable-webgpu --allow-read tests/instanseg.mjs

# kernel variants — correctness exact, timing indicative
deno run --unstable-webgpu --allow-read --allow-write tools/convbench.mjs --shapes 4 --per-shape

# the real profile
open http://localhost:8000/demo/profile.html
```

`demo/profile.html` takes ~10 s: it measures the machine's roofs empirically (compute,
bandwidth, ridge point, concurrent-workgroup capacity, launch floor) before profiling
anything, and reports progress as it goes.

---

## Status of the plan

Done: the Cellpose work, the StarDist/InstanSeg port, and vendoring into the downstream
repo (which now consumes `vendor/webgpu-cellseg/` and no longer carries its own copy of
the engines or the profiling tooling).

## What's next, cheapest first

1. **`subgroup-matrix`** (Metal simdgroup matmul) — available on this adapter, unused.
   The conv is at ~40% of roof and parameter tuning is exhausted, so this is the only
   substantial lever left on it. Real rewrite: implicit GEMM with im2col in shared memory.
2. **The flow-QC kernel dispatches over the whole image** regardless of mask coverage,
   which is why the small-mask case gained ~1.8× against the large-mask case's 8.2×.
   Bound the dispatch to the union of mask bounding boxes, or compact mask pixels.
3. **Re-test f16** properly (see above).
4. **Mask assembly after the flow QC** — seed growth and label assignment — is the next
   CPU item once the above shrink around it.
5. **Re-measure StarDist and InstanSeg.** Their conv shapes were never profiled — the
   constants in `conv-kernel.js` were tuned on Cellpose's layer shapes and inherited.
   `SHAPES` in `src/profile/convbench.js` is Cellpose-only; adding their shapes would show
   whether one set of constants really suits all three, or whether the shallower nets want
   a different split. Their 3.0×/5.2× against Cellpose's 11× is a hint that they might.
6. **A profiling cross-check across engines.** `demo/profile.html` had a panel comparing a
   second engine; it was removed when only Cellpose had the new kernel. Now that all three
   share it, that is the natural regression against the whole family.

---

## Loose ends

- `tools/profile.mjs` and `demo/profile.html` define their workloads independently. They
  agree today; they will drift.
- `src/profile/convbench.js`'s `SHAPES` and its time-share weights are Cellpose's. Any
  variant scored there is scored against Cellpose's mix, which is now only part of what
  the kernel serves — see item 5.
- Both engines prebuild conv pipelines for K∈{1,3} in their constructors. Correct for
  every current checkpoint, and anything else is built lazily, but a checkpoint using
  another kernel size silently pays shader compilation on its first forward.
- Workloads are built from `tests/refdata/cellpose_img_075` (240×300), including a
  synthetic 3×3 tiling to reach a multi-tile working resolution. Real content at a larger
  size would profile the tiling path more honestly.
- `docs/PHASE0.md` and `docs/PHASE1.md` carry numbers measured downstream on different
  images. The method and conclusions transfer; the specific milliseconds do not.
- `results/` is written by both tools and is not in `.gitignore` yet — decide whether
  those JSON files belong in the repo before committing.
