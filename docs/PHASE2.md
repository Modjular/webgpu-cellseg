# Phase 2 — subgroup-matrix, first cut

docs/PHASE1.md's item 2: `subgroup-matrix` (Metal simdgroup matmul via
`chromium-experimental-subgroup-matrix`) was flagged as the only substantial lever left on
`conv` once parameter tuning hit an interior optimum (Phase 1c), and explicitly called out
as deserving its own phase rather than a tuning pass. This is that phase's first cut: the
WGSL syntax for this API (undocumented anywhere findable — see below), a correctness-checked
GEMM microkernel using it, and a first throughput number. It is **not** a replacement for the
shipping conv kernel yet — see "What this is not" below.

Run it yourself: `python3 -m http.server 8000` then open
`http://localhost:8000/demo/subgroup_matrix_gemm.html` in Chrome.

## The API had to be reverse-engineered from the compiler, not read from docs

`adapter.features` reports `chromium-experimental-subgroup-matrix` on this machine (Chrome
150, Metal via ANGLE), and Dawn's `wgsl.def` (the source of truth for builtin signatures)
gives the type shape — `subgroup_matrix<Kind, T, C, R>` — but not the actual surface syntax,
and the GPU-Web WGSL working group was still actively changing the load/store layout
parameter as of the 2026-03-24 meeting. Guessing from search results produced two wrong
syntaxes in a row. What worked was writing a shader, calling
`shaderModule.getCompilationInfo()`, and reading Tint's own overload-resolution error, which
lists every candidate signature and marks which template/argument positions matched:

```
error: no matching call to 'subgroupMatrixMultiply(subgroup_matrix_left<f32, 8, 8>, subgroup_matrix_right<f32, 8, 8>)'
4 candidate functions:
 • 'subgroupMatrixMultiply<TR ✗>(subgroup_matrix<left, T, K, R> ✓, subgroup_matrix<right, T, C, K> ✓) -> subgroup_matrix<result, TR, C, R>' where:
      ✗  'TR' is 'f32'
      ✓  'T' is 'f32'
```

That single error revealed both that the named types (`subgroup_matrix_left/right/result<T,
C, R>`) were already correct and that the result-type template parameter needed to be given
explicitly. This is a reusable technique for tracking bleeding-edge WGSL extensions in
general, not just this one: `getCompilationInfo()` diagnostics are more current than any
document.

**Validated on this adapter, Chrome 150, Metal backend:**

```wgsl
enable chromium_experimental_subgroup_matrix;

let a = subgroupMatrixLoad<subgroup_matrix_left<f32, 8, 8>>(&buf, offset, /*col_major*/ false, /*stride*/ n);
let b = subgroupMatrixLoad<subgroup_matrix_right<f32, 8, 8>>(&buf2, offset2, false, n2);
var acc = subgroup_matrix_result<f32, 8, 8>();          // zero-value construction
acc = subgroupMatrixMultiplyAccumulate(a, b, acc);       // TR inferred from acc here
// or, with no running accumulator:
let c = subgroupMatrixMultiply<f32>(a, b);               // TR must be given explicitly
subgroupMatrixStore(&outBuf, outOffset, c, false, outStride);
```

- Tile shape is 8×8 on this hardware (Metal `simdgroup_matrix` is fixed 8×8); one simdgroup
  is 32 lanes (`@workgroup_size(32)` for a single-tile kernel), matching `subgroups` on this
  adapter.
- `subgroupMatrixLoad` reads directly out of a `storage` buffer — no shared-memory staging
  required to use the op at all, which is what made a first correctness check cheap.
- `requiredFeatures` needs both `chromium-experimental-subgroup-matrix` and (implicitly)
  `subgroups`; launching Chrome also needed `--enable-dawn-features=allow_unsafe_apis` for
  the device request to actually pick up the requested feature in this environment.

## The GEMM: correct, and a first (unoptimized) throughput number

`demo/subgroup_matrix_gemm.html` runs two checks. The first is an 8×8 identity sanity check
that pins down calling convention (`A * I == A` under `col_major=false`, confirming
row-major storage and stride-in-elements). The second is the actual target: a tiled GEMM,
`C[Cout,HW] = A[Cout,Cin] @ B[Cin,HW]`, which is **exactly the shape and layout a K=1
(pointwise) conv already uses** — no im2col needed, so it was the right first target rather
than K=3.

One workgroup = one simdgroup = one 8×8 output tile, looping over `Cin` in 8-wide chunks,
loading both operand tiles straight out of the storage buffers each iteration (no
shared-memory staging — see below). At a representative shape (Cout=256, Cin=256, HW=8192,
all multiples of 8, chosen for round dimensions rather than tied to one exact layer):

| | value |
|---|---|
| correctness | 9 spot-checked entries against a CPU f64 reference, maxRel 5.73e-7 (f32 summation noise, same order as Phase 1c's CB-staging reorder) |
| throughput | ~650 GFLOP/s, ~21% of the ~3100 GFLOP/s roof |
| shipping kernel, for comparison | 1227 GFLOP/s / 40% of roof (Phase 1c, BLK=16/2×2/CB=4) |

So: **correct, and already at half the shipping kernel's throughput with zero staging
optimization** — no shared-memory reuse, no register blocking beyond the hardware's native
8×8 tile, one dispatch per 8×8 output block. That headroom is the signal that this is worth
continuing, matching Phase 1c's own read that ~40% of roof leaves ~2.5× on the table and
subgroup-matrix is the only lever left to get at it.

## What this is not, yet

This is a GEMM microkernel, not a conv. Concretely still missing before it could replace
anything in `src/conv-kernel.js`:

1. **im2col for K=3.** The pointwise (K=1) case is a GEMM already, which is why it was the
   right first target — but K=3 is ~95% of conv time (docs/PHASE0.md, docs/PHASE1.md). The
   "B" operand for K=3 needs the 3×3 spatial neighborhood unfolded into the reduction
   dimension; it is not naturally contiguous in memory the way the K=1 case is. This is the
   real rewrite the phase 1 handoff flagged, and it is still ahead, not done here.
2. **Shared-memory staging.** Every workgroup currently re-reads its `A`-tile and `B`-tile
   rows from global memory on every dispatch; neighboring workgroups sharing a `Cin` chunk
   share nothing. This is the CB-staging lesson from Phase 1c repeating one level up.
3. **Multiple tiles per workgroup.** One simdgroup, one 8×8 output tile, one dispatch — the
   RBY×RBX-style register blocking that was worth 1.6× in the scalar kernel has no analogue
   here yet.
4. **BN fusion, relu, residual, bias.** The shipping kernel's `mode="bn"` staging path (BN
   scale/shift and relu folded into the input read) and the skip/residual adds have no
   equivalent in this microkernel at all.
5. **Non-multiple-of-8 shapes.** `Cin=2` (the stem) and any other non-8-multiple dimension is
   unhandled — `subgroupMatrixLoad` against a storage buffer needs in-bounds reads, and nothing
   here pads or guards for it.
6. **f16 element type.** Dawn's `wgsl.def` shows `subgroup_matrix` element types beyond f32
   (f16, i8/u8, i32/u32) — untested here. Given docs/PHASE1.md's f16 rejection and this
   session's item 2 (`tests/cellpose_f16.mjs`), any f16-element subgroup-matrix variant would
   need the same IoU/cell-count bar, not a tolerance, before being trusted.

## Next, in order

1. Add shared-memory staging to the GEMM microkernel (item 2 above) and re-measure — this is
   the cheapest remaining lever on the number already in hand, same lesson as Phase 1c.
2. im2col the K=3 case. This is the one that actually matters (95% of conv time) and the one
   the original phase-1 handoff meant by "a real rewrite... deserves its own phase."
3. Once K=3 is real, fold BN/relu/residual back in and correctness-check against the frozen
   reference the way every other kernel change in this repo has been (bit-exact where
   possible, IoU/cell-count where not, never a raw tolerance alone).
4. Re-run `tests/refdata`-shaped correctness plus a Chrome throughput comparison against the
   shipping kernel before considering this for `src/conv-kernel.js` itself.
