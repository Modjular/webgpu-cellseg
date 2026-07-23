# Tests

Fidelity harnesses that check the WGSL forward + JS decode against the NumPy/PyTorch
references. Paths are resolved relative to each test file, so you can run them from anywhere.

## Run

```bash
# Pure-logic tile-grid + taper-mask unit tests (no GPU) — validates the tiling fix
deno run --allow-read tests/cellpose_tiling.mjs

# WGSL forward + dynamics vs PyTorch dumps (needs WebGPU)
deno run --unstable-webgpu --allow-read tests/cellpose_forward.mjs
deno run --unstable-webgpu --allow-read tests/stardist.mjs
deno run --unstable-webgpu --allow-read tests/instanseg.mjs
```

Each prints per-sample `max|Δ|`, mask counts, and AP@0.5, ending in `ALL OK` / `FAIL`.

## Reference data

`refdata/<model>/` holds a **minimal subset** — the smallest sample (`cellpose_img_075`,
240×300) per model — so the harnesses run out of the box. `refdata/tiling/` holds the
authoritative tile-grid + taper-mask reference dumped from cellpose.

Layout:
```
refdata/cellpose/   cellpose_img_075[.input/.output/.masks/.raw.bin, .meta.json] (+ _ch23 two-channel)
refdata/stardist/   sd_cellpose_img_075[.input/.prob/.dist/.labels.bin, .meta.json]
refdata/instanseg/  is_cellpose_img_075[.input/.output/.labels.bin, .meta.json]
refdata/tiling/     tiling_ref.json + taper_*.f32.bin
```

## Regenerating the full reference set

The committed subset is one sample per model. To reproduce the full references (or add more
samples), use the generators in `../reference/` and `../tools/` — e.g.
`reference/baseline_pytorch.py` for the cellpose dumps. These scripts were written against the
original development layout, so adjust their `refdata/` / `images/` paths to your setup before
running.
