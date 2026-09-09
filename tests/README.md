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

StarDist has a second sample, `sd_he_histo`, covering the **RGB H&E checkpoint**
(`2D_versatile_he`, which `stardist.mjs` exercises alongside the grayscale
`2D_versatile_fluo`). It is a 256×256 centre-crop of StarDist's shipped H&E example
(`stardist.data.test_image_he_2d()`), with a 3-channel `input.bin` (`[3,H,W]`). The
harness picks the weights (`stardist-fluo` vs `stardist-he`) from the sample prefix.

Layout:
```
refdata/cellpose/   cellpose_img_075[.input/.output/.masks/.raw.bin, .meta.json] (+ _ch23 two-channel)
refdata/stardist/   sd_cellpose_img_075[.input/.prob/.dist/.labels.bin, .meta.json]   (fluo, grayscale)
                    sd_he_histo[.input/.prob/.dist/.labels.bin, .meta.json]           (H&E, RGB [3,H,W])
refdata/instanseg/  is_cellpose_img_075[.input/.output/.labels.bin, .meta.json]
refdata/tiling/     tiling_ref.json + taper_*.f32.bin
```

## Regenerating the full reference set

The committed subset is one sample per model. To reproduce the full references (or add more
samples), use the generators in `../reference/` and `../tools/` — e.g.
`reference/baseline_pytorch.py` for the cellpose dumps. These scripts were written against the
original development layout, so adjust their `refdata/` / `images/` paths to your setup before
running.

The StarDist references come from the original TensorFlow/Keras model. Set up its isolated
env once (kept apart from the Torch exporters — TF and Torch pin conflicting deps):

```bash
python3 -m venv tools/.venv-stardist
tools/.venv-stardist/bin/pip install -r tools/requirements-stardist.txt

tools/.venv-stardist/bin/python tools/export_stardist.py         # 2D_versatile_fluo -> sd_*
tools/.venv-stardist/bin/python tools/export_stardist_he.py      # H&E weights (h5py repack)
tools/.venv-stardist/bin/python tools/export_stardist_he_ref.py  # 2D_versatile_he -> sd_he_histo
```

`export_stardist_he_ref.py` writes straight into `tests/refdata/stardist/` and runs on CPU
(~256×256 crop); it self-checks a NumPy reference forward against Keras (< 1e-4) before
committing anything.
