# webgpu-cellseg

Microscopy **cell/nucleus segmentation in the browser, on WebGPU**. These segmentation models
run on WebGPU with a JavaScript instance-decode. **No PyTorch, no TensorFlow, no CUDA, no
ML framework at inference time.** Each model is one self-contained ES module.

| Model | Module | Task | Native diameter | Weights license |
|---|---|---|---|---|
| **Cellpose cyto3** | `src/cellpose.js` | generalist cytoplasm | 30 px | BSD-3-Clause |
| **StarDist** `2D_versatile_fluo` | `src/stardist.js` | star-convex nuclei (fluorescence) | — | BSD-3-Clause |
| **InstanSeg** `brightfield_nuclei` | `src/instanseg.js` | brightfield nuclei | — | Apache-2.0 |

The Cellpose port implements cellpose-exact **tiled inference**, so it matches desktop
Cellpose at **AP@0.5 = 1.0** across diameters (not just the native one). Read more about the
porting journey in [`docs/GOTCHAS.md`](docs/GOTCHAS.md).

## Quickstart (demo)

WebGPU needs a secure context, so serve the folder over HTTP (don't open `file://`):

```bash
python3 -m http.server 8000
# then open http://localhost:8000/demo/
```

Pick a model, pick or upload an image (PNG/JPEG/etc., or **TIFF** — including 16-bit/float
grayscale, decoded client-side via the [`tiff`](https://www.npmjs.com/package/tiff) package
from jsDelivr), hit **Segment**. Requires a WebGPU browser (Chrome/Edge 113+, or Safari 18+).

## Use as a library

```js
import { CellposeWebGPU } from "./src/cellpose.js";

// loads ./weights/cellpose-cyto3/{manifest.json,weights.bin} by default
const cp = await CellposeWebGPU.load();

// gray: Float32Array length H*W (any range; normalized internally) — the channel to segment
// chan2 (optional): Float32Array length H*W, a second (nuclear) channel — cellpose's
// channels=[cyto, nuclear]; omit for cellpose's grayscale mode (channels=[0,0]).
const { labels, timings } = await cp.segmentImage(gray, H, W, { diameter: 30, chan2 });
// labels: Int32Array[H*W], 0 = background, 1..N = instances
```

`StarDistWebGPU` and `InstanSegWebGPU` follow the same shape (`.load()` → `.segmentImage()`)
but have a fixed, narrower channel contract — neither accepts extra channels. `StarDistWebGPU`
(`prob_thresh`/`nms_thresh`) is strictly single-channel; `InstanSegWebGPU` (`seed_thresh`/
`mask_thresh`) takes a fixed, order-sensitive 3-channel RGB input. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and, for why each model's channel handling
differs, [`docs/GOTCHAS.md`](docs/GOTCHAS.md) §8.

### Hosting the weights elsewhere

`load()` defaults to the co-located `weights/<model>/` folder, but takes a base URL so you can
serve the ~47 MB of weights from a CDN instead of shipping them with your app:

```js
const cp = await CellposeWebGPU.load(
  "https://huggingface.co/<you>/webgpu-cellseg/resolve/main/cellpose-cyto3/"
);
```

Good options: a **HuggingFace** model repo (free, CORS-enabled, versioned) or **jsDelivr** on
a GitHub tag (CORS + immutable URLs, but it caps GitHub files at ~20 MB — fine for StarDist /
InstanSeg, may reject the 26 MB cyto3 blob). You **can't** point at the upstream model
servers: they serve PyTorch checkpoints (wrong format) and aren't CORS-enabled. See
[`docs/GOTCHAS.md`](docs/GOTCHAS.md) §7.

## Repository layout

```
src/           cellpose.js · stardist.js · instanseg.js   (the engines — MIT)
demo/          index.html (landing) + per-model demos + sample images
weights/       <model>/{weights.bin, manifest.json, LICENSE, NOTICE, SHA256SUMS}
tools/         export_*.py — regenerate the .bin weights from upstream checkpoints
reference/     mini_*.py NumPy oracles + Jupyter walkthroughs + baseline_pytorch.py
tests/         Deno/puppeteer fidelity harnesses + a minimal refdata subset
docs/          GOTCHAS.md · ARCHITECTURE.md
```

## Reproducing the weights

The blobs under `weights/` are repackagings of the upstream pretrained checkpoints (raw
float32 tensors + a JSON manifest of names/offsets/shapes; numeric values unchanged).
Regenerate and verify them from source:

```bash
python tools/export_cellpose.py     # needs cellpose 3.x (Python 3.12 — see GOTCHAS §5)
python tools/export_stardist.py     # needs stardist / tensorflow
python tools/export_instanseg.py    # needs instanseg-torch / pytorch
# then check:
cd weights/cellpose-cyto3 && shasum -c SHA256SUMS
```

## Validation

`tests/` contains the fidelity harnesses that check the WGSL forward + decode against the
NumPy/PyTorch references (AP@0.5 on masks, raw-output diff). A **minimal** refdata subset is
committed so they run out of the box; regenerate the full set with `reference/baseline_pytorch.py`.

```bash
deno run --unstable-webgpu --allow-read tests/cellpose_forward.mjs
deno run --allow-read tests/cellpose_tiling.mjs      # pure-logic tile/taper unit tests
```

## Attribution & licensing

- **Original code** in this repo (WGSL kernels, JS engines, demos, tools, reference ports)
  is **MIT** — see [`LICENSE`](LICENSE).
- **Model weights** under `weights/` are redistributed under their **upstream** licenses,
  which permit redistribution with attribution. Each model folder carries its upstream
  `LICENSE` + a `NOTICE` with provenance and citations:
  - `weights/cellpose-cyto3` — BSD-3-Clause, © Howard Hughes Medical Institute. Cite Stringer
    et al., *Cellpose* (Nat. Methods, 2021).
  - `weights/stardist-fluo` — BSD-3-Clause. Cite Schmidt et al. (MICCAI 2018), Weigert et al.
    (WACV 2020).
  - `weights/instanseg-brightfield` — Apache-2.0. Cite Goldsborough et al., *InstanSeg*.
- **Sample images** in `demo/images/` are from the Cellpose sample dataset and retain
  whatever terms apply to it.

This project is not affiliated with or endorsed by the Cellpose, StarDist, or InstanSeg teams
or their institutions.
