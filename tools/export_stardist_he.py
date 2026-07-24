"""
export_stardist_he.py — export the pretrained StarDist2D `2D_versatile_he` model
weights into `weights/stardist-he/`, sibling to `weights/stardist-fluo/`.

Unlike export_stardist.py, this needs neither TensorFlow nor the `stardist` package:
the release zip's `weights_best.h5` is a plain Keras HDF5 weights file, and per
config.json the only architectural difference from 2D_versatile_fluo is the first
conv's input channels (n_channel_in=3, RGB, vs 1 for the grayscale fluo model) — same
depth-3 U-Net otherwise. We read the conv datasets directly with h5py and transpose
kernels from Keras [kh,kw,cin,cout] to the [cout,cin,kh,kw] layout the WGSL conv wants,
mirroring export_stardist.py's export_weights().

Tensor tags are written using the *fluo* script's layer names (conv2d, conv2d_1, ...)
rather than this checkpoint's own Keras names (conv2d_1, conv2d_2, ... — an artifact of
Keras auto-naming across separate model instances, in the same layer order) so
stardist.js's forward pass needs no per-model tensor-naming logic.

Run:  python3 tools/export_stardist_he.py
"""
import hashlib
import io
import json
import os
import urllib.request
import zipfile

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "..", "weights", "stardist-he")
URL = "https://github.com/stardist/stardist-models/releases/download/v0.1/python_2D_versatile_he.zip"

CONV_LAYERS = [
    "conv2d", "conv2d_1",
    "down_level_0_no_0", "down_level_0_no_1",
    "down_level_1_no_0", "down_level_1_no_1",
    "down_level_2_no_0", "down_level_2_no_1",
    "middle_0", "middle_2",
    "up_level_2_no_0", "up_level_2_no_2",
    "up_level_1_no_0", "up_level_1_no_2",
    "up_level_0_no_0", "up_level_0_no_2",
    "features", "prob", "dist",
]
# This checkpoint's own Keras layer names, in the same forward order as CONV_LAYERS.
H5_KEYS = [
    "conv2d_1", "conv2d_2",
    "down_level_0_no_0", "down_level_0_no_1",
    "down_level_1_no_0", "down_level_1_no_1",
    "down_level_2_no_0", "down_level_2_no_1",
    "middle_0", "middle_2",
    "up_level_2_no_0", "up_level_2_no_2",
    "up_level_1_no_0", "up_level_1_no_2",
    "up_level_0_no_0", "up_level_0_no_2",
    "features", "prob", "dist",
]

NOTICE = """StarDist 2D_versatile_he — model weights (WebGPU-converted)
============================================================

These files (`weights.bin`, `manifest.json`) are a repackaging of the pretrained
**StarDist `2D_versatile_he`** model into a raw little-endian float32 tensor blob plus
a JSON manifest, for in-browser WebGPU inference. The numeric weights are unchanged from
the upstream checkpoint; only the container format differs.

Upstream model
--------------
- Project:  StarDist (https://github.com/stardist/stardist)
- Model:    2D_versatile_he (bundled pretrained H&E histology nuclei model, RGB input)
- Export:   via `tools/export_stardist_he.py`
- License:  BSD-3-Clause (see the accompanying `LICENSE` file)

License notice (BSD-3-Clause redistribution requirement)
--------------------------------------------------------
Redistribution in source and binary forms is permitted provided that the conditions in
`LICENSE` are met, including reproduction of the copyright notice, the list of conditions,
and the disclaimer. The names of the copyright holders/contributors may not be used to
endorse or promote derived products without specific prior written permission.

Please cite
-----------
Schmidt, U., Weigert, M., Broaddus, C. & Myers, G. Cell Detection with Star-convex
Polygons. MICCAI 2018.
Weigert, M., Schmidt, U., Haase, R., Sugawara, K. & Myers, G. Star-convex Polyhedra for
3D Object Detection and Segmentation in Microscopy. WACV 2020.

Provenance / integrity
----------------------
`SHA256SUMS` records the file hashes. Regenerate from the upstream model with
`tools/export_stardist_he.py` to verify end to end.
"""


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"fetching {URL}")
    with urllib.request.urlopen(URL) as r:
        zdata = r.read()
    zf = zipfile.ZipFile(io.BytesIO(zdata))
    config = json.loads(zf.read("config.json"))
    thresholds = json.loads(zf.read("thresholds.json"))
    assert config["n_channel_in"] == 3, f"expected RGB input, got n_channel_in={config['n_channel_in']}"
    h5bytes = zf.read("weights_best.h5")

    tensors, blob, off = {}, [], 0
    with h5py.File(io.BytesIO(h5bytes), "r") as f:
        for tag, key in zip(CONV_LAYERS, H5_KEYS):
            w = f[f"{key}/{key}/kernel:0"][()]  # [kh,kw,cin,cout]
            b = f[f"{key}/{key}/bias:0"][()]  # [cout]
            w = np.ascontiguousarray(w.transpose(3, 2, 0, 1)).astype("<f4")  # [cout,cin,kh,kw]
            for t, arr in ((f"{tag}.w", w), (f"{tag}.b", b.astype("<f4"))):
                flat = arr.ravel()
                tensors[t] = {"offset": off, "length": int(flat.size), "shape": list(arr.shape)}
                blob.append(flat)
                off += flat.size

    weights_path = os.path.join(OUTDIR, "weights.bin")
    manifest_path = os.path.join(OUTDIR, "manifest.json")
    weights_bytes = np.concatenate(blob).astype("<f4").tobytes()
    with open(weights_path, "wb") as fh:
        fh.write(weights_bytes)
    manifest = {"tensors": tensors, "conv_layers": CONV_LAYERS}
    manifest_text = json.dumps(manifest, indent=1)
    with open(manifest_path, "w") as fh:
        fh.write(manifest_text)

    license_src = os.path.join(HERE, "..", "weights", "stardist-fluo", "LICENSE")
    with open(license_src) as fh:
        license_text = fh.read()
    with open(os.path.join(OUTDIR, "LICENSE"), "w") as fh:
        fh.write(license_text)
    with open(os.path.join(OUTDIR, "NOTICE"), "w") as fh:
        fh.write(NOTICE)

    sums = [
        (hashlib.sha256(weights_bytes).hexdigest(), "weights.bin"),
        (hashlib.sha256(manifest_text.encode()).hexdigest(), "manifest.json"),
    ]
    with open(os.path.join(OUTDIR, "SHA256SUMS"), "w") as fh:
        fh.write("".join(f"{h}  {n}\n" for h, n in sums))

    print(f"wrote {weights_path} ({off * 4 / 1e6:.1f} MB) + {manifest_path} ({len(tensors)} tensors)")
    print(f"thresholds: prob={thresholds['prob']:.4f} nms={thresholds['nms']} (wire these into the UI defaults)")


if __name__ == "__main__":
    main()
