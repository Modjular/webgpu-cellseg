// Re-tests the f16-shared-memory conv variant against IoU/cell-count agreement with the
// PyTorch reference, instead of the raw per-conv tolerance docs/PHASE1.md rejected it on.
//
// docs/PHASE1.md measured ~2.8e-4 relative / ~1e-2 absolute error on one conv's output and
// rejected the variant because that is ~500x the CB-staging reorder noise and the flow
// field feeds a 200-step Euler integration whose trajectories decide instance boundaries —
// but never actually checked whether that perturbation changes which masks survive. This
// runs the full forward + dynamics pipeline with f16-tile convs swapped in and holds it to
// the same AP@0.5 bar tests/cellpose_forward.mjs uses for the shipping f32 kernel.
//
// Only one reference fixture exists in this repo (cellpose_img_075[_ch23]) — this is not
// the "real image set" docs/PHASE1.md asks for on revisit, just what's available here.
//
// Run:
//   deno run --unstable-webgpu --allow-read tests/cellpose_f16.mjs [name ...]
import { CellposeWebGPU } from "../src/cellpose.js";
import { convWGSL } from "../src/conv-kernel.js";

const D = (p) => new URL(p, import.meta.url);
function readF32(path) {
  const u8 = Deno.readFileSync(path);
  return new Float32Array(u8.buffer, u8.byteOffset, u8.byteLength / 4);
}
function readI32(path) {
  const u8 = Deno.readFileSync(path);
  return new Int32Array(u8.buffer, u8.byteOffset, u8.byteLength / 4);
}
function maxAbsDiff(a, b) {
  let m = 0, arg = -1;
  for (let i = 0; i < a.length; i++) { const d = Math.abs(a[i] - b[i]); if (d > m) { m = d; arg = i; } }
  return [m, arg];
}

// Average precision @ IoU threshold (cellpose-style: TP/(TP+FP+FN)), greedy by IoU. Kept
// identical to tests/cellpose_forward.mjs so the two are comparable.
function averagePrecision(gt, pred, H, W, thr = 0.5) {
  const ng = gt.reduce((m, v) => v > m ? v : m, 0);
  const np = pred.reduce((m, v) => v > m ? v : m, 0);
  if (ng === 0 && np === 0) return { ap: 1, tp: 0, fp: 0, fn: 0, ng, np };
  const inter = new Float64Array((ng + 1) * (np + 1));
  const ag = new Float64Array(ng + 1), ap_ = new Float64Array(np + 1);
  for (let i = 0; i < H * W; i++) {
    const g = gt[i], p = pred[i];
    if (g > 0) ag[g]++;
    if (p > 0) ap_[p]++;
    if (g > 0 && p > 0) inter[g * (np + 1) + p]++;
  }
  const pairs = [];
  for (let g = 1; g <= ng; g++) for (let p = 1; p <= np; p++) {
    const I = inter[g * (np + 1) + p];
    if (I === 0) continue;
    const iou = I / (ag[g] + ap_[p] - I);
    if (iou > thr) pairs.push([iou, g, p]);
  }
  pairs.sort((a, b) => b[0] - a[0]);
  const gUsed = new Uint8Array(ng + 1), pUsed = new Uint8Array(np + 1);
  let tp = 0;
  for (const [, g, p] of pairs) if (!gUsed[g] && !pUsed[p]) { gUsed[g] = 1; pUsed[p] = 1; tp++; }
  const fp = np - tp, fn = ng - tp;
  return { ap: tp / (tp + fp + fn), tp, fp, fn, ng, np };
}

// Same storage-only f16 transform as src/profile/conv-variants.js's V4 (half-width shared
// memory, f32 accumulation, weights untouched) — applied to the kernel that actually
// ships today (BLK=16, 2x2, CB=4) rather than the pre-Phase-1c variant that benchmark
// file is built from, so this measures the same kernel docs/PHASE1.md would be revising.
function f16Variant(K) {
  return "enable f16;\n" + convWGSL(K)
    .replace(/var<workgroup> tile: array<f32,/, "var<workgroup> tile: array<f16,")
    .replace(/var<workgroup> ws:   array<f32,/, "var<workgroup> ws:   array<f16,")
    .replace("      tile[i] = v;", "      tile[i] = f16(v);")
    .replace("      ws[i] = wv;", "      ws[i] = f16(wv);")
    // Widen at the point of use so every FMA below stays f32.
    .replace(/let (v\d+_\d+) = tile\[([^\]]+)\];/g, "let $1 = f32(tile[$2]);")
    .replace(/let (w\d+_\d+) = ws\[([^\]]+)\];/g, "let $1 = f32(ws[$2]);");
}

const names = Deno.args.length ? Deno.args
  : [...Deno.readDirSync(D("refdata/cellpose"))].filter(e => e.name.endsWith(".meta.json"))
      .map(e => e.name.replace(".meta.json", "")).sort();

const manifest = JSON.parse(Deno.readTextFileSync(D("../weights/cellpose-cyto3/manifest.json")));
const binU8 = Deno.readFileSync(D("../weights/cellpose-cyto3/weights.bin"));
const cp = await CellposeWebGPU.create();
cp.loadWeights(manifest, binU8.buffer);

const d = cp.device;
if (!d.features.has("shader-f16")) {
  console.log("shader-f16 not available on this adapter/runtime — cannot re-test. Skipping.");
  Deno.exit(0);
}
const mk = (code) => d.createComputePipeline({
  layout: "auto", compute: { module: d.createShaderModule({ code }), entryPoint: "main" },
});
// Swap in the f16-tile pipelines. Same binding layout and uniform struct as the shipping
// f32 kernel (only the workgroup-local tile/ws arrays change width), so nothing else in
// the engine — bind group creation, weight buffers, uniforms — needs to change.
cp.pConvK = new Map([[1, mk(f16Variant(1))], [3, mk(f16Variant(3))]]);

let allOk = true;
for (const name of names) {
  const meta = JSON.parse(Deno.readTextFileSync(D(`refdata/cellpose/${name}.meta.json`)));
  const { H, W, Hp, Wp } = meta;
  const input = readF32(D(`refdata/cellpose/${name}.input.bin`));
  const refOut = readF32(D(`refdata/cellpose/${name}.output.bin`));
  const refMasks = readI32(D(`refdata/cellpose/${name}.masks.bin`));

  const { output } = await cp.forwardFromInput(input, Hp, Wp);
  const [md, arg] = maxAbsDiff(output, refOut);

  const dP = new Float32Array(2 * H * W), cprob = new Float32Array(H * W);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    dP[y * W + x] = output[y * Wp + x];
    dP[H * W + y * W + x] = output[Hp * Wp + y * Wp + x];
    cprob[y * W + x] = output[2 * Hp * Wp + y * Wp + x];
  }
  const masks = cp.computeMasks(dP, cprob, H, W,
    { niter: meta.niter, cellprob_threshold: meta.cellprob_threshold, min_size: meta.min_size,
      flow_threshold: meta.flow_threshold });
  const nMasks = masks.reduce((m, v) => v > m ? v : m, 0);
  const ap = averagePrecision(refMasks, masks, H, W, 0.5);
  // Cellpose-style IoU over the whole label map, not per-mask — this is what would
  // actually be lost if two label maps have the same count but disagree on boundaries.
  let inter = 0, union = 0;
  for (let i = 0; i < H * W; i++) {
    if (refMasks[i] > 0 || masks[i] > 0) union++;
    if (refMasks[i] > 0 && masks[i] > 0) inter++;
  }
  const pixelIoU = inter / union;

  const ok = ap.ap === 1 && nMasks === meta.n_masks;
  allOk = allOk && ok;

  console.log(`== ${name} (${W}x${H}) ==`);
  console.log(`  forward: max|Δ| vs f32 reference = ${md.toExponential(2)} @${arg}`);
  console.log(`  masks: f16=${nMasks} ref=${meta.n_masks}  AP@0.5=${ap.ap.toFixed(3)} (tp${ap.tp}/fp${ap.fp}/fn${ap.fn})  pixelIoU=${pixelIoU.toFixed(4)}  ${ok ? "OK" : "FAIL"}`);
  console.log("");
}
console.log(allOk ? "F16 RE-TEST: ALL OK (agrees with reference on cell count and IoU)" : "F16 RE-TEST: FAIL (diverges from reference)");
if (!allOk) Deno.exit(1);
