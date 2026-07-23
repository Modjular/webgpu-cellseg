// Headless validation of the JS preprocessing + WebGPU forward pass + JS
// dynamics against the PyTorch reference dumps in refdata/.  Run:
//   deno run --unstable-webgpu --allow-read test_forward.js [name ...]
//
// Names ending in _ch<c1><c2> are the two-channel (cyto+nuclei) references; they
// are picked up automatically from refdata/, no special-casing needed here.
import { CellposeWebGPU, preprocess } from "../src/cellpose.js";

// resolve data paths relative to THIS file, so the test runs from any CWD
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

// Average precision @ IoU threshold (cellpose-style: TP/(TP+FP+FN)), greedy by IoU.
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

const names = Deno.args.length ? Deno.args
  : [...Deno.readDirSync(D("refdata/cellpose"))].filter(e => e.name.endsWith(".meta.json"))
      .map(e => e.name.replace(".meta.json", "")).sort();

const manifest = JSON.parse(Deno.readTextFileSync(D("../weights/cellpose-cyto3/manifest.json")));
const binU8 = Deno.readFileSync(D("../weights/cellpose-cyto3/weights.bin"));
const cp = await CellposeWebGPU.create();
cp.loadWeights(manifest, binU8.buffer);
console.log("loaded", Object.keys(manifest.tensors).length, "weight tensors\n");

let allOk = true;
for (const name of names) {
  const meta = JSON.parse(Deno.readTextFileSync(D(`refdata/cellpose/${name}.meta.json`)));
  const { H, W, Hp, Wp } = meta;
  const input = readF32(D(`refdata/cellpose/${name}.input.bin`));
  const refOut = readF32(D(`refdata/cellpose/${name}.output.bin`));
  const refMasks = readI32(D(`refdata/cellpose/${name}.masks.bin`));

  // Preprocessing: our own normalize99 + reflect-pad, from the raw selected
  // channels, must reproduce the reference net input. The forward check below
  // is fed refdata's input.bin, so without this the JS preprocessing (and the
  // whole two-channel path through it) would never actually be exercised.
  const raw = readF32(D(`refdata/cellpose/${name}.raw.bin`));
  const hasChan2 = Array.isArray(meta.channels) && meta.channels[1] > 0;
  const pre = preprocess(raw.subarray(0, H * W),
    hasChan2 ? raw.subarray(H * W, 2 * H * W) : null, H, W);
  const [pd, parg] = maxAbsDiff(pre.data, input);
  const preOk = pre.Hp === Hp && pre.Wp === Wp && pd < 1e-4;
  allOk = allOk && preOk;

  const t0 = performance.now();
  const { output } = await cp.forwardFromInput(input, Hp, Wp);
  const fwdMs = performance.now() - t0;
  const [md, arg] = maxAbsDiff(output, refOut);
  const fwdOk = md < 2e-2;
  allOk = allOk && fwdOk;

  // dynamics from REFERENCE output (isolates dynamics from forward)
  const dP = new Float32Array(2 * H * W), cprob = new Float32Array(H * W);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    dP[y * W + x] = refOut[y * Wp + x];
    dP[H * W + y * W + x] = refOut[Hp * Wp + y * Wp + x];
    cprob[y * W + x] = refOut[2 * Hp * Wp + y * Wp + x];
  }
  const td = performance.now();
  const masks = cp.computeMasks(dP, cprob, H, W,
    { niter: meta.niter, cellprob_threshold: meta.cellprob_threshold, min_size: meta.min_size,
      flow_threshold: meta.flow_threshold });
  const dynMs = performance.now() - td;
  const nMasks = masks.reduce((m, v) => v > m ? v : m, 0);
  const apRefOut = averagePrecision(refMasks, masks, H, W, 0.5);

  // dynamics from OUR forward output too (end-to-end forward+dynamics)
  const dP2 = new Float32Array(2 * H * W), cprob2 = new Float32Array(H * W);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    dP2[y * W + x] = output[y * Wp + x];
    dP2[H * W + y * W + x] = output[Hp * Wp + y * Wp + x];
    cprob2[y * W + x] = output[2 * Hp * Wp + y * Wp + x];
  }
  const masks2 = cp.computeMasks(dP2, cprob2, H, W,
    { niter: meta.niter, cellprob_threshold: meta.cellprob_threshold, min_size: meta.min_size,
      flow_threshold: meta.flow_threshold });
  const apOurs = averagePrecision(refMasks, masks2, H, W, 0.5);

  console.log(`== ${name} (${W}x${H}) channels=${JSON.stringify(meta.channels ?? [0, 0])} ==`);
  console.log(`  preprocess: max|Δ|=${pd.toExponential(2)} @${parg}  ${preOk ? "OK" : "FAIL"}`);
  console.log(`  forward: max|Δ|=${md.toExponential(2)} @${arg}  ${fwdOk ? "OK" : "FAIL"}  (${fwdMs.toFixed(0)}ms)`);
  console.log(`  dynamics(refOut): masks=${nMasks} vs ref=${meta.n_masks}  AP@0.5=${apRefOut.ap.toFixed(3)} (tp${apRefOut.tp}/fp${apRefOut.fp}/fn${apRefOut.fn})  (${dynMs.toFixed(0)}ms)`);
  console.log(`  end-to-end(ourFwd): AP@0.5=${apOurs.ap.toFixed(3)} (tp${apOurs.tp}/fp${apOurs.fp}/fn${apOurs.fn})\n`);
}
console.log(allOk ? "FORWARD: ALL OK" : "FORWARD: FAIL");
