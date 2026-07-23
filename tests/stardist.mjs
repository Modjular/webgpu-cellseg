// Headless validation of the StarDist WebGPU forward + JS polygon-NMS against the
// reference dumps in refdata/sd_*.  Run:
//   deno run --unstable-webgpu --allow-read test_stardist.js [name ...]
//
// The reference prob/dist come from Keras on the same padded input; the reference
// labels come from StarDist's own C-NMS on that same prob/dist (see
// export_stardist.py), so this isolates the WGSL forward and the JS NMS/geometry.
import { StarDistWebGPU, normalize99, padTo16 } from "../src/stardist.js";
const D = (p) => new URL(p, import.meta.url);   // resolve data relative to this file

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
// AP@thr (cellpose-style TP/(TP+FP+FN)), greedy by IoU.
function averagePrecision(gt, pred, H, W, thr = 0.5) {
  const ng = gt.reduce((m, v) => v > m ? v : m, 0);
  const np = pred.reduce((m, v) => v > m ? v : m, 0);
  if (ng === 0 && np === 0) return { ap: 1, tp: 0, fp: 0, fn: 0 };
  const inter = new Map();
  const ag = new Float64Array(ng + 1), ap_ = new Float64Array(np + 1);
  for (let i = 0; i < H * W; i++) {
    const g = gt[i], p = pred[i];
    if (g > 0) ag[g]++;
    if (p > 0) ap_[p]++;
    if (g > 0 && p > 0) { const k = g * (np + 1) + p; inter.set(k, (inter.get(k) || 0) + 1); }
  }
  const pairs = [];
  for (const [k, I] of inter) {
    const g = Math.floor(k / (np + 1)), p = k % (np + 1);
    const iou = I / (ag[g] + ap_[p] - I);
    if (iou > thr) pairs.push([iou, g, p]);
  }
  pairs.sort((a, b) => b[0] - a[0]);
  const gUsed = new Uint8Array(ng + 1), pUsed = new Uint8Array(np + 1);
  let tp = 0;
  for (const [, g, p] of pairs) if (!gUsed[g] && !pUsed[p]) { gUsed[g] = 1; pUsed[p] = 1; tp++; }
  return { ap: tp / (tp + (np - tp) + (ng - tp)), tp, fp: np - tp, fn: ng - tp };
}

const names = Deno.args.length ? Deno.args
  : [...Deno.readDirSync(D("refdata/stardist"))].filter(e => e.name.startsWith("sd_") && e.name.endsWith(".meta.json"))
      .map(e => e.name.replace(".meta.json", "")).sort();

const manifest = JSON.parse(Deno.readTextFileSync(D("../weights/stardist-fluo/manifest.json")));
const binU8 = Deno.readFileSync(D("../weights/stardist-fluo/weights.bin"));
const sd = await StarDistWebGPU.create();
sd.loadWeights(manifest, binU8.buffer);
console.log("loaded", Object.keys(manifest.tensors).length, "weight tensors\n");

let allOk = true;
for (const name of names) {
  const meta = JSON.parse(Deno.readTextFileSync(D(`refdata/stardist/${name}.meta.json`)));
  const { H, W, Hp, Wp, gh, gw, n_rays, prob_thresh, nms_thresh } = meta;
  const input = readF32(D(`refdata/stardist/${name}.input.bin`));      // normalized [H,W]
  const refProb = readF32(D(`refdata/stardist/${name}.prob.bin`));     // [gh,gw]
  const refDist = readF32(D(`refdata/stardist/${name}.dist.bin`));     // [gh,gw,32]
  const refLabels = readI32(D(`refdata/stardist/${name}.labels.bin`)); // [H,W]

  // preprocess: our reflect-pad of the (already normalized) reference input
  const { data, Hp: myHp, Wp: myWp } = padTo16(input, H, W);
  const preOk = myHp === Hp && myWp === Wp;

  const t0 = performance.now();
  const { prob, dist } = await sd.forwardFromInput(data, Hp, Wp);   // prob[gh,gw], dist[32,gh,gw]
  const fwdMs = performance.now() - t0;

  // prob is [gh,gw] like the reference; dist is [32,gh,gw] so transpose the
  // reference [gh,gw,32] for the comparison
  const [pd, parg] = maxAbsDiff(prob, refProb);
  const refDistCHW = new Float32Array(n_rays * gh * gw);
  for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) for (let k = 0; k < n_rays; k++)
    refDistCHW[k * gh * gw + y * gw + x] = refDist[(y * gw + x) * n_rays + k];
  const [dd, darg] = maxAbsDiff(dist, refDistCHW);
  const fwdOk = pd < 1e-3 && dd < 5e-2;
  allOk = allOk && preOk && fwdOk;

  const crop = (full) => {
    const l = new Int32Array(H * W);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) l[y * W + x] = full[y * Wp + x];
    return l;
  };

  // CPU NMS from OUR forward output
  const tc = performance.now();
  const labelsCPU = crop(sd.computeMasks(prob, dist, gh, gw, Hp, Wp, { prob_thresh, nms_thresh }));
  const cpuMs = performance.now() - tc;
  const nCPU = labelsCPU.reduce((m, v) => v > m ? v : m, 0);
  const apCPU = averagePrecision(refLabels, labelsCPU, H, W, 0.5);

  // GPU NMS (sd003) from OUR forward output — the path under test here
  const tg = performance.now();
  const labelsGPU = crop(await sd.computeMasksGPU(prob, dist, gh, gw, Hp, Wp, { prob_thresh, nms_thresh }));
  const gpuMs = performance.now() - tg;
  const nGPU = labelsGPU.reduce((m, v) => v > m ? v : m, 0);
  const apGPU = averagePrecision(refLabels, labelsGPU, H, W, 0.5);
  allOk = allOk && apGPU.ap >= apCPU.ap - 0.01;

  console.log(`== ${name} (${W}x${H} -> grid ${gw}x${gh}) ==`);
  console.log(`  forward: max|Δprob|=${pd.toExponential(2)} @${parg}  max|Δdist|=${dd.toExponential(2)} @${darg}  ${fwdOk ? "OK" : "FAIL"}  (${fwdMs.toFixed(0)}ms)`);
  console.log(`  NMS(CPU): ${nCPU} masks  AP@0.5=${apCPU.ap.toFixed(3)} (tp${apCPU.tp}/fp${apCPU.fp}/fn${apCPU.fn})  (${cpuMs.toFixed(0)}ms)`);
  console.log(`  NMS(GPU): ${nGPU} vs ref=${meta.n_instances}  AP@0.5=${apGPU.ap.toFixed(3)} (tp${apGPU.tp}/fp${apGPU.fp}/fn${apGPU.fn})  (${gpuMs.toFixed(0)}ms)\n`);
}
console.log(allOk ? "STARDIST: ALL OK" : "STARDIST: FAIL");
