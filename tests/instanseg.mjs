// Headless validation of the InstanSeg WebGPU forward + JS decode against the
// reference dumps in refdata/is_*.  Run:
//   deno run --unstable-webgpu --allow-read test_instanseg.js [name ...]
import { InstanSegWebGPU } from "../src/instanseg.js";
const D = (p) => new URL(p, import.meta.url);   // resolve data relative to this file

function readF32(p) { const u8 = Deno.readFileSync(p); return new Float32Array(u8.buffer, u8.byteOffset, u8.byteLength / 4); }
function readI32(p) { const u8 = Deno.readFileSync(p); return new Int32Array(u8.buffer, u8.byteOffset, u8.byteLength / 4); }
function maxAbsDiff(a, b, n) { let m = 0, arg = -1; for (let i = 0; i < n; i++) { const d = Math.abs(a[i] - b[i]); if (d > m) { m = d; arg = i; } } return [m, arg]; }

function averagePrecision(gt, pred, H, W, thr = 0.5) {
  // count instances by DISTINCT labels present (InstanSeg's reference labels are
  // not compact — max id > number of instances), matching mini_instanseg's metric.
  const ng = gt.reduce((m, v) => v > m ? v : m, 0), np = pred.reduce((m, v) => v > m ? v : m, 0);
  if (ng === 0 && np === 0) return { ap: 1, tp: 0, fp: 0, fn: 0 };
  const inter = new Map(), ag = new Float64Array(ng + 1), ap_ = new Float64Array(np + 1);
  for (let i = 0; i < H * W; i++) { const a = gt[i], b = pred[i]; if (a) ag[a]++; if (b) ap_[b]++;
    if (a && b) { const k = a * (np + 1) + b; inter.set(k, (inter.get(k) || 0) + 1); } }
  let nG = 0, nP = 0; for (let i = 1; i <= ng; i++) if (ag[i]) nG++; for (let i = 1; i <= np; i++) if (ap_[i]) nP++;
  const pairs = [];
  for (const [k, I] of inter) { const a = Math.floor(k / (np + 1)), b = k % (np + 1); const iou = I / (ag[a] + ap_[b] - I); if (iou > thr) pairs.push([iou, a, b]); }
  pairs.sort((x, y) => y[0] - x[0]);
  const gu = new Uint8Array(ng + 1), pu = new Uint8Array(np + 1); let tp = 0;
  for (const [, a, b] of pairs) if (!gu[a] && !pu[b]) { gu[a] = pu[b] = 1; tp++; }
  return { ap: tp / (tp + (nP - tp) + (nG - tp)), tp, fp: nP - tp, fn: nG - tp };
}

function padClamp(norm, H, W) {   // normalized [3,H,W] -> padded/8, clamped [-2,3]
  const div = 8, Hp = Math.ceil(H / div) * div, Wp = Math.ceil(W / div) * div;
  const d = new Float32Array(3 * Hp * Wp);
  for (let c = 0; c < 3; c++) { const si = c * H * W, di = c * Hp * Wp;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) d[di + y * Wp + x] = norm[si + y * W + x];
    for (let y = H; y < Hp; y++) for (let x = 0; x < W; x++) d[di + y * Wp + x] = d[di + Math.max(0, 2 * H - 1 - y) * Wp + x];
    for (let y = 0; y < Hp; y++) for (let x = W; x < Wp; x++) d[di + y * Wp + x] = d[di + y * Wp + Math.max(0, 2 * W - 1 - x)];
  }
  for (let i = 0; i < d.length; i++) d[i] = Math.min(3, Math.max(-2, d[i]));
  return { d, Hp, Wp };
}

const names = Deno.args.length ? Deno.args
  : [...Deno.readDirSync(D("refdata/instanseg"))].filter(e => e.name.startsWith("is_") && e.name.endsWith(".meta.json"))
      .map(e => e.name.replace(".meta.json", "")).sort();

const manifest = JSON.parse(Deno.readTextFileSync(D("../weights/instanseg-brightfield/manifest.json")));
const is = await InstanSegWebGPU.create();
is.loadWeights(manifest, Deno.readFileSync(D("../weights/instanseg-brightfield/weights.bin")).buffer);
console.log("loaded", Object.keys(manifest.tensors).length, "weight tensors\n");

let allOk = true;
for (const name of names) {
  const meta = JSON.parse(Deno.readTextFileSync(D(`refdata/instanseg/${name}.meta.json`)));
  const { H, W } = meta;
  const input = readF32(D(`refdata/instanseg/${name}.input.bin`));          // [3,H,W] normalized
  const refOut = readF32(D(`refdata/instanseg/${name}.output.bin`));         // [5,H,W]
  const refLabels = readI32(D(`refdata/instanseg/${name}.labels.bin`));

  const { d, Hp, Wp } = padClamp(input, H, W);
  const t0 = performance.now();
  const { out } = await is.forwardFromInput(d, Hp, Wp);         // [5,Hp,Wp]
  const fwdMs = performance.now() - t0;
  // compare cropped [5,H,W]
  const HWp = Hp * Wp;
  let md = 0, arg = -1;
  for (let c = 0; c < 5; c++) for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const dif = Math.abs(out[c * HWp + y * Wp + x] - refOut[(c * H + y) * W + x]);
    if (dif > md) { md = dif; arg = c; }
  }
  const fwdOk = md < 2e-2; allOk = allOk && fwdOk;

  const td = performance.now();
  const labels = is.computeMasks(out, Hp, Wp, H, W, {});
  const decMs = performance.now() - td;
  const n = labels.reduce((m, v) => v > m ? v : m, 0);
  const ap = averagePrecision(refLabels, labels, H, W, 0.5);
  allOk = allOk && ap.ap >= 0.95;

  console.log(`== ${name} (${W}x${H}) ==`);
  console.log(`  forward: max|Δ|=${md.toExponential(2)} (ch${arg})  ${fwdOk ? "OK" : "FAIL"}  (${fwdMs.toFixed(0)}ms)`);
  console.log(`  masks: ${n} vs ref=${meta.n_instances}  AP@0.5=${ap.ap.toFixed(3)} (tp${ap.tp}/fp${ap.fp}/fn${ap.fn})  (${decMs.toFixed(0)}ms)\n`);
}
console.log(allOk ? "INSTANSEG: ALL OK" : "INSTANSEG: FAIL");
