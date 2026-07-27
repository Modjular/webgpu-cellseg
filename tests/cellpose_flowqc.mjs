// Equivalence harness for the GPU flow-consistency QC.
//
// `computeMasks()` reconstructs each mask's flow field on the CPU, one mask at a time
// over its own bounding box; `computeMasksGPU()` diffuses every mask at once over the
// whole image (see FLOWDIFF_WGSL in src/cellpose.js for why those are the same
// computation). That step decides which masks survive the flow threshold, so the two
// implementations must produce identical label maps — not merely similar ones, and not
// merely the same *count*, which can coincide while different masks are dropped.
//
// cellpose_forward.mjs only exercises the CPU path, so without this the GPU path — which
// is what segmentImage() actually uses — has no regression cover at all.
//
// Run:
//   deno run --unstable-webgpu --allow-read tests/cellpose_flowqc.mjs [name ...]
import { CellposeWebGPU } from "../src/cellpose.js";

const D = (p) => new URL(p, import.meta.url);
const readF32 = (p) => { const u8 = Deno.readFileSync(p); return new Float32Array(u8.buffer, u8.byteOffset, u8.byteLength / 4); };
const readI32 = (p) => { const u8 = Deno.readFileSync(p); return new Int32Array(u8.buffer, u8.byteOffset, u8.byteLength / 4); };
const maxLabel = (a) => a.reduce((m, v) => v > m ? v : m, 0);

const names = Deno.args.length ? Deno.args
  : [...Deno.readDirSync(D("refdata/cellpose"))].filter((e) => e.name.endsWith(".meta.json"))
      .map((e) => e.name.replace(".meta.json", "")).sort();

const manifest = JSON.parse(Deno.readTextFileSync(D("../weights/cellpose-cyto3/manifest.json")));
const binU8 = Deno.readFileSync(D("../weights/cellpose-cyto3/weights.bin"));
const cp = await CellposeWebGPU.create();
cp.loadWeights(manifest, binU8.buffer);

let allOk = true;
for (const name of names) {
  const meta = JSON.parse(Deno.readTextFileSync(D(`refdata/cellpose/${name}.meta.json`)));
  const { H, W, Hp, Wp } = meta;
  const refOut = readF32(D(`refdata/cellpose/${name}.output.bin`));
  const refMasks = readI32(D(`refdata/cellpose/${name}.masks.bin`));

  // Drive both paths from the reference forward output, so any difference is the flow QC
  // and not the network.
  const dP = new Float32Array(2 * H * W), cprob = new Float32Array(H * W);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    dP[y * W + x] = refOut[y * Wp + x];
    dP[H * W + y * W + x] = refOut[Hp * Wp + y * Wp + x];
    cprob[y * W + x] = refOut[2 * Hp * Wp + y * Wp + x];
  }
  const opts = { niter: meta.niter, cellprob_threshold: meta.cellprob_threshold,
                 min_size: meta.min_size, flow_threshold: meta.flow_threshold };

  const t0 = performance.now();
  const cpuMasks = cp.computeMasks(dP, cprob, H, W, opts);
  const cpuMs = performance.now() - t0;
  const t1 = performance.now();
  const gpuMasks = await cp.computeMasksGPU(dP, cprob, H, W, opts);
  const gpuMs = performance.now() - t1;

  let diff = 0;
  for (let i = 0; i < cpuMasks.length; i++) if (cpuMasks[i] !== gpuMasks[i]) diff++;
  const nCpu = maxLabel(cpuMasks), nGpu = maxLabel(gpuMasks);
  const ok = diff === 0 && nCpu === nGpu;
  allOk = allOk && ok;

  // The flow threshold only matters if it is actually rejecting something — a test where
  // every mask survives would pass no matter how wrong the reconstruction was.
  const beforeQC = cp.computeMasks(dP, cprob, H, W, { ...opts, flow_threshold: 0 });
  const rejected = maxLabel(beforeQC) - nCpu;

  console.log(`== ${name} (${W}x${H}) ==`);
  console.log(`  masks: cpu=${nCpu} gpu=${nGpu} ref=${meta.n_masks}   flow-QC rejected ${rejected}`);
  console.log(`  label maps differ on ${diff}/${cpuMasks.length} pixels  ${ok ? "OK" : "FAIL"}`);
  console.log(`  cpu ${cpuMs.toFixed(0)}ms  gpu ${gpuMs.toFixed(0)}ms  (${(cpuMs / gpuMs).toFixed(2)}x)`);
  if (rejected === 0) console.log("  NOTE: flow QC rejected nothing here — this sample does not exercise the threshold");
  console.log("");
}
console.log(allOk ? "FLOW QC: ALL OK" : "FLOW QC: FAIL");
if (!allOk) Deno.exit(1);
