// Phase 0 measurement driver — where the time goes, and whether it is recoverable.
//
//   deno run --unstable-webgpu --allow-read --allow-write tools/profile.mjs
//   deno run --unstable-webgpu --allow-read --allow-write tools/profile.mjs --repeats 9
//   deno run --unstable-webgpu --allow-read --allow-write tools/profile.mjs --only ref_075
//
// Runs headless on Deno's WebGPU — no browser, no node_modules — for the same reason the
// fidelity tests do, and it reads the same `tests/refdata/` fixtures so object counts are
// comparable with them.
//
// Writes results/phase0-<date>.json. See docs/PHASE0.md for how to read the output and
// docs/PHASE1.md for what has already been done about it.
import { runPhase0 } from "../src/profile/run.js";

const D = (p) => new URL(p, import.meta.url);
const args = Deno.args;
const flag = (n, d = null) => { const i = args.indexOf(n); return i === -1 ? d : args[i + 1]; };
const has = (n) => args.includes(n);

const REPEATS = Number(flag("--repeats", 5));
const ONLY = flag("--only", null);

const readF32 = (p) => {
  const u8 = Deno.readFileSync(p);
  return new Float32Array(u8.buffer, u8.byteOffset, u8.byteLength / 4);
};

// The fixture is one 240x300 sample, which is the only real-content image in the repo.
// Tile count is what drives most of the interesting behaviour (the taper-blend path, the
// per-tile pipeline drain), and it is set by the *working* resolution — so it is varied
// here by diameter and by tiling the sample, rather than by needing more images.
const FIXTURE = "cellpose_img_075";
const meta = JSON.parse(Deno.readTextFileSync(D(`../tests/refdata/cellpose/${FIXTURE}.meta.json`)));
const raw = readF32(D(`../tests/refdata/cellpose/${FIXTURE}.raw.bin`));
const { H, W } = meta;
const gray = raw.subarray(0, H * W);

/** Repeat the sample n x n times. Synthetic content, but the shapes are what is measured. */
function tiled(n) {
  const H2 = H * n, W2 = W * n;
  const out = new Float32Array(H2 * W2);
  for (let y = 0; y < H2; y++) {
    for (let x = 0; x < W2; x++) out[y * W2 + x] = gray[(y % H) * W + (x % W)];
  }
  return { gray: out, H: H2, W: W2 };
}

const t3 = tiled(3);
const WORKLOADS = [
  { id: "ref_075", gray, H, W, opts: { diameter: 30 },
    why: "the fidelity fixture at its native diameter — real content, comparable mask counts" },
  { id: "ref_075_d15", gray, H, W, opts: { diameter: 15 },
    why: "half the diameter doubles the working resolution, so more tiles at the same input size" },
  { id: "tiled3x3", ...t3, opts: { diameter: 30 },
    why: "the sample tiled 3x3 (720x900) — synthetic content, exercises the multi-tile blend path" },
];

const selected = ONLY ? WORKLOADS.filter((w) => w.id === ONLY) : WORKLOADS;
if (!selected.length) {
  console.error(`no workload "${ONLY}"; known: ${WORKLOADS.map((w) => w.id).join(", ")}`);
  Deno.exit(1);
}

const loadWeights = async () => {
  const manifest = JSON.parse(Deno.readTextFileSync(D("../weights/cellpose-cyto3/manifest.json")));
  const binU8 = Deno.readFileSync(D("../weights/cellpose-cyto3/weights.bin"));
  return { manifest, bin: binU8.buffer };
};

console.log(`→ profiling ${selected.length} workload(s), ${REPEATS} repeats each`);
const report = await runPhase0({ workloads: selected, loadWeights, repeats: REPEATS });

const pct = (x) => x == null ? "—" : (x * 100).toFixed(1) + "%";
const r = report.roofs;
console.log(`\nGPU: ${report.gpu}`);
console.log(`  compute roof   ${r.sweep.peakGflops.toFixed(0)} GFLOP/s (swept plateau)`);
console.log(`  bandwidth roof ${r.bandwidth.gbps.toFixed(0)} GB/s`);
console.log(`  ridge          ${r.ridgeFromSweep} FLOP/byte (swept), ${r.ridgeFromRatio.toFixed(0)} (ratio)`);
console.log(`  wave capacity  ${r.waveCapacity} workgroups — ${r.wave.note}`);
console.log(`  launch floor   ${report.launchFloor.medianNs} ns`);
if (!r.check.ok) console.log(`  !! roof check FAILED (${r.check.computeVsSweepPlateau.toFixed(2)}) — ${r.check.note}`);

for (const res of report.results) {
  const g = res.gates, c = res.classification;
  console.log(`\n─── ${res.workload.id} — ${res.workload.W}×${res.workload.H}, `
    + `${res.stages.tiles ?? 1} tile(s), ${res.objects} objects, ±${pct(res.totalMsSpread)} spread`);
  console.log(`  wall ${g.wallMs.toFixed(0)} ms = GPU ${g.gpuBusyMs.toFixed(0)} (${pct(g.gpuShare)})`
    + `  CPU ${g.cpuMs.toFixed(0)} (${pct(g.cpuShare)})  stall ${g.stallMs.toFixed(0)} (${pct(g.stallShare)})`);
  for (const [k, v] of Object.entries(res.stages)
    .filter(([k]) => k !== "total" && k !== "tiles").sort((a, b) => b[1] - a[1])) {
    console.log(`     ${k.padEnd(14)} ${v.toFixed(1).padStart(8)} ms  ${pct(v / g.wallMs)}`);
  }
  console.log(`  per-tile stall ${g.tileStallMs.toFixed(0)} ms over ${g.tiles ?? 1} tiles`);
  if (res.overlap.medianRatio != null) {
    console.log(`  pass overlap   sum/span median ${res.overlap.medianRatio.toFixed(2)}, `
      + `max ${res.overlap.maxRatio.toFixed(2)} — ${res.overlap.serial ? "serial, per-pass numbers valid" : "OVERLAPPING, per-pass attribution unsound"}`);
  }
  console.log(`  time-weighted attainment ${pct(c.timeWeightedAttainment)} (covering ${pct(c.coverage)} of GPU time)`);
  for (const [k, v] of Object.entries(c.buckets)) {
    if (!v.dispatches) continue;
    console.log(`     ${k.padEnd(26)} ${pct(v.share).padStart(6)}  ${v.dispatches} dispatches`);
  }
  console.log(`  GATE A (Amdahl,  >=${pct(g.gateA.threshold)}): ${g.gateA.pass ? "PASS" : "FAIL"} at ${pct(g.gateA.value)} — ${g.gateA.verdict}`);
  console.log(`  GATE B (headroom,>=${pct(g.gateB.threshold)}): ${g.gateB.pass ? "PASS" : "FAIL"} at ${pct(g.gateB.value)} — ${g.gateB.verdict}`);
}

if (!has("--no-write")) {
  try {
    Deno.mkdirSync(D("../results"), { recursive: true });
    const out = `results/phase0-${new Date().toISOString().slice(0, 10)}.json`;
    Deno.writeTextFileSync(D(`../${out}`), JSON.stringify(report, null, 2));
    console.log(`\nwrote ${out}`);
  } catch (e) {
    console.log(`\n(not written: ${e.message} — pass --allow-write to save results)`);
  }
}
