// Conv kernel benchmark — the tuning loop behind src/conv-kernel.js.
//
//   deno run --unstable-webgpu --allow-read --allow-write tools/convbench.mjs
//   deno run --unstable-webgpu --allow-read tools/convbench.mjs --shapes 4 --per-shape
//   deno run --unstable-webgpu --allow-read tools/convbench.mjs --only baseline,shipping
//
// Runs one conv layer at a time against every kernel variant, checks each variant's
// output against the frozen pre-optimisation kernel, and reports GFLOP/s against a roof
// measured in the same session. Layer shapes are the real cyto3 ones, weighted by their
// measured share of conv time, so a variant's aggregate score predicts what it will do to
// the whole pipeline rather than flattering it on one convenient size.
//
// The four constants in src/conv-kernel.js are a *joint* optimum — they trade against
// each other through registers and threadgroup memory — so re-run the full sweep after
// changing any of them, not just the one you touched. See docs/PHASE1.md.
import { requestDevice } from "../src/device.js";
import { benchVariants, SHAPES } from "../src/profile/convbench.js";
import { VARIANTS } from "../src/profile/conv-variants.js";
import { intensitySweep } from "../src/profile/roofs.js";
import { timestampsWork } from "../src/profile/timing.js";

const D = (p) => new URL(p, import.meta.url);
const args = Deno.args;
const flag = (n, d = null) => { const i = args.indexOf(n); return i === -1 ? d : args[i + 1]; };
const has = (n) => args.includes(n);

const NSHAPES = flag("--shapes", null);
const ONLY = flag("--only", null);

const device = await requestDevice();

// Absolute attainment needs trustworthy per-dispatch timing. Where that is unavailable
// (Deno's WebGPU on Metal — see timestampsWork) everything falls back to wall clock:
// variant-vs-variant speedups remain valid, absolute GFLOP/s and attainment do not, so
// they are suppressed rather than printed as a number nobody should act on.
const timed = await timestampsWork(device);
const sweep = await intensitySweep(device);
const roofGflops = sweep.peakGflops;
if (!timed) {
  console.log("\n! timestamp queries unusable here — timing by batched wall clock.");
  console.log("  Variant comparisons are sound; absolute attainment is approximate.");
  console.log("  For the per-dispatch profile, open demo/profile.html in Chrome.");
}

let variants = VARIANTS;
if (ONLY) {
  const want = ONLY.split(",");
  variants = VARIANTS.filter((v) => want.includes(v.id));
  // The first variant is the correctness reference and the speedup denominator, so it has
  // to be present whatever the filter says.
  if (!variants.some((v) => v.id === VARIANTS[0].id)) variants.unshift(VARIANTS[0]);
}
const shapes = NSHAPES ? SHAPES.slice(0, Number(NSHAPES)) : SHAPES;

// Compile everything first so a syntax error is one clear message rather than a device
// error mid-benchmark. Pipeline *limits* are not caught here — they fail through async
// validation and are reported per variant by benchVariants.
const compileErrors = [];
for (const v of variants) {
  const sources = typeof v.wgsl === "function"
    ? [...new Set(shapes.filter((s) => !v.applicable || v.applicable(s)).map((s) => v.wgsl(s)))]
    : [v.wgsl];
  for (const code of sources) {
    const info = await device.createShaderModule({ code, label: v.id }).getCompilationInfo();
    for (const m of info.messages) {
      if (m.type === "error") {
        compileErrors.push(`${v.id}:${m.lineNum}:${m.linePos}: ${m.message}\n      ${(code.split("\n")[m.lineNum - 1] || "").trim()}`);
      }
    }
  }
}
if (compileErrors.length) {
  console.error("shader compilation failed:");
  for (const e of compileErrors) console.error("  " + e);
  Deno.exit(1);
}

const out = await benchVariants(device, variants, { shapes, roofGflops });

const pct = (x) => x == null ? "—" : (x * 100).toFixed(2) + "%";
console.log(roofGflops ? `\nroof ${roofGflops.toFixed(0)} GFLOP/s (measured this session)\n` : "\nwall-clock timing; attainment suppressed\n");
console.log("variant".padEnd(20) + "ok".padEnd(6) + "GFLOP/s".padStart(9)
  + "attain".padStart(9) + "proj speedup".padStart(14) + "  what");
for (const s of out.summary) {
  if (s.unavailable) {
    console.log(s.variant.padEnd(20) + "n/a".padEnd(6) + "—".padStart(9) + "—".padStart(9)
      + "—".padStart(14) + "  " + s.what);
    console.log("".padEnd(20) + "   " + s.unavailable);
    continue;
  }
  console.log(
    s.variant.padEnd(20)
    + (s.allOk ? "ok" : "FAIL").padEnd(6)
    + s.weightedGflops.toFixed(0).padStart(9)
    + pct(s.weightedAttainment).padStart(9)
    + (s.projectedSpeedup.toFixed(2) + "x").padStart(14)
    + "  " + s.what);
  if (!s.allOk) console.log("".padEnd(20) + "   failing shapes: " + s.failures.join(", "));
}

if (has("--per-shape")) {
  console.log("\nper shape:");
  for (const sk of [...new Set(out.rows.map((r) => JSON.stringify(r.shape)))]) {
    const s = JSON.parse(sk);
    console.log(`\n  ${s.Cin}->${s.Cout} @ ${s.H}x${s.W} k${s.K}  (${(s.share * 100).toFixed(1)}% of conv time)`);
    for (const r of out.rows.filter((r) => JSON.stringify(r.shape) === sk)) {
      console.log("    " + r.variant.padEnd(20)
        + (r.ns / 1e6).toFixed(3).padStart(9) + " ms"
        + r.gflops.toFixed(0).padStart(8) + " GFLOP/s"
        + pct(r.attainment).padStart(9)
        + (r.speedup.toFixed(2) + "x").padStart(9)
        + (r.ok ? "" : `   FAIL relerr=${r.maxRel.toExponential(1)}`));
    }
  }
}

if (!has("--no-write")) {
  try {
    Deno.mkdirSync(D("../results"), { recursive: true });
    const f = `results/convbench-${new Date().toISOString().slice(0, 10)}.json`;
    Deno.writeTextFileSync(D(`../${f}`), JSON.stringify({ roofGflops, ...out }, null, 2));
    console.log(`\nwrote ${f}`);
  } catch (e) {
    console.log(`\n(not written: ${e.message} — pass --allow-write to save results)`);
  }
}
