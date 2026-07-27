// Phase 0 measurement run.
//
// One run produces the machine's empirical roofs, a stage-resolved Amdahl split, the
// per-dispatch table with analytical costs, the time-weighted attainable-roof-normalised
// roofline, the four-way classification, and both gate decisions. See docs/PHASE0.md for
// what each of those means and why they are the right things to measure here.
//
// Deliberately environment-agnostic: workloads arrive with their pixels already decoded,
// so this runs under `deno --unstable-webgpu` and in a browser without change. Image
// decoding is the caller's problem (tools/profile.mjs reads the test fixtures).

import { requestDevice, adapterDescription } from "../device.js";
import { CellposeWebGPU } from "../cellpose.js";
import { DispatchRecorder, attach, launchFloor, timestampsWork } from "./timing.js";
import { measureRoofs } from "./roofs.js";
import { annotate, summarise } from "./cost.js";
import { classify, gates } from "./classify.js";

const median = (xs) => {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  return s[s.length >> 1];
};
const spread = (xs) => {
  if (xs.length < 2) return 0;
  const s = [...xs].sort((a, b) => a - b);
  return (s[s.length - 1] - s[0]) / (median(xs) || 1);
};

/**
 * Profile one workload.
 *
 * @param wl { id, why, gray, chan2, H, W, opts }
 *
 * `repeats` runs are timed after a discarded warm-up. Per-dispatch durations are reduced
 * element-wise by position — the dispatch sequence is deterministic, so position is a
 * stable key — and the run-to-run spread travels with the result, because this is
 * usually a passively cooled laptop and thermal drift is a real confound.
 */
export async function profileWorkload(cp, wl, { repeats = 5, waveCapacity = null } = {}) {
  const { gray, chan2 = null, H, W } = wl;
  const opts = { diameter: 30, min_size: 15, ...wl.opts, chan2 };

  cp._stages = {};
  const recorder = new DispatchRecorder(cp.device, { available: await timestampsWork(cp.device) });
  const detach = attach(cp, recorder);

  // Warm-up: shader compilation, buffer-pool fill, GPU clock ramp. Discarded.
  await cp.segmentImage(gray, H, W, opts);
  await recorder.drain();
  recorder.reset();

  const runs = [];
  const overlaps = [];
  for (let i = 0; i < repeats; i++) {
    const res = await cp.segmentImage(gray, H, W, opts);
    const records = await recorder.drain();
    let n = 0;
    for (let k = 0; k < res.labels.length; k++) if (res.labels[k] > n) n = res.labels[k];
    runs.push({ records, stages: { ...res.timings.stages, total: res.timings.total }, objects: n });
    // Read the overlap check before resetting — reset() clears the records it is computed
    // from, and taking it at the end silently produced an empty report.
    overlaps.push(recorder.overlapReport());
    recorder.reset();
  }
  detach();
  cp._stages = null;

  const len = Math.min(...runs.map((r) => r.records.length));
  const dispatches = [];
  for (let i = 0; i < len; i++) {
    const base = runs[0].records[i];
    const nss = runs.map((r) => r.records[i].ns);
    dispatches.push({ ...base, ns: median(nss), nsSpread: spread(nss) });
  }
  const stageKeys = new Set(runs.flatMap((r) => Object.keys(r.stages)));
  const stages = {};
  for (const k of stageKeys) stages[k] = median(runs.map((r) => r.stages[k] || 0));

  const annotated = dispatches.map((d) => annotate(d, { waveCapacity }));
  return {
    workload: { id: wl.id, why: wl.why, W, H, channels: chan2 ? 2 : 1,
                opts: { ...opts, chan2: !!chan2 } },
    objects: runs[0].objects,
    repeats,
    totalMsSpread: spread(runs.map((r) => r.stages.total)),
    stages,
    dispatches: annotated,
    gpuBusyNs: annotated.reduce((s, r) => s + r.ns, 0),
    summary: summarise(annotated),
    overlap: {
      encoders: overlaps.reduce((s, o) => s + o.encoders, 0),
      medianRatio: median(overlaps.map((o) => o.medianRatio).filter((v) => v != null)),
      maxRatio: Math.max(...overlaps.map((o) => o.maxRatio ?? 0)),
      serial: overlaps.every((o) => o.serial !== false),
    },
  };
}

/**
 * Full Phase 0 run: machine roofs once, then every workload.
 *
 * @param loadWeights async () => ({ manifest, bin }) — supplied by the caller because
 *        fetch() and Deno.readFileSync() are not interchangeable.
 */
export async function runPhase0({ workloads, loadWeights, repeats = 5, gateThresholds = {},
                                 onProgress = () => {} } = {}) {
  const step = async (label, fn) => {
    onProgress(label);
    const t0 = performance.now();
    const v = await fn();
    onProgress(`${label} — ${((performance.now() - t0) / 1000).toFixed(1)}s`);
    return v;
  };
  const device = await requestDevice();
  // Probed, not asked: see timestampsWork(). Deno's WebGPU on Metal advertises the
  // feature and returns all-zero timestamps, which would produce an empty profile rather
  // than an error. Run the full Phase 0 in a browser (demo/profile.html); Deno is fine
  // for the fidelity tests and for tools/convbench.mjs, which falls back to wall clock.
  if (!(await timestampsWork(device))) {
    throw new Error(
      "timestamp queries do not work on this device (the feature may be advertised but "
      + "return zeros — Deno's WebGPU on Metal does this). Per-dispatch attribution is "
      + "impossible here; open demo/profile.html in Chrome instead.");
  }

  const roofs = await step("measuring machine roofs (compute, bandwidth, ridge, wave capacity)", () => measureRoofs(device));
  const floor = await step("measuring launch floor", () => launchFloor(device));

  const cp = new CellposeWebGPU(device);
  const { manifest, bin } = await step("loading weights", loadWeights).then(async (w) => w);
  cp.loadWeights(manifest, bin);

  const results = [];
  for (const wl of workloads) {
    const r = await step(`profiling ${wl.id} (${repeats} repeats + warm-up)`,
      () => profileWorkload(cp, wl, { repeats, waveCapacity: roofs.waveCapacity }));
    const cls = classify(r.dispatches, roofs, floor, gateThresholds);
    results.push({
      ...r,
      dispatches: cls.rows,
      classification: {
        buckets: cls.buckets, causes: cls.causes,
        timeWeightedAttainment: cls.timeWeightedAttainment,
        coverage: cls.coverage, recoverableShare: cls.recoverableShare,
      },
      gates: gates(r.stages, r.gpuBusyNs, cls, gateThresholds),
    });
  }

  return {
    generated: new Date().toISOString(),
    gpu: adapterDescription(device),
    roofs, launchFloor: floor,
    results,
  };
}
