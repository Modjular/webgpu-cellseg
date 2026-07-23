// Unit-test the pure-JS tiling helpers against cellpose's authoritative output
// (refdiag/tiling_ref.json + taper_*.f32.bin, from tiling_ref.py).
// Run: deno run --allow-read test_tiling.mjs
import { getPadYX, tileGrid, taperMask } from "../src/cellpose.js";

const D = (p) => new URL(p, import.meta.url);   // resolve refdata relative to this file
const ref = JSON.parse(await Deno.readTextFile(D("refdata/tiling/tiling_ref.json")));
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
let fails = 0;
const check = (name, got, want) => {
  const ok = eq(got, want);
  if (!ok) { fails++; console.log(`  FAIL ${name}\n    got  ${JSON.stringify(got)}\n    want ${JSON.stringify(want)}`); }
  return ok;
};

for (const c of ref.cases) {
  const tag = `${c.Lyr}x${c.Lxr}`;
  const [ypad1, ypad2, xpad1, xpad2] = getPadYX(c.Lyr, c.Lxr, 16, 1);
  check(`${tag} pad`, [ypad1, ypad2, xpad1, xpad2], c.pad);
  const Ly = c.Lyr + ypad1 + ypad2, Lx = c.Lxr + xpad1 + xpad2;
  check(`${tag} Ly,Lx`, [Ly, Lx], [c.Ly, c.Lx]);

  const g = tileGrid(Ly, Lx, ref.bsize, ref.tile_overlap);
  check(`${tag} ny,nx`, [g.ny, g.nx], [c.ny, c.nx]);
  check(`${tag} tile_shape`, [g.bsizeY, g.bsizeX], c.tile_shape);
  check(`${tag} ntiles`, g.ny * g.nx, c.ntiles);

  // reconstruct ysub/xsub in row-major (j outer, i inner) exactly as make_tiles
  const ysub = [], xsub = [];
  for (let j = 0; j < g.ny; j++) for (let i = 0; i < g.nx; i++) {
    ysub.push([g.ystart[j], g.ystart[j] + g.bsizeY]);
    xsub.push([g.xstart[i], g.xstart[i] + g.bsizeX]);
  }
  check(`${tag} ysub`, ysub, c.ysub);
  check(`${tag} xsub`, xsub, c.xsub);
}

// taper masks
for (const t of ref.taper) {
  const raw = new Float32Array((await Deno.readFile(D(`refdata/tiling/taper_${t.ly}x${t.lx}.f32.bin`))).buffer);
  const got = taperMask(t.ly, t.lx);
  let maxd = 0;
  for (let i = 0; i < raw.length; i++) maxd = Math.max(maxd, Math.abs(got[i] - raw[i]));
  const ok = maxd < 1e-6;
  if (!ok) fails++;
  console.log(`  taper ${t.ly}x${t.lx}: max|diff| = ${maxd.toExponential(3)}  ${ok ? "ok" : "FAIL"}`);
}

console.log(fails === 0 ? "\nALL TILING TESTS PASSED" : `\n${fails} FAILURES`);
if (fails) Deno.exit(1);
