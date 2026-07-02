// smoke-viewer.mjs — load the real app, exercise the GPU path at several depths,
// confirm the engine chosen, capture screenshots, and verify GPU vs forced-CPU
// renders agree (catches orientation flips and gross errors).
import { chromium } from '@playwright/test';
import { chromiumArgs } from './chromium-launch.mjs';
import { spawn } from 'node:child_process';
import { readdirSync, existsSync, mkdirSync } from 'node:fs';
function resolveChromium() {
  if (process.env.CHROMIUM_PATH && existsSync(process.env.CHROMIUM_PATH)) return process.env.CHROMIUM_PATH;
  const dirs = readdirSync('/nix/store').filter((d) => /-chromium-\d/.test(d) && !d.includes('sandbox')).sort().reverse();
  for (const d of dirs) { const p = `/nix/store/${d}/bin/chromium`; if (existsSync(p)) return p; }
}
const PORT = process.env.PORT || 8151;
const server = spawn(process.execPath, ['tools/serve.mjs'], { env: { ...process.env, PORT }, stdio: 'ignore' });
const base = `http://127.0.0.1:${PORT}`;
async function ws() { for (let i = 0; i < 100; i++) { try { const r = await fetch(base + '/index.html'); if (r.ok) return; } catch {} await new Promise((r) => setTimeout(r, 100)); } throw new Error('no server'); }

mkdirSync('screenshots', { recursive: true });
let fail = 0;
try {
  await ws();
  const browser = await chromium.launch({ executablePath: resolveChromium(),
    args: chromiumArgs() });
  const page = await browser.newPage({ viewport: { width: 420, height: 760 } });
  page.on('pageerror', (e) => { console.error('PAGE ERROR:', e.message); fail++; });
  page.on('console', (m) => { if (m.type() === 'error') console.error('console.error:', m.text()); });
  await page.goto(base + '/index.html');

  // helper: render a view and return {engine, pixelsMeanLuma, fp}
  async function renderView(state, label) {
    const prev = await page.evaluate(() => window.__doneCount || 0);
    await page.evaluate((s) => window.__viewer.setState(s), state);
    await page.waitForFunction((p) => (window.__doneCount || 0) > p, prev, { timeout: 45000 });
    const info = await page.evaluate(() => ({ engine: window.__lastDone.engine, ...window.__viewer.getState() }));
    await page.screenshot({ path: `screenshots/gpu_${label}.png` });
    return info;
  }

  // 1) home -> gpu-naive
  await page.waitForFunction(() => (window.__doneCount || 0) > 0, { timeout: 20000 });
  let home = await page.evaluate(() => window.__lastDone.engine);
  console.log(`home engine: ${home}  ${home === 'gpu-naive' ? 'OK' : 'FAIL'}`);
  if (home !== 'gpu-naive') fail++;
  await page.screenshot({ path: 'screenshots/gpu_home.png' });

  // 2) seahorse 2^-41 -> gpu-perturb
  const sea = await renderView({ cx: '-0.743643887037158704752191506114774', cy: '0.131825904205311970493132056385139', radius: 5e-13 }, 'seahorse');
  console.log(`seahorse engine: ${sea.engine} zoom 2^${sea.zoom.toFixed(1)}  ${sea.engine === 'gpu-perturb' ? 'OK' : 'FAIL'}`);
  if (sea.engine !== 'gpu-perturb') fail++;

  // 3) deep, below the df64 float32-exponent floor (2^-112) -> GPU perturb FLOATEXP.
  // Supersampling off + an escaping exterior point so the (heavy) fe shader stays fast
  // under SwiftShader and renders a non-black image; we assert the engine dispatches.
  await page.evaluate(() => window.__viewer.setSupersample(1));
  const deep = await renderView({ cx: '0.36', cy: '0.09', radius: 1.5 * 2 ** -150, maxIter: 1500 }, 'deepfe');
  console.log(`deep fe(2^-150) engine: ${deep.engine} zoom 2^${deep.zoom.toFixed(1)}  ${deep.engine === 'gpu-perturb-fe' ? 'OK' : 'FAIL'}`);
  if (deep.engine !== 'gpu-perturb-fe') fail++;

  // 4) GPU vs forced-CPU parity at a mid zoom (catches orientation flip / gross error)
  const parity = await page.evaluate(async () => {
    const V = window.__viewer;
    function snap() { const c = document.getElementById('view'); const g = c.getContext('2d');
      return g.getImageData(0, 0, c.width, c.height); }
    async function renderAndWait(fn) { const p = window.__doneCount || 0; fn();
      await new Promise((res) => { const t = setInterval(() => { if ((window.__doneCount || 0) > p) { clearInterval(t); res(); } }, 30); }); }
    // GPU render at a seahorse-ish 2^-20
    await renderAndWait(() => V.setState({ cx: '-0.745', cy: '0.113', radius: 1.5 * 2 ** -20 }));
    const gpuEngine = window.__lastDone.engine;
    const a = snap();
    // force CPU and re-render same view
    V.forceCpu = true; V._gpuChecked = false; V.gpu = null;
    await renderAndWait(() => V.render());
    const cpuEngine = window.__lastDone.engine;
    const b = snap();
    V.forceCpu = false; V._gpuChecked = false; V.gpu = null;
    // mean abs luma diff
    let sum = 0, n = a.data.length / 4;
    for (let i = 0; i < a.data.length; i += 4) {
      const la = (a.data[i] + a.data[i + 1] + a.data[i + 2]) / 3;
      const lb = (b.data[i] + b.data[i + 1] + b.data[i + 2]) / 3;
      sum += Math.abs(la - lb);
    }
    return { gpuEngine, cpuEngine, meanLumaDiff: sum / n, w: a.width, h: a.height };
  });
  console.log(`parity @2^-20: gpu=${parity.gpuEngine} cpu=${parity.cpuEngine} meanLumaDiff=${parity.meanLumaDiff.toFixed(2)} (${parity.w}x${parity.h})`);
  // orientation flip or gross error -> large diff. Boundary precision -> small.
  if (parity.meanLumaDiff > 12) { console.log('  FAIL: GPU and CPU renders differ too much (flip or bug?)'); fail++; }
  else console.log('  OK: GPU and CPU renders agree (boundary-level diff only)');

  await browser.close();
} catch (e) { console.error('ERROR:', e.stack || e); fail++; } finally { server.kill(); }
console.log(fail ? `\nSMOKE FAILURES: ${fail}` : '\nSMOKE OK');
process.exit(fail ? 1 : 0);
