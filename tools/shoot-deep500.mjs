// shoot-deep500.mjs — drive the ACTUAL viewer to the deep boundary coordinate at
// 2^-400 and 2^-500 and capture what the user really sees: the engine that ran (must
// be gpu-perturb-fe, NOT a CPU fallback, after Spawn 10 lowered GPU_PERTURB_FE_FLOOR to
// 2^-600), the glitch count, the reference length, and the wall-clock time. This is the
// end-to-end integration proof that "zoom to 2^500" now renders on the GPU.
//
//   GPU=1 node tools/shoot-deep500.mjs
import { chromium } from '@playwright/test';
import { chromiumArgs } from './chromium-launch.mjs';
import { spawn } from 'node:child_process';
import { readdirSync, existsSync, mkdirSync } from 'node:fs';
function chrome(){ const d=readdirSync('/nix/store').filter(x=>/-chromium-\d/.test(x)&&!x.includes('sandbox')).sort().reverse(); for(const x of d){const p=`/nix/store/${x}/bin/chromium`; if(existsSync(p))return p;} }
const PORT = process.env.PORT || 8158;
const server = spawn(process.execPath, ['tools/serve.mjs'], { env: { ...process.env, PORT }, stdio: 'ignore' });
const base = `http://127.0.0.1:${PORT}`;
async function ws(){ for(let i=0;i<100;i++){ try{ const r=await fetch(base+'/index.html'); if(r.ok) return; }catch{} await new Promise(r=>setTimeout(r,100)); } throw new Error('no server'); }
mkdirSync('screenshots', { recursive: true });

// Genuine boundary coordinate to ~2^-520 (tools/gen-deep-coord.mjs).
const RE = process.env.RE || '-1.3690780185200794247564219525584239050789875505480606319269707706679171351435598418313845872694681181796577691164703094161056071235175896190526041326788181608562069022572';
const IM = process.env.IM || '-0.0718176768448516592331078176393659885285102748352266772001772456485196902477913914772927891534615241257659389266050933958039395896106078019328906531433514534019106897806';
const radiusFor = (bits) => 1.5 * 2 ** -bits;
const VIEWS = (process.env.BITS || '400,500').split(',').map(Number).map((bits) => ({ tag: `deep${bits}`, bits }));

try {
  await ws();
  const browser = await chromium.launch({ executablePath: chrome(), args: chromiumArgs() });
  const page = await browser.newPage({ viewport: { width: 420, height: 760 }, deviceScaleFactor: 2 });
  page.on('pageerror', (e) => console.error('PAGE ERROR', e.message));
  await page.goto(base + '/index.html');
  await page.waitForFunction(() => (window.__doneCount || 0) > 0, null, { timeout: 30000 });
  const SS = Number(process.env.SS || 2);   // default ss=2 → the depth cap should drop effSS to 1 deep
  await page.evaluate((s) => window.__viewer.setSupersample(s), SS);
  console.log(`renderer driving the viewer; deep boundary coordinate; ss select=${SS}\n`);
  for (const v of VIEWS) {
    const c = await page.evaluate(() => window.__doneCount || 0);
    const t0 = Date.now();
    await page.evaluate((vv) => window.__viewer.setState(vv), { cx: RE, cy: IM, radius: radiusFor(v.bits) });
    let ok = true;
    try { await page.waitForFunction((p) => (window.__doneCount || 0) > p, c, { timeout: 240000 }); }
    catch { ok = false; }
    const ms = Date.now() - t0;
    await page.waitForTimeout(200);
    await page.screenshot({ path: `screenshots/${v.tag}.png` });
    const info = await page.evaluate(() => ({ engine: window.__lastDone && window.__lastDone.engine,
      glitches: window.__lastDone && window.__lastDone.glitches, refLen: window.__lastDone && window.__lastDone.refLen,
      relocations: window.__lastDone && window.__lastDone.relocations, effSS: window.__viewer._effSS,
      maxIter: window.__viewer.maxIter, zoom: +window.__viewer.zoomLevel().toFixed(1), mode: window.__viewer._mode }));
    console.log(`${v.tag} (zoom 2^${v.bits}) ${ok ? 'DONE' : 'TIMEOUT'} ${ms}ms ->`, JSON.stringify(info));
  }
  await browser.close();
} catch (e) { console.error('ERROR', e.stack || e); } finally { server.kill(); }
