// shoot-df64.mjs — render the ACTUAL viewer at a df64-band deep coordinate on the
// selected backend and screenshot it. This is the band that was 90% wrong on real
// NVIDIA before the highp-sampler fix; the shot is the end-to-end visual confirmation
// (covers the COLOR_FRAG uSn highp change too). GPU=1 for the real GPU.
import { chromium } from '@playwright/test';
import { chromiumArgs } from './chromium-launch.mjs';
import { spawn } from 'node:child_process';
import { readdirSync, existsSync, mkdirSync } from 'node:fs';
function chrome(){ const d=readdirSync('/nix/store').filter(x=>/-chromium-\d/.test(x)&&!x.includes('sandbox')).sort().reverse(); for(const x of d){const p=`/nix/store/${x}/bin/chromium`; if(existsSync(p))return p;} }
const PORT = process.env.PORT || 8159;
const server = spawn(process.execPath, ['tools/serve.mjs'], { env: { ...process.env, PORT }, stdio: 'ignore' });
const base = `http://127.0.0.1:${PORT}`;
async function ws(){ for(let i=0;i<100;i++){ try{ const r=await fetch(base+'/index.html'); if(r.ok) return; }catch{} await new Promise(r=>setTimeout(r,100)); } throw new Error('no server'); }
mkdirSync('screenshots', { recursive: true });

// A seahorse-valley deep coordinate, ~2^-50 (df64 band).
const RE = '-0.743643887037158704752191506114774';
const IM = '0.131825904205311970493132056385139';
const RADIUS = 1.5 * 2 ** -50;

try {
  await ws();
  const browser = await chromium.launch({ executablePath: chrome(), args: chromiumArgs() });
  const page = await browser.newPage({ viewport: { width: 420, height: 640 }, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.error('PAGE ERROR', e.message));
  page.on('console', (m) => { const t = m.text(); if (/engine|glitch|gpu|render/i.test(t)) console.log('  [page]', t); });
  await page.goto(base + '/index.html');
  await page.waitForFunction(() => window.__viewer, { timeout: 15000 });
  await page.evaluate(() => { window.__doneCount = 0; });
  await page.evaluate(([re, im, r]) => window.__viewer.setState({ cx: re, cy: im, radius: r, maxIter: 13000 }), [RE, IM, RADIUS]);
  // wait for a 'done' render to land
  await page.waitForFunction(() => (window.__doneCount || 0) > 0, { timeout: 60000 }).catch(() => {});
  await page.waitForTimeout(1500);
  const info = await page.evaluate(() => ({
    engine: (window.__lastDone || {}).engine || '?',
    glitches: (window.__lastDone || {}).glitches,
    zoom: Math.log2(1.5 / window.__viewer.getState().radius).toFixed(2),
  })).catch(() => ({}));
  const mode = process.env.GPU ? `gpu${process.env.GPU}` : 'swift';
  const out = `screenshots/df64_2e50_${mode}.png`;
  await page.screenshot({ path: out });
  console.log(`saved ${out}  engine=${info.engine} zoom=2^${info.zoom} glitches=${info.glitches}`);
  await browser.close();
} finally { server.kill(); }
