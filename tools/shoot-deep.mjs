// shoot-deep.mjs — render the ACTUAL viewer at the deep "ultra" coordinate at
// several depths around/past 2^218 and capture what the user actually sees, plus
// the engine that ran, the glitch count, and the timing. This is the direct check
// of the "can't zoom past 2^218" report.
import { chromium } from '@playwright/test';
import { chromiumArgs } from './chromium-launch.mjs';
import { spawn } from 'node:child_process';
import { readdirSync, existsSync, mkdirSync } from 'node:fs';
function chrome(){ const d=readdirSync('/nix/store').filter(x=>/-chromium-\d/.test(x)&&!x.includes('sandbox')).sort().reverse(); for(const x of d){const p=`/nix/store/${x}/bin/chromium`; if(existsSync(p))return p;} }
const PORT = process.env.PORT || 8157;
const server = spawn(process.execPath, ['tools/serve.mjs'], { env: { ...process.env, PORT }, stdio: 'ignore' });
const base = `http://127.0.0.1:${PORT}`;
async function ws(){ for(let i=0;i<100;i++){ try{ const r=await fetch(base+'/index.html'); if(r.ok) return; }catch{} await new Promise(r=>setTimeout(r,100)); } throw new Error('no server'); }
mkdirSync('screenshots', { recursive: true });

const RE = '-1.369078017863660784890619576747781310848768032841633323730495873496232879296538490243106365484246242476783355722';
const IM = '-0.071817675972918479944583194368632476442138106251769795140812120871593742404751576456750164324645880810732436640';
// radius for a target zoom level 2^bits:  zoom = log2(1.5/radius)
const radiusFor = (bits) => 1.5 * 2 ** -bits;
const VIEWS = [
  { tag: 'deep190', bits: 190 },
  { tag: 'deep218', bits: 218 },
  { tag: 'deep271', bits: 271 },
];

try {
  await ws();
  const browser = await chromium.launch({ executablePath: chrome(), args:chromiumArgs() });
  const page = await browser.newPage({ viewport:{width:420,height:760}, deviceScaleFactor:2 });
  page.on('pageerror', e=>console.error('PAGE ERROR', e.message));
  await page.goto(base + '/index.html');
  await page.waitForFunction(()=> (window.__doneCount||0)>0, null, {timeout:30000});
  // keep supersampling off so a deep SwiftShader draw stays within timeout
  await page.evaluate(()=>window.__viewer.setSupersample(1));
  for (const v of VIEWS) {
    const c = await page.evaluate(()=>window.__doneCount||0);
    const view = { cx: RE, cy: IM, radius: radiusFor(v.bits) };
    const t0 = Date.now();
    await page.evaluate((vv)=>window.__viewer.setState(vv), view);
    let ok = true;
    try { await page.waitForFunction(p=> (window.__doneCount||0)>p, c, {timeout:120000}); }
    catch { ok = false; }
    const ms = Date.now() - t0;
    await page.waitForTimeout(200);
    await page.screenshot({ path:`screenshots/${v.tag}.png` });
    const info = await page.evaluate(()=>({ engine: window.__lastDone && window.__lastDone.engine,
      glitches: window.__lastDone && window.__lastDone.glitches, refLen: window.__lastDone && window.__lastDone.refLen,
      maxIter: window.__viewer.maxIter, zoom: +window.__viewer.zoomLevel().toFixed(1), mode: window.__viewer._mode }));
    console.log(`${v.tag} (2^${v.bits}) ${ok?'DONE':'TIMEOUT'} ${ms}ms ->`, JSON.stringify(info));
  }
  await browser.close();
} catch (e) { console.error('ERROR', e.stack||e); } finally { server.kill(); }
