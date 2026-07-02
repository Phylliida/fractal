// shoot-ss.mjs — capture the SAME view at supersampling Off vs 4× to show the
// anti-aliasing the supersampling adds (run after `npm run serve` on PORT/BASE).
import { chromium } from '@playwright/test';
import { chromiumArgs } from './chromium-launch.mjs';
import { spawn } from 'node:child_process';
import { readdirSync, existsSync, mkdirSync } from 'node:fs';
function chrome(){ const d=readdirSync('/nix/store').filter(x=>/-chromium-\d/.test(x)&&!x.includes('sandbox')).sort().reverse(); for(const x of d){const p=`/nix/store/${x}/bin/chromium`; if(existsSync(p))return p;} }
const PORT = process.env.PORT || 8155;
const server = spawn(process.execPath, ['tools/serve.mjs'], { env: { ...process.env, PORT }, stdio: 'ignore' });
const base = `http://127.0.0.1:${PORT}`;
async function ws(){ for(let i=0;i<100;i++){ try{ const r=await fetch(base+'/index.html'); if(r.ok) return; }catch{} await new Promise(r=>setTimeout(r,100)); } throw new Error('no server'); }
mkdirSync('screenshots', { recursive: true });
try {
  await ws();
  const browser = await chromium.launch({ executablePath: chrome(), args:chromiumArgs() });
  const page = await browser.newPage({ viewport:{width:420,height:760}, deviceScaleFactor:2 });
  page.on('pageerror', e=>console.error('PAGE ERROR', e.message));
  await page.goto(base + '/index.html');
  await page.waitForFunction(()=> (window.__doneCount||0)>0, null, {timeout:30000});
  const view = { cx:'-0.743643887037158704752191506114774', cy:'0.131825904205311970493132056385139', radius:5e-13, maxIter:2500 };
  for (const ss of [1, 2]) {
    const c = await page.evaluate(()=>window.__doneCount||0);
    await page.evaluate((s)=>window.__viewer.setSupersample(s), ss);
    await page.evaluate((v)=>window.__viewer.setState(v), view);
    await page.waitForFunction(p=> (window.__doneCount||0)>p, c, {timeout:90000});
    await page.waitForTimeout(300);
    await page.screenshot({ path:`screenshots/ss_${ss}x.png` });
    const info = await page.evaluate(()=>({ engine: window.__lastDone.engine, effSS: window.__viewer._effSS, cW: window.__viewer.cW, cH: window.__viewer.cH }));
    console.log(`ss=${ss}x ->`, JSON.stringify(info));
  }
  await browser.close();
} catch (e) { console.error('ERROR', e.stack||e); } finally { server.kill(); }
