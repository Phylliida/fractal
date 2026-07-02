import { chromium } from '@playwright/test';
import { chromiumArgs } from './chromium-launch.mjs';
import { readdirSync, existsSync } from 'node:fs';
function chrome(){ const d=readdirSync('/nix/store').filter(x=>/-chromium-\d/.test(x)&&!x.includes('sandbox')).sort().reverse(); for(const x of d){const p=`/nix/store/${x}/bin/chromium`; if(existsSync(p))return p;} }
const browser = await chromium.launch({ executablePath: chrome(), args:chromiumArgs() });
const page = await browser.newPage({ viewport:{width:420,height:760}, deviceScaleFactor:2 });
const base = process.env.BASE || 'http://127.0.0.1:8137';
await page.goto(base);
await page.waitForFunction(()=> (window.__doneCount||0)>0, null, {timeout:30000});
await page.waitForTimeout(400);
await page.screenshot({ path:`screenshots/ux-hint.png` });           // first-run hint visible
// dismiss the hint via Escape (capture listener), then show the bare HUD + corner btns
await page.keyboard.press('Escape');
await page.waitForTimeout(350);
await page.screenshot({ path:`screenshots/ux-home2.png` });          // HUD coords + corner buttons
// pan a bit + zoom to confirm coords update
await page.evaluate(()=>window.__viewer.setState({cx:'-0.743643887037158704752191506114774',cy:'0.131825904205311970493132056385139',radius:5e-13,maxIter:8000}));
await page.waitForFunction((p)=> (window.__doneCount||0)>p, await page.evaluate(()=>window.__doneCount||0), {timeout:60000}).catch(()=>{});
await page.waitForTimeout(500);
await page.screenshot({ path:`screenshots/ux-deep-coords.png` });    // deep coord readout
await browser.close();
console.log('done');
