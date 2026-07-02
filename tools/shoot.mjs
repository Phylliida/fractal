import { chromium } from '@playwright/test';
import { chromiumArgs } from './chromium-launch.mjs';
import { readdirSync, existsSync } from 'node:fs';
function chrome(){ const d=readdirSync('/nix/store').filter(x=>/-chromium-\d/.test(x)&&!x.includes('sandbox')).sort().reverse(); for(const x of d){const p=`/nix/store/${x}/bin/chromium`; if(existsSync(p))return p;} }
const browser = await chromium.launch({ executablePath: chrome(), args:chromiumArgs() });
const page = await browser.newPage({ viewport:{width:420,height:760}, deviceScaleFactor:2 });
const base = process.env.BASE || 'http://127.0.0.1:8137';
async function shot(name, setup){
  await page.goto(base);
  await page.waitForFunction(()=> (window.__doneCount||0)>0, null, {timeout:30000});
  if (setup){ const c=await page.evaluate(()=>window.__doneCount||0); await page.evaluate(setup); await page.waitForFunction(p=> (window.__doneCount||0)>p, c, {timeout:60000}); }
  await page.waitForTimeout(300);
  await page.screenshot({ path:`screenshots/${name}.png` });
  const s = await page.evaluate(()=>window.__lastDone);
  console.log(name, JSON.stringify(s));
}
await shot('home', null);
await shot('seahorse', ()=>window.__viewer.setState({cx:'-0.743643887037158704752191506114774',cy:'0.131825904205311970493132056385139',radius:5e-13,maxIter:8000}));
await shot('deep60', ()=>window.__viewer.setState({cx:'-0.743643887037158704752191506114774',cy:'0.131825904205311970493132056385139',radius:1e-18,maxIter:25000}));
await browser.close();
