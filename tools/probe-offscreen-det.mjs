// quick repro of the e2e determinism scenario, offscreen vs main-thread
import { chromium } from '@playwright/test';
import { launchOpts } from './chromium-launch.mjs';
import { spawn } from 'node:child_process';
const PORT = process.env.PORT || 8167;
const server = spawn(process.execPath, ['tools/serve.mjs'], { env: { ...process.env, PORT }, stdio: 'ignore' });
const baseURL = `http://127.0.0.1:${PORT}`;
async function waitServer() { for (let i=0;i<100;i++){ try{const r=await fetch(baseURL+'/index.html'); if(r.ok)return;}catch{} await new Promise(r=>setTimeout(r,100)); } throw new Error('no server'); }
const fpSrc = (G) => { const c=document.getElementById('view'); const g=c.getContext('2d'); const {data}=g.getImageData(0,0,c.width,c.height); let h=0x811c9dc5; for(let gy=0;gy<G;gy++)for(let gx=0;gx<G;gx++){const px=Math.floor((gx+0.5)/G*c.width),py=Math.floor((gy+0.5)/G*c.height);const o=(py*c.width+px)*4;const q=((data[o]>>3)<<10)|((data[o+1]>>3)<<5)|(data[o+2]>>3);h^=q;h=Math.imul(h,0x01000193)>>>0;} return h>>>0; };
async function run(offscreen){
  const browser = await chromium.launch(launchOpts());
  const page = await browser.newPage({ viewport:{width:600,height:600} });
  page.on('pageerror',(e)=>console.error('PAGEERR',e.message));
  await page.goto(baseURL+'/index.html?offscreen='+(offscreen?'1':'0'));
  await page.waitForFunction(()=>(window.__doneCount||0)>0,null,{timeout:30000});
  const fps=[];
  for (let rep=0; rep<4; rep++){
    const c0 = await page.evaluate(()=>window.__doneCount||0);
    await page.evaluate(()=>{ const v=window.__viewer; v.backingW=256;v.backingH=256;v.canvas.width=256;v.canvas.height=256;v.stable.width=256;v.stable.height=256; v.setState({cx:'-0.5',cy:'0',radius:1.5}); });
    await page.waitForFunction((p)=>(window.__doneCount||0)>p, c0, {timeout:30000});
    const info = await page.evaluate(()=>({off:window.__viewer._useOffscreen(),eng:window.__lastDone?.engine,ss:window.__viewer._effSS,cw:window.__viewer.cW}));
    const fp = await page.evaluate(fpSrc, 16);
    fps.push(fp);
    if (rep===0) console.log(`  [${offscreen?'offscreen':'main'}] off=${info.off} eng=${info.eng} effSS=${info.ss} cW=${info.cw}`);
  }
  console.log(`  [${offscreen?'offscreen':'main'}] fps:`, fps.join(', '), fps.every(x=>x===fps[0])?'STABLE':'**UNSTABLE**');
  await browser.close();
}
try { await waitServer(); await run(false); await run(true); } catch(e){ console.error(e); } finally { server.kill(); }
