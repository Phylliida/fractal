// probe-ctxloss.mjs — verify the WEBGL_lose_context extension actually dispatches
// webglcontextlost / webglcontextrestored on this headless backend, so the
// context-loss recovery path can be tested. De-risks the recovery work before it
// is wired into the renderer/viewer. Tries several variations (tick delay, attached
// canvas, longer timeout) to find what restores headless.
import { chromium } from '@playwright/test';
import { chromiumArgs, resolveChromium } from './chromium-launch.mjs';

const browser = await chromium.launch({ executablePath: resolveChromium(), args: chromiumArgs() });
const page = await browser.newPage();
page.on('pageerror', (e) => console.error('PAGE ERROR:', e.message));
await page.goto('about:blank');

const result = await page.evaluate(async () => {
  const out = [];
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  async function trial(name, { attach, tickBeforeRestore }) {
    const canvas = document.createElement('canvas');
    canvas.width = 64; canvas.height = 64;
    if (attach) document.body.appendChild(canvas);
    const gl = canvas.getContext('webgl2');
    if (!gl) return { name, ok: false, why: 'no webgl2' };
    const ext = gl.getExtension('WEBGL_lose_context');
    if (!ext) return { name, ok: false, why: 'no WEBGL_lose_context' };

    let lostFired = false, restoredFired = false;
    canvas.addEventListener('webglcontextlost', (e) => { lostFired = true; e.preventDefault(); });
    canvas.addEventListener('webglcontextrestored', () => { restoredFired = true; });

    ext.loseContext();
    // let the lost event dispatch
    for (let i = 0; i < 20 && !lostFired; i++) await sleep(10);
    if (tickBeforeRestore) await sleep(50);
    ext.restoreContext();
    // wait up to 5s for restore
    for (let i = 0; i < 100 && !restoredFired; i++) await sleep(50);

    let usable = false;
    try {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([1, 2, 3, 4]));
      usable = gl.getError() === gl.NO_ERROR && !gl.isContextLost();
    } catch (e) { usable = false; }
    if (attach) canvas.remove();
    return { name, lostFired, restoredFired, isLost: gl.isContextLost(), usable };
  }

  out.push(await trial('detached+tick', { attach: false, tickBeforeRestore: true }));
  out.push(await trial('attached+tick', { attach: true, tickBeforeRestore: true }));
  return out;
});

console.log(JSON.stringify(result, null, 2));
await browser.close();
const anyRestored = result.some((r) => r.restoredFired && r.usable);
process.exit(anyRestored ? 0 : 1);
