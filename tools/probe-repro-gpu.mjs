// probe-repro-gpu.mjs — GPU-vs-CPU-oracle at Danielle's reported "glitchy fine detail"
// coordinate (Spawn 31): a SHORT, EARLY-ESCAPING reference (refLen ~3.5k vs maxIter
// ~63k, SA skip within ~200 of the reference end) — a corner the standard validation
// coordinate (non-escaping reference) never stresses. probe-wall already proved the
// CPU oracle EXACT here (0/144 vs BigInt); this measures what the GPU adds on top.
//   GPU=1 node tools/probe-repro-gpu.mjs
import { chromium } from '@playwright/test';
import { spawn } from 'node:child_process';
import { launchOpts, gpuMode } from './chromium-launch.mjs';

const PORT = process.env.PORT || 8157;
const server = spawn(process.execPath, ['tools/serve.mjs'], { env: { ...process.env, PORT }, stdio: 'ignore' });
const baseURL = `http://127.0.0.1:${PORT}`;
for (let i = 0; i < 100; i++) { try { const r = await fetch(baseURL + '/test/gpu/harness.html'); if (r.ok) break; } catch { /* retry */ } await new Promise((r) => setTimeout(r, 100)); }

const RE = '-0.013723903427109727127929602217373808131181131374105015124101269177883848493753482931960573952845398672419';
const IM = '0.720216709412699393589401716000260792609362764831527728273017596899498993965708580806931908277598917947246';
const R = 2.99709927e-76;                 // ≈ 2^-250.9
const IT = Number(process.env.IT || 63150);
const W = Number(process.env.W || 160);

try {
  const browser = await chromium.launch(launchOpts());
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.error('PAGE ERROR:', e.message));
  await page.goto(baseURL + '/test/gpu/harness.html');
  await page.waitForFunction(() => window.__ready === true, { timeout: 15000 });
  await page.evaluate(() => window.__gpu.init(256, 256));
  console.log('mode=' + gpuMode() + '  renderer:', JSON.parse(JSON.stringify(await page.evaluate(() => window.__gpu.info()))).renderer);
  console.log(`\nGPU vs CPU oracle at the reported coordinate, 2^-251, ${W}×${W}, maxIter ${IT} (CPU==BigInt per probe-wall)\n`);
  console.log('config             | mism%    inside  meanΔsn   maxΔsn   | skip   refLen');
  for (const cfg of [
    { label: 'rs + SA (prod)  ', rs: true, series: true },
    { label: 'rs   no SA      ', rs: true, series: false },
    { label: 'fe   no SA      ', fe: true, series: false },
    { label: 'df64 no SA      ', df64: true, series: false },
  ]) {
    const r = await page.evaluate((qq) => window.__gpu.comparePerturb(qq), {
      re: RE, im: IM, radius: R, maxIter: IT, width: W, height: W, checkStep: 1,
      rs: !!cfg.rs, fe: !!cfg.fe, df64: !!cfg.df64, series: cfg.series,
    });
    console.log(`${cfg.label} | ${(100 * r.mism / Math.max(1, r.compared)).toFixed(3)}%  ${String(r.insideMismatch).padStart(5)}  ` +
      `${r.meanAbs.toExponential(2)}  ${r.maxAbs.toExponential(2)} | ${String(r.saSkip).padStart(6)} ${String(r.refLen).padStart(6)}`);
    if (r.mism / Math.max(1, r.compared) > 0.01 && r.examples?.length) {
      for (const ex of r.examples.slice(0, 4)) console.log('   ex:', JSON.stringify(ex));
    }
  }
  await browser.close();
} catch (e) { console.error('ERROR:', e.stack || e); } finally { server.kill(); }
