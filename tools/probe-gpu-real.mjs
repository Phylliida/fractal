// Probe: can headless Chromium use the REAL NVIDIA GPU (RTX 3090) for WebGL2,
// instead of SwiftShader? Tries several GPU flag combinations and reports the
// unmasked vendor/renderer + whether a float render-to-texture round-trips.
// Run: node tools/probe-gpu-real.mjs
import { chromium } from '@playwright/test';
import { readdirSync, existsSync } from 'node:fs';

function resolveChromium() {
  if (process.env.CHROMIUM_PATH && existsSync(process.env.CHROMIUM_PATH)) return process.env.CHROMIUM_PATH;
  const dirs = readdirSync('/nix/store').filter((d) => /-chromium-\d/.test(d) && !d.includes('sandbox')).sort().reverse();
  for (const d of dirs) { const p = `/nix/store/${d}/bin/chromium`; if (existsSync(p)) return p; }
  return undefined;
}
const exe = resolveChromium();

const COMMON = ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage'];

const CONFIGS = [
  { name: 'swiftshader (baseline)', args: [...COMMON, '--disable-gpu', '--enable-unsafe-swiftshader', '--headless=new'] },
  { name: 'angle-vulkan + headless=new', args: [...COMMON, '--headless=new', '--use-angle=vulkan', '--use-gl=angle',
      '--enable-features=Vulkan', '--ignore-gpu-blocklist', '--enable-gpu'] },
  { name: 'angle-vulkan + ozone-headless', args: [...COMMON, '--headless=new', '--ozone-platform=headless',
      '--use-angle=vulkan', '--use-gl=angle', '--enable-features=Vulkan', '--ignore-gpu-blocklist'] },
  { name: 'angle-gl-egl (native NVIDIA EGL)', args: [...COMMON, '--headless=new', '--use-angle=gl-egl',
      '--use-gl=angle', '--ignore-gpu-blocklist'] },
  { name: 'use-gl=egl', args: [...COMMON, '--headless=new', '--use-gl=egl', '--ignore-gpu-blocklist'] },
  { name: 'use-gl=angle default backend', args: [...COMMON, '--headless=new', '--use-gl=angle', '--ignore-gpu-blocklist'] },
  { name: 'no-disable-gpu plain', args: [...COMMON, '--headless=new', '--ignore-gpu-blocklist', '--enable-gpu'] },
];

const PROBE = () => {
  const c = document.createElement('canvas');
  c.width = 64; c.height = 64;
  const gl = c.getContext('webgl2', { antialias: false, preserveDrawingBuffer: true });
  if (!gl) return { webgl2: false };
  const dbg = gl.getExtension('WEBGL_debug_renderer_info');
  const out = {
    webgl2: true,
    vendor: dbg ? gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) : '?',
    renderer: dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : '?',
    colorBufferFloat: !!gl.getExtension('EXT_color_buffer_float'),
    maxTexSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
  };
  return out;
};

for (const cfg of CONFIGS) {
  let browser, line;
  try {
    browser = await chromium.launch({ executablePath: exe, args: cfg.args, timeout: 30000 });
    const page = await browser.newPage();
    const errs = [];
    page.on('pageerror', (e) => errs.push(String(e)));
    const r = await page.evaluate(PROBE);
    line = JSON.stringify(r);
  } catch (e) {
    line = 'LAUNCH/EVAL ERROR: ' + (e.message || e);
  } finally {
    if (browser) await browser.close().catch(() => {});
  }
  console.log(`\n### ${cfg.name}\n  ${line}`);
}
