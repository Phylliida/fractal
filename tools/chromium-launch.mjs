// Shared Chromium launch config for every tool + the Playwright e2e suite.
//
// WHY THIS EXISTS: headless Chromium defaults to the SwiftShader CPU rasterizer
// for WebGL, even when a real GPU is present — and our old launch args made that
// explicit with `--disable-gpu --enable-unsafe-swiftshader`. To actually exercise
// (and profile) the shaders on real hardware you must select an ANGLE GPU backend.
// This module centralizes that choice behind the `GPU` env var so every launcher
// agrees.
//
//   GPU unset / 0 / off / cpu / swiftshader  -> SwiftShader (portable default; CI-safe)
//   GPU=1 / vulkan / angle / true            -> ANGLE on Vulkan -> the real GPU
//   GPU=gl / egl / gl-egl / gles             -> ANGLE on native GLES via NVIDIA EGL
//
// Verified on this host (RTX 3090, NVIDIA 580.82.09, headless, no X display):
//   vulkan -> "ANGLE (NVIDIA, Vulkan 1.4.312 (RTX 3090), NVIDIA)"   maxTex 32768
//   gl     -> "ANGLE (NVIDIA, RTX 3090/PCIe/SSE2, OpenGL ES 3.2)"   maxTex 32768
//   (both expose EXT_color_buffer_float, which the RGBA32F render targets need)
// See tools/probe-gpu-real.mjs for the full flag-combo sweep that found these.
import { readdirSync, existsSync } from 'node:fs';

export function resolveChromium() {
  if (process.env.CHROMIUM_PATH && existsSync(process.env.CHROMIUM_PATH)) return process.env.CHROMIUM_PATH;
  try {
    // Prefer the NEWEST nix-store chromium: this sandbox lacks /sys/devices/system/cpu,
    // which cores older Chromium even in new-headless mode; only the newest survives.
    const dirs = readdirSync('/nix/store')
      .filter((d) => /-chromium-\d/.test(d) && !d.includes('sandbox'))
      .sort().reverse();
    for (const d of dirs) { const p = `/nix/store/${d}/bin/chromium`; if (existsSync(p)) return p; }
  } catch { /* not nixos */ }
  return undefined; // fall back to Playwright's bundled browser
}

export function gpuMode() {
  const v = (process.env.GPU || '').toLowerCase().trim();
  if (v === '' || v === '0' || v === 'off' || v === 'cpu' || v === 'false' || v === 'swiftshader') return 'swiftshader';
  if (v === 'gl' || v === 'egl' || v === 'gl-egl' || v === 'gles') return 'gl';
  return 'vulkan'; // GPU=1 / vulkan / angle / true / anything else truthy
}

// Args common to every mode. '--headless=new' is appended LAST so it wins over
// Playwright's own (old) '--headless'; old headless crashes in this sandbox.
const COMMON = ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage'];

export function chromiumArgs() {
  const mode = gpuMode();
  if (mode === 'gl') {
    // Native OpenGL ES 3.2 via the NVIDIA EGL driver (libEGL_nvidia + libGLESv2_nvidia).
    return [...COMMON, '--use-angle=gl-egl', '--use-gl=angle', '--ignore-gpu-blocklist', '--enable-gpu', '--headless=new'];
  }
  if (mode === 'vulkan') {
    // ANGLE over Vulkan (NVIDIA ICD). This is Chrome's modern default backend on Linux.
    return [...COMMON, '--use-angle=vulkan', '--use-gl=angle', '--enable-features=Vulkan',
            '--ignore-gpu-blocklist', '--enable-gpu', '--headless=new'];
  }
  // swiftshader (portable CPU fallback)
  return [...COMMON, '--disable-gpu', '--enable-unsafe-swiftshader', '--headless=new'];
}

// Convenience for chromium.launch({...}); merges any extra options.
export function launchOpts(extra = {}) {
  return { executablePath: resolveChromium(), args: chromiumArgs(), ...extra };
}
