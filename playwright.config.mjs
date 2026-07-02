import { defineConfig, devices } from '@playwright/test';
import { resolveChromium, chromiumArgs, gpuMode } from './tools/chromium-launch.mjs';

const PORT = process.env.PORT || 8137;

// Chromium resolution + GPU/SwiftShader flag selection live in the shared helper
// (tools/chromium-launch.mjs) so the e2e suite and every tool agree. Default is
// SwiftShader (CI-safe); set GPU=1 to run the suite on the real GPU (Vulkan), or
// GPU=gl for native GLES. See the helper for the verified flag combos.
const executablePath = resolveChromium();
const launchArgs = chromiumArgs();
if (gpuMode() !== 'swiftshader') console.log(`[playwright] GPU mode: ${gpuMode()} (real GPU)`);

export default defineConfig({
  testDir: './test/e2e',
  timeout: 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  reporter: process.env.CI ? 'line' : [['list']],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    headless: true,
    viewport: { width: 390, height: 760 }, // mobile-ish default
    trace: 'retain-on-failure',
    launchOptions: {
      ...(executablePath ? { executablePath } : {}),
      // Flags come from the shared helper (SwiftShader by default; real GPU when
      // GPU=1/gl). '--headless=new' is last in that list so it wins over
      // Playwright's own (old) '--headless', the only mode that survives this sandbox.
      args: launchArgs,
    },
  },
  projects: [
    { name: 'mobile-chrome', use: { ...devices['Pixel 7'] } },
    { name: 'desktop-chrome', use: { ...devices['Desktop Chrome'], viewport: { width: 1100, height: 800 } } },
  ],
  webServer: {
    command: `node tools/serve.mjs`,
    url: `http://127.0.0.1:${PORT}`,
    reuseExistingServer: !process.env.CI,
    timeout: 20_000,
  },
});
