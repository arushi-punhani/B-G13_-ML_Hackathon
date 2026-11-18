// Real local backend + frontend browser integration on a permitted student-test trip.
import { spawn } from 'node:child_process';
import { chromium } from 'playwright';
import { createServer } from 'vite';
import assert from 'node:assert/strict';

const python = process.env.PATHSNAP_PYTHON || 'python';
const apiUrl = 'http://127.0.0.1:18765';
const backend = spawn(python, ['-m', 'uvicorn', 'backend.app:app', '--host', '127.0.0.1', '--port', '18765'], { cwd: '..', stdio: ['ignore', 'pipe', 'pipe'] });
let backendLog = '';
backend.stderr.on('data', (data) => { backendLog += data.toString(); });
let browser;
let server;
try {
  let ready = false;
  for (let i = 0; i < 120; i++) {
    if (backend.exitCode !== null) throw new Error(`Backend exited: ${backendLog}`);
    try { const r = await fetch(`${apiUrl}/health`); if (r.ok && (await r.json()).model_loaded) { ready = true; break; } } catch { /* wait for startup */ }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  if (!ready) throw new Error(`Backend did not start: ${backendLog}`);
  const trips = await (await fetch(`${apiUrl}/trips`)).json();
  assert.equal(trips.length, 10);
  assert.ok(trips.some((trip) => trip.trip_id === '00000052'));
  server = await createServer({ server: { host: '127.0.0.1', port: 5173, strictPort: true }, logLevel: 'silent' });
  await server.listen();
  browser = await chromium.launch({ headless: true, ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}) });
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.route('https://tile.openstreetmap.org/**', (route) => route.abort());
  await page.goto('http://127.0.0.1:5173/');
  await page.getByLabel('Backend address').fill(apiUrl);
  await page.getByRole('button', { name: /connect server/i }).click();
  await page.getByLabel('Evaluation trip').selectOption('00000052');
  const responsePromise = page.waitForResponse((r) => r.url() === `${apiUrl}/predict/00000052` && r.request().method() === 'POST', { timeout: 120000 });
  await page.getByRole('button', { name: /run reconstruction/i }).click();
  const response = await responsePromise;
  assert.equal(response.status(), 200);
  const body = await response.json();
  assert.equal(body.trip_id, '00000052');
  assert.ok(body.gps_track.length && body.predicted_route_geojson.features.length && body.road_network_geojson.features.length);
  assert.equal('reference_route' in body, false);
  await page.getByText('Route membership proxy', { exact: true }).waitFor();
  assert.ok(await page.getByText(/not observation-level accuracy/i).isVisible());
  await page.locator('.leaflet-container').waitFor();
  assert.ok(await page.locator('.leaflet-interactive').count() > 2);
  assert.deepEqual(errors, []);
  console.log(`Real backend integration passed: ${trips.length} trips; 00000052 returned ${body.gps_track.length} GPS observations, ${body.road_network_geojson.features.length} road arcs and ${body.predicted_route_geojson.features.length} predicted arcs.`);
} finally {
  await browser?.close();
  await server?.close();
  backend.kill();
}
