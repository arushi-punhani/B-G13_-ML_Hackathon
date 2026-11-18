// Browser integration smoke test with explicitly synthetic, intercepted API data.
import { chromium } from 'playwright';
import { createServer } from 'vite';
import assert from 'node:assert/strict';

const feature = (type, coordinates) => ({ type: 'Feature', properties: {}, geometry: { type, coordinates } });
const fc = (features) => ({ type: 'FeatureCollection', features });
const fixture = { trip_id: '00000001', gps_track: [{ longitude: 8, latitude: 50 }, { longitude: 8.01, latitude: 50.01 }], road_network_geojson: fc([feature('LineString', [[8, 50], [8.01, 50.01]])]), predicted_route_geojson: fc([feature('LineString', [[8, 50], [8.01, 50.01]])]), metrics: { decoded_observations: 2 } };
const server = await createServer({ server: { host: '127.0.0.1', port: 0 }, logLevel: 'silent' });
await server.listen();
let browser;
try {
  browser = await chromium.launch({ headless: true, ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}) });
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.route('http://localhost:8000/trips', (route) => route.fulfill({ json: [{ trip_id: '00000001' }] }));
  await page.route('http://localhost:8000/predict/00000001', async (route) => {
    assert.equal(route.request().method(), 'POST');
    await route.fulfill({ json: fixture });
  });
  await page.route('https://tile.openstreetmap.org/**', (route) => route.abort());
  const address = server.httpServer.address();
  await page.goto(`http://127.0.0.1:${address.port}/`);
  await page.getByRole('button', { name: /connect server/i }).click();
  await page.getByLabel('Evaluation trip').selectOption('00000001');
  await page.getByRole('button', { name: /run reconstruction/i }).click();
  await page.getByText('Decoded observations').waitFor();
  await page.locator('.leaflet-container').waitFor();
  assert.equal(await page.locator('.leaflet-interactive').count() >= 3, true, 'GPS and route layers should render');
  assert.deepEqual(errors, []);
  console.log('Browser smoke test passed: mocked trip selection, prediction, metrics, and real Leaflet layers rendered.');
} finally {
  await browser?.close();
  await server.close();
}
