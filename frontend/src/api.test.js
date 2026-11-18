import { describe, expect, it, vi } from 'vitest';
import { getTrips, normalizeBaseUrl, predict, validatePrediction } from './api.js';

const feature = (type, coordinates) => ({ type: 'Feature', properties: { reference: 'must not render' }, geometry: { type, coordinates } });
const collection = (...features) => ({ type: 'FeatureCollection', features });
const result = { trip_id: '00000001', gps_track: [{ longitude: 8, latitude: 50 }], road_network_geojson: collection(), predicted_route_geojson: collection(feature('LineString', [[8, 50], [8.1, 50.1]])), metrics: { connected_transition_rate: 0.9 } };

describe('backend contract', () => {
  it('preserves leading-zero trip IDs', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ([{ trip_id: '00000001' }]) }));
    expect(await getTrips('http://127.0.0.1:8000/', undefined)).toEqual([{ id: '00000001', label: 'Trip 00000001', partition: '' }]);
  });
  it('sends the selected trip ID and strips feature properties', async () => {
    const fetcher = vi.fn().mockResolvedValue({ ok: true, json: async () => result });
    vi.stubGlobal('fetch', fetcher);
    const data = await predict('http://127.0.0.1:8000', '00000001');
    expect(fetcher.mock.calls[0][0]).toBe('http://127.0.0.1:8000/predict/00000001');
    expect(fetcher.mock.calls[0][1].method).toBe('POST');
    expect(data.route.features[0].properties).toEqual({});
    expect(data.metrics[0].label).toBe('Connected transition rate');
    expect(data.metrics[0].value).toBe('90.00');
  });
  it('rejects mismatched trip IDs and malformed coordinates', () => {
    expect(() => validatePrediction(result, '00000002')).toThrow(/does not match/);
    expect(() => validatePrediction({ ...result, gps_track: [{ longitude: 200, latitude: 50 }] }, '00000001')).toThrow(/GPS/);
  });
  it('does not invent an accuracy metric when no reference is available', () => {
    const data = validatePrediction({ ...result, metrics: { route_membership_proxy_rate: null, decoded_observations: 1 } }, '00000001');
    expect(data.metrics.map((metric) => metric.label)).toEqual(['Decoded observations']);
  });
  it('reports backend failures and rejects duplicate trip IDs', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 503, json: async () => ({ detail: 'Model unavailable' }) }));
    await expect(getTrips('http://localhost:8000')).rejects.toThrow('Model unavailable');
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ([{ trip_id: '1' }, { trip_id: '1' }]) }));
    await expect(getTrips('http://localhost:8000')).rejects.toThrow(/duplicate/);
  });
  it('validates server URLs', () => {
    expect(normalizeBaseUrl('http://192.168.1.2:8000/')).toBe('http://192.168.1.2:8000');
    expect(() => normalizeBaseUrl('javascript:alert(1)')).toThrow();
  });
});
