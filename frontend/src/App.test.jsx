import { expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App.jsx';

vi.mock('./MapView.jsx', () => ({ default: ({ result }) => <div>Map for {result.tripId}</div> }));

it('connects, selects a server-provided trip and displays server results', async () => {
  const user = userEvent.setup();
  const fc = (features) => ({ type: 'FeatureCollection', features });
  const f = (type, coordinates) => ({ type: 'Feature', properties: {}, geometry: { type, coordinates } });
  vi.stubGlobal('fetch', vi.fn().mockResolvedValueOnce({ ok: true, json: async () => ([{ trip_id: '00000001' }]) }).mockResolvedValueOnce({ ok: true, json: async () => ({
    trip_id: '00000001', gps_track: [{ longitude: 8, latitude: 50 }], road_network_geojson: fc([]), predicted_route_geojson: fc([f('LineString', [[8, 50], [8.1, 50.1]])]), metrics: { decoded_observations: 1 },
  }) }));
  render(<App />);
  expect(screen.getByRole('button', { name: /run reconstruction/i })).toBeDisabled();
  await user.click(screen.getByRole('button', { name: /connect server/i }));
  await user.selectOptions(await screen.findByLabelText(/evaluation trip/i), '00000001');
  await user.click(screen.getByRole('button', { name: /run reconstruction/i }));
  expect(await screen.findByText('Map for 00000001')).toBeInTheDocument();
  expect(screen.getByText('Decoded observations')).toBeInTheDocument();
  expect(fetch.mock.calls[1][0]).toBe('http://localhost:8000/predict/00000001');
});

it('shows connection errors without inventing trips or results', async () => {
  const user = userEvent.setup();
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('network error')));
  render(<App />);
  await user.click(screen.getByRole('button', { name: /connect server/i }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Unable to reach the backend');
  expect(screen.getByRole('button', { name: /run reconstruction/i })).toBeDisabled();
});
