export function normalizeBaseUrl(value) {
  const url = new URL(value.trim());
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
    throw new Error('Enter an HTTP or HTTPS backend address without credentials, a query, or a fragment.');
  }
  return url.href.replace(/\/$/, '');
}

async function request(baseUrl, path, options = {}) {
  let response;
  try {
    response = await fetch(`${normalizeBaseUrl(baseUrl)}${path}`, options);
  } catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new Error('Unable to reach the backend. Check its address, network connection, and CORS settings.');
  }
  if (!response.ok) {
    let message = `Backend returned HTTP ${response.status}.`;
    try {
      const body = await response.json();
      if (typeof body.detail === 'string') message = body.detail;
      else if (typeof body.message === 'string') message = body.message;
    } catch { /* Non-JSON error responses keep the HTTP status. */ }
    throw new Error(message);
  }
  try { return await response.json(); }
  catch { throw new Error('The backend returned an invalid JSON response.'); }
}

export async function getTrips(baseUrl, signal) {
  const data = await request(baseUrl, '/trips', { signal });
  if (!Array.isArray(data)) throw new Error('Invalid trip response: expected an array.');
  const trips = data.map((trip) => {
    if (!trip || typeof trip.trip_id !== 'string' || !trip.trip_id.trim()) {
      throw new Error('Invalid trip response: each trip needs a string trip_id.');
    }
    return { id: trip.trip_id, label: `Trip ${trip.trip_id}`, partition: typeof trip.partition === 'string' ? trip.partition : '' };
  });
  if (new Set(trips.map((trip) => trip.id)).size !== trips.length) throw new Error('The backend returned duplicate trip IDs.');
  return trips;
}

const isCoordinate = (position) => Array.isArray(position) && position.length >= 2 &&
  Number.isFinite(position[0]) && Number.isFinite(position[1]) &&
  Math.abs(position[0]) <= 180 && Math.abs(position[1]) <= 90;

function validGeometry(geometry, allowed) {
  if (!geometry || !allowed.includes(geometry.type)) return false;
  const c = geometry.coordinates;
  if (geometry.type === 'Point') return isCoordinate(c);
  if (geometry.type === 'MultiPoint') return Array.isArray(c) && c.length > 0 && c.every(isCoordinate);
  if (geometry.type === 'LineString') return Array.isArray(c) && c.length >= 2 && c.every(isCoordinate);
  if (geometry.type === 'MultiLineString') return Array.isArray(c) && c.length > 0 && c.every((line) => Array.isArray(line) && line.length >= 2 && line.every(isCoordinate));
  return false;
}

function featureCollection(value, name, allowed) {
  if (!value || value.type !== 'FeatureCollection' || !Array.isArray(value.features) ||
      !value.features.every((feature) => feature?.type === 'Feature' && validGeometry(feature.geometry, allowed))) {
    throw new Error(`Invalid ${name} geometry in the backend response.`);
  }
  // Keep geometry only: arbitrary properties, including reference labels, are not displayed.
  return { type: 'FeatureCollection', features: value.features.map(({ geometry }) => ({ type: 'Feature', properties: {}, geometry })) };
}

export function validatePrediction(data, requestedTripId) {
  if (!data || data.trip_id !== requestedTripId) throw new Error('The prediction response does not match the selected trip.');
  if (!Array.isArray(data.gps_track)) throw new Error('Invalid GPS track in the backend response.');
  const gps = featureCollection({ type: 'FeatureCollection', features: data.gps_track.map((point) => ({
    type: 'Feature', properties: {}, geometry: { type: 'Point', coordinates: [point.longitude, point.latitude] },
  })) }, 'GPS', ['Point', 'MultiPoint']);
  const roads = featureCollection(data.road_network_geojson || { type: 'FeatureCollection', features: [] }, 'road network', ['LineString', 'MultiLineString']);
  const route = featureCollection(data.predicted_route_geojson, 'predicted route', ['LineString', 'MultiLineString']);
  if (!gps.features.length || !route.features.length) throw new Error('The backend returned no GPS observations or predicted route.');
  if (!data.metrics || typeof data.metrics !== 'object' || Array.isArray(data.metrics)) throw new Error('Invalid metrics in the backend response.');
  const definitions = [
    ['route_membership_proxy_rate', 'Route membership proxy', 'percentage', data.metrics.route_membership_proxy_note || 'Predicted arcs occurring anywhere in the reference route; not observation-level accuracy.'],
    ['candidate_coverage', 'Candidate coverage', 'percentage', 'Observations with candidate roads / total observations.'],
    ['prediction_coverage', 'Prediction coverage', 'percentage', 'Decoded observations / total observations.'],
    ['connected_transition_rate', 'Connected transition rate', 'percentage', 'Server-reported reconstructed route connectivity diagnostic.'],
    ['decoded_observations', 'Decoded observations', 'count', 'Observations with an HMM prediction.'],
    ['unresolved_route_gaps', 'Unresolved route gaps', 'count', 'Gaps not bridged by the server reconstruction.'],
    ['inserted_intermediate_arcs', 'Intermediate arcs', 'count', 'Road arcs inserted by route reconstruction.'],
  ];
  const metrics = definitions.filter(([key]) => data.metrics[key] != null).map(([key, label, format, description]) => {
    const value = data.metrics[key];
    if (typeof value !== 'number' || !Number.isFinite(value) || (format === 'percentage' && (value < 0 || value > 1))) throw new Error(`Invalid ${label} metric in the backend response.`);
    return { label, value: format === 'percentage' ? (value * 100).toFixed(2) : value, unit: format === 'percentage' ? '%' : '', description };
  });
  return { tripId: data.trip_id, gps, roads, route, metrics, message: `Standalone geometric HMM · ${data.partition || 'trip'} partition · Reference routes remain on the server.` };
}

export async function predict(baseUrl, tripId, signal) {
  const data = await request(baseUrl, `/predict/${encodeURIComponent(tripId)}`, { method: 'POST', signal });
  return validatePrediction(data, tripId);
}
