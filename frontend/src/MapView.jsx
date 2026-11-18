import { useEffect } from 'react';
import L from 'leaflet';
import { CircleMarker, GeoJSON, LayerGroup, LayersControl, MapContainer, TileLayer, useMap } from 'react-leaflet';

function FitToData({ data }) {
  const map = useMap();
  useEffect(() => {
    const bounds = L.geoJSON(data).getBounds();
    if (bounds.isValid()) map.fitBounds(bounds, { padding: [25, 25], maxZoom: 17 });
    map.invalidateSize();
  }, [data, map]);
  return null;
}

export default function MapView({ result }) {
  const positions = result.gps.features.flatMap(({ geometry }) => geometry.type === 'Point' ? [geometry.coordinates] : geometry.coordinates);
  return <MapContainer center={[0, 0]} zoom={2} scrollWheelZoom className="map" aria-label="Predicted trip map">
    <TileLayer url="https://tile.openstreetmap.org/{z}/{x}/{y}.png" attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' />
    <FitToData data={{ type: 'FeatureCollection', features: [...result.route.features, ...result.gps.features] }} />
    <LayersControl position="topright">
      <LayersControl.Overlay checked name="GPS observations">
        <LayerGroup>{positions.map(([lon, lat], index) => <CircleMarker key={index} center={[lat, lon]} radius={3} pathOptions={{ color: '#1874c5', fillColor: '#1874c5', fillOpacity: 0.7, weight: 1 }} />)}</LayerGroup>
      </LayersControl.Overlay>
      <LayersControl.Overlay name="Road network">
        <GeoJSON key={`roads-${result.tripId}`} data={result.roads} style={{ color: '#7d8d94', weight: 2, opacity: 0.65 }} />
      </LayersControl.Overlay>
      <LayersControl.Overlay checked name="Predicted route">
        <GeoJSON key={`route-${result.tripId}`} data={result.route} style={{ color: '#db5b32', weight: 5, opacity: 0.9 }} />
      </LayersControl.Overlay>
    </LayersControl>
  </MapContainer>;
}
