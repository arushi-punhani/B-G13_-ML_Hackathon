import { useEffect, useRef, useState } from 'react';
import { getTrips, normalizeBaseUrl, predict } from './api.js';
import MapView from './MapView.jsx';

const DEFAULT_API = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export default function App() {
  const [address, setAddress] = useState(() => localStorage.getItem('pathsnap-api') || DEFAULT_API);
  const [connectedUrl, setConnectedUrl] = useState('');
  const [trips, setTrips] = useState([]);
  const [selected, setSelected] = useState('');
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const request = useRef(null);

  useEffect(() => () => request.current?.abort(), []);

  async function connect(event) {
    event.preventDefault();
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setBusy('Connecting'); setError(''); setTrips([]); setSelected(''); setResult(null); setConnectedUrl('');
    try {
      const base = normalizeBaseUrl(address);
      const available = await getTrips(base, controller.signal);
      if (controller.signal.aborted) return;
      setAddress(base); setConnectedUrl(base); setTrips(available);
      localStorage.setItem('pathsnap-api', base);
    } catch (err) {
      if (err.name !== 'AbortError') setError(err.message);
    } finally {
      if (request.current === controller) setBusy('');
    }
  }

  async function runPrediction(event) {
    event.preventDefault();
    if (!selected || !connectedUrl) return;
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setBusy('Reconstructing'); setError(''); setResult(null);
    try {
      const data = await predict(connectedUrl, selected, controller.signal);
      if (!controller.signal.aborted) setResult(data);
    } catch (err) {
      if (err.name !== 'AbortError') setError(err.message);
    } finally {
      if (request.current === controller) setBusy('');
    }
  }

  function changeAddress(value) {
    request.current?.abort(); request.current = null;
    setAddress(value); setConnectedUrl(''); setTrips([]); setSelected(''); setResult(null); setError(''); setBusy('');
  }

  return <div className="app-shell">
    <header className="topbar"><div className="brand-mark">⌁</div><div><div className="brand">PATHSNAP</div><div className="brand-sub">ROUTE INTELLIGENCE</div></div><span className="topbar-label">EVALUATOR WORKSPACE</span></header>
    <main>
      <section className="hero"><div><span className="eyebrow">MAP MATCHING / LIVE INFERENCE</span><h1>Trace the road<br /><em>behind the signal.</em></h1><p>Choose an available trip and run the backend reconstruction pipeline to inspect the predicted route and evaluation results.</p></div><div className="hero-index"><span>PATHSNAP</span><strong>LIVE</strong><small>TRAJECTORY ANALYSIS</small></div></section>
      <div className="workspace">
        <aside className="sidebar">
          <div className="panel-title"><span className="section-number">01</span><div><h2>Connection</h2><p>Model server · Laptop 1</p></div></div>
          <form onSubmit={connect}>
            <label htmlFor="address">Backend address</label>
            <input id="address" type="url" value={address} onChange={(event) => changeAddress(event.target.value)} placeholder="http://192.168.1.10:8000" required />
            <button type="submit" className="button secondary" disabled={Boolean(busy)}>{busy === 'Connecting' ? 'Connecting…' : connectedUrl ? 'Reconnect server' : 'Connect server'} <span>↗</span></button>
          </form>
          <div className={`connection ${connectedUrl ? 'online' : ''}`}><span className="status-dot" />{connectedUrl ? 'SERVER CONNECTED' : 'SERVER NOT CONNECTED'}</div>
          <div className="divider" />
          <div className="panel-title"><span className="section-number">02</span><div><h2>Select a trip</h2><p>Available from the server</p></div></div>
          <form onSubmit={runPrediction}>
            <label htmlFor="trip">Evaluation trip</label>
            <select id="trip" value={selected} disabled={!connectedUrl || Boolean(busy) || !trips.length} onChange={(event) => { setSelected(event.target.value); setResult(null); setError(''); }} required>
              <option value="">{connectedUrl && !trips.length ? 'No trips available' : 'Select a trip'}</option>
              {trips.map((trip) => <option key={trip.id} value={trip.id}>{trip.label}</option>)}
            </select>
            <p className="field-hint">{connectedUrl ? `${trips.length} trip${trips.length === 1 ? '' : 's'} available` : 'Connect to load the trip list.'}</p>
            <button type="submit" className="button primary" disabled={!selected || !connectedUrl || Boolean(busy)}>{busy === 'Reconstructing' ? 'Running inference…' : 'Run reconstruction'} <span>→</span></button>
          </form>
          {error && <div className="alert" role="alert"><strong>Unable to continue</strong><p>{error}</p></div>}
          <div className="sidebar-note"><span>↗</span><p>The selected trip ID is sent to the model server. Results appear only after the backend responds.</p></div>
        </aside>
        <section className="results">
          <div className="results-heading"><div><span className="eyebrow">02 / RECONSTRUCTION</span><h2>Route overview</h2><p>{result ? `Trip ${result.tripId} · Backend prediction` : 'Your trip visualization will appear here.'}</p></div><div className="live-pill"><span className="status-dot" />{busy === 'Reconstructing' ? 'PROCESSING' : result ? 'RESULT READY' : 'AWAITING TRIP'}</div></div>
          <div className="map-frame">{result ? <MapView key={result.tripId} result={result} /> : <div className="map-empty"><div className="map-grid" /><div className="empty-content"><div className="empty-icon">⌖</div><h3>{busy === 'Reconstructing' ? 'Reconstructing route' : 'No trip selected yet'}</h3><p>{busy === 'Reconstructing' ? 'Waiting for the model server to return its prediction.' : 'Connect to the model server and select a trip to begin.'}</p></div></div>}</div>
          <div className="map-footer"><div className="legend"><span><i className="dot gps" />GPS observations</span><span><i className="line predicted" />Predicted route</span><span><i className="line road" />Road network</span></div><span className="map-hint">Use layers on the map to compare</span></div>
          <div className="metrics-heading"><div><span className="eyebrow">03 / EVALUATION</span><h2>Inference results</h2></div><span>SERVER-REPORTED METRICS</span></div>
          {result ? <><div className="metrics-grid">{result.metrics.length ? result.metrics.map((metric, index) => <article className="metric" key={`${metric.label}-${index}`}><span>{metric.label}</span><strong>{metric.value}{metric.unit && <small> {metric.unit}</small>}</strong>{metric.description && <p>{metric.description}</p>}</article>) : <p className="empty-metrics">The backend did not return evaluation metrics for this trip.</p>}</div>{result.message && <p className="result-message">{result.message}</p>}</> : <div className="empty-metrics">Metrics will appear when the backend returns an evaluation result.</div>}
        </section>
      </div>
      <footer><span>PATHSNAP <b> / </b> MAP-MATCHING &amp; ROUTE RECONSTRUCTION</span><span>EVALUATOR INTERFACE</span></footer>
    </main>
  </div>;
}
