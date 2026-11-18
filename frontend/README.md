# PathSnap evaluator frontend

This React/Vite UI uses the actual FastAPI backend in `backend/app.py`. It fetches server-provided trip IDs, requests inference, and renders GPS observations, candidate road-network geometry, the predicted route and server-reported diagnostics. Reference routes stay server-side. `route_membership_proxy_rate` is explicitly labelled as a proxy, not observation-level accuracy.

The backend currently runs the standalone geometric HMM, **not** a trained classifier plus HMM. When `evaluator_test/` exists at the project root on Laptop 1, the backend discovers its trip folders automatically and generates candidate features in memory when a trip is requested. Otherwise, it serves the supplied `student_test` development trips. The interface displays the active partition. A final classifier-based pipeline remains to be integrated.

## Laptop 1: backend

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Keep this terminal open to see incoming requests. Run `ipconfig` on Laptop 1 to find its LAN IPv4 address. Both laptops need to be on a network that allows connections to Laptop 1 port 8000. The backend's default CORS settings allow `http://localhost:5173` and `http://127.0.0.1:5173`; these are the frontend **browser origin** on Laptop 2. If another frontend origin is used, set `PATHSNAP_CORS_ORIGINS` to a comma-separated list of exact origins before starting Uvicorn.

To explicitly select the partition before starting the backend:

```powershell
$env:PATHSNAP_PARTITION = "evaluator_test"  # or "student_test" for development
```

For evaluator use, put `evaluator_test/` at the project root on Laptop 1, with one subdirectory per trip and matching `.track`, `.nodes` and `.arcs` files. The optional `.route` file is read only after inference to calculate a server-side route-membership proxy; its contents are not returned to the UI. If the reference is absent, the proxy is omitted. The folder is ignored by Git. Keep evaluator data separate from development and model selection.

## Laptop 2: frontend

Install Node.js `^20.19.0` or `>=22.12.0`. From the project root in PowerShell:

```powershell
Set-Location -LiteralPath "frontend"
npm ci
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Open `http://localhost:5173/` in Laptop 2's browser. Enter `http://<LAPTOP_1_LAN_IP>:8000` in **Backend address**, click **Connect server**, select a trip and click **Run reconstruction**. The address is stored in browser local storage. `localhost` on Laptop 2 points to Laptop 2, so use Laptop 1's LAN IP for a two-laptop setup.

For same-computer use, start both processes and use `http://localhost:8000` in the UI. OpenStreetMap tiles require network access. The network layer contains candidate roads returned by the backend; toggle it using the map's layer control.

## API used

- `GET /health`: service readiness.
- `GET /trips`: array of objects with string `trip_id` and `partition`.
- `POST /predict/{trip_id}`: returns `trip_id`, `metrics`, `gps_track`, `predictions`, `predicted_route_geojson`, `road_network_geojson` and `unresolved_gaps`. Candidate-road geometry was added to the backend response for the map. GeoJSON uses `[longitude, latitude]`. Road and predicted route features are separate line segments, so gaps are not drawn as invented straight lines.

The UI does not request or display reference labels. Backend diagnostics are displayed with their descriptions. Because `.route` is an arc sequence rather than point-level labels, the route-membership proxy must not be described as accuracy. The UI handles unavailable servers, empty lists, invalid responses and inference errors. The backend needs its raw data and preprocessed Parquet files on Laptop 1.

## Verify

In `frontend/`:

```powershell
npm test
npm run build
```

`npm run test:browser` uses synthetic intercepted responses. `npm run test:integration` launches the real backend, uses student-test trip `00000052` and exercises the UI in a browser; the backend's Python dependencies and Chrome/Chromium must be installed. On Windows, for example:

```powershell
$env:CHROME_PATH = 'C:\Program Files\Google\Chrome\Application\chrome.exe'
$env:PATHSNAP_PYTHON = (Resolve-Path '..\.venv\Scripts\python.exe').Path
npm run test:integration
```

The integration test exercises a local browser and backend, not a physical two-laptop network or the hidden evaluator partition. The team should test every final evaluator trip over the actual network once that data and final pipeline are available.
