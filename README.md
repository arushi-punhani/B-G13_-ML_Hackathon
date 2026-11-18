# PathSnap

PathSnap matches GPS observations to road-network candidates and reconstructs a
connected route with an HMM/Viterbi decoder.

## Backend API

The API serves the ten reserved trips from `Student folder/student_test`. It
uses the standalone geometry-based HMM/Viterbi decoder in
`PathSnap_Standalone_HMM_Evaluated.py` and expands disconnected predicted arcs
through the full road network using
`PathSnap_Full_Route_Reconstruction.py`. Decoder settings are loaded from
`backend/artifacts/standalone_hmm_config.json`. Reference `.route` files are
read only on the server after prediction to calculate an evaluation diagnostic;
they are not model inputs and are never returned by the API.

The standalone HMM uses normalized distance and heading emissions plus
connectivity-aware Viterbi decoding; it is a geometry-based heuristic, not a
trained classifier. Since the supplied `.route` files are sequences of arcs,
not point-aligned labels, `route_membership_proxy_rate` only measures whether a
predicted arc appears somewhere in that trip's reference route. It is not
observation-level accuracy.

### Setup and run on Laptop 1

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

For a machine that already has the project dependencies installed, create the
environment with `--system-site-packages` to reuse them. This standalone HMM
does not require a separate learned-model artifact.

### Endpoints

- `GET /health` — service readiness.
- `GET /trips` — available student-test trip IDs and counts.
- `POST /predict/{trip_id}` — run candidate scoring and HMM/Viterbi decoding.
- `GET /docs` — interactive API documentation.

Prediction responses include the GPS points, selected arc IDs, road-segment
geometry and server-computed diagnostics. They do not include the reference
route. The route-membership proxy and connected-transition rate are diagnostics,
not official observation-level accuracy.

The evaluator UI is in [`frontend/`](frontend/README.md). It connects to
`GET /trips` and `POST /predict/{trip_id}` and displays GPS observations,
candidate-road geometry, the reconstructed route and server diagnostics. See
its README for exact two-laptop setup and CORS instructions.

For a UI hosted on another origin, set `PATHSNAP_CORS_ORIGINS` to a
comma-separated list of allowed origins before starting Uvicorn, for example:

```powershell
$env:PATHSNAP_CORS_ORIGINS = "http://192.168.1.20:3000"
```
