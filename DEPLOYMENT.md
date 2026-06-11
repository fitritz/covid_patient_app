# PulseAI Deployment Notes (Render)

This repo includes a root-level `render.yaml` Blueprint that creates:
- `covid-patient-api` - Flask/Gunicorn backend
- `covid-patient-app` - React static frontend

## Backend (Render)
Backend lives in: `webapp/backend`

### Env vars required
- `MONGO_URI` = your MongoDB Atlas connection string (must include the database name at the end)
  - Example: `mongodb+srv://<user>:<pass>@<cluster>/<db>?retryWrites=true&w=majority`
- `FRONTEND_URL` (optional but recommended) = your deployed frontend base URL for CORS allowlist

Render will start the backend using the root `render.yaml`.

### Health check
After deployment, verify:
- `GET /health` (e.g. `https://<render-backend-domain>/health`)

## Frontend (Render Static Site)
Frontend lives in: `webapp/frontend`

The React app calls the backend using:
- `REACT_APP_API_URL` (build-time) if provided
- otherwise it falls back to `https://covid-patient-api.onrender.com`

### Recommended Render env var
- `REACT_APP_API_URL` = your Render backend URL

## Quick local test
1) Set `MONGO_URI` in your environment (or create `webapp/backend/.env` for local development)
2) Start backend: `python webapp/backend/app.py`
3) Start frontend: `cd webapp/frontend && npm install && npm start`
