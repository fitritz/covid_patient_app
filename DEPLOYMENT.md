# PulseAI Deployment Notes (Vercel)

Deploy this repo as two Vercel projects:
- `covid-patient-api` - Flask backend from `webapp/backend`
- `covid-patient-app` - React frontend from `webapp/frontend`

## Backend (Vercel)
Backend lives in: `webapp/backend`

### Env vars required
- `MONGO_URI` = your MongoDB Atlas connection string (must include the database name at the end)
  - Example: `mongodb+srv://<user>:<pass>@<cluster>/<db>?retryWrites=true&w=majority`
- `FRONTEND_URL` = your deployed frontend base URL for CORS allowlist

### Vercel project settings
- Root Directory: `webapp/backend`
- Framework Preset: Other
- Build Command: leave empty
- Output Directory: leave empty
- Install Command: `pip install -r requirements.txt`

### Health check
After deployment, verify:
- `GET /health` (e.g. `https://covid-patient-api.vercel.app/health`)

## Frontend (Vercel)
Frontend lives in: `webapp/frontend`

The React app calls the backend using:
- `REACT_APP_API_URL` (build-time) if provided
- otherwise it falls back to `https://covid-patient-api.vercel.app`

### Vercel project settings
- Root Directory: `webapp/frontend`
- Framework Preset: Create React App
- Build Command: `npm run build`
- Output Directory: `build`

### Recommended frontend env var
- `REACT_APP_API_URL` = your Vercel backend URL

## Quick local test
1) Set `MONGO_URI` in your environment (or create `webapp/backend/.env` for local development)
2) Start backend: `python webapp/backend/app.py`
3) Start frontend: `cd webapp/frontend && npm install && npm start`
