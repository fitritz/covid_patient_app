# PulseAI Deployment Notes (Single Render Service)

Deploy this repo as one Render web service. The service builds the React frontend, then Flask serves both the frontend and the API from the same domain.

## Render Web Service
Create a new Web Service from the GitHub repo.

### Env vars required
- `MONGO_URI` = your MongoDB Atlas connection string (must include the database name at the end)
  - Example: `mongodb+srv://<user>:<pass>@<cluster>/<db>?retryWrites=true&w=majority`

### Render settings
- Name: `covid-patient-app`
- Language: `Python 3`
- Branch: `main`
- Root Directory: leave empty
- Build Command: `pip install -r webapp/backend/requirements.txt && cd webapp/frontend && npm ci && npm run build`
- Start Command: `cd webapp/backend && gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
- Instance Type: Free

### Verify
After deployment, verify:
- Frontend: `https://<your-render-service>.onrender.com`
- Health: `https://<your-render-service>.onrender.com/health`
- API info: `https://<your-render-service>.onrender.com/api`

## Quick local test
1) Set `MONGO_URI` in your environment (or create `webapp/backend/.env` for local development)
2) Start backend: `python webapp/backend/app.py`
3) Start frontend: `cd webapp/frontend && npm install && npm start`
