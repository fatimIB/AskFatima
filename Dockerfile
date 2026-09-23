# =========================
# 1. Build the React frontend
# =========================
FROM node:22-slim AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package*.json ./

RUN npm ci

COPY frontend/ .

RUN npm run build


# =========================
# 2. Run FastAPI + React
# =========================
FROM python:3.12-slim

WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt ./backend/requirements.txt

RUN pip install --no-cache-dir torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend
COPY backend ./backend

# Copy React production build
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Hugging Face Spaces uses port 7860
EXPOSE 7860

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]