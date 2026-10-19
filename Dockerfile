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

# Copy backend source code and knowledge documents
COPY backend ./backend

# Keep Hugging Face models in the image
ENV HF_HOME=/app/.cache/huggingface

# Download the embedding and cross-encoder models
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"


# Build the Chroma vector database inside the image
RUN python -m backend.app.rag.vector_store

# Copy React production build
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

EXPOSE 7860

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]