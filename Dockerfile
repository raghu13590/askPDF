# ---- Frontend build ----
FROM node:20-alpine AS frontend
WORKDIR /ui
COPY frontend/package.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ---- Backend runtime ----
FROM python:3.11-slim AS backend
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    curl \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz

RUN mkdir -p /models/supertonic /data/audio

# Clone Hugging Face repo with ONNX models + voice styles
RUN git lfs install && git clone https://huggingface.co/Supertone/supertonic /models/supertonic

# Copy backend app code
COPY backend/app /app/app

# Make sure /app/app exists before writing helper.py
RUN curl -o /app/app/helper.py https://raw.githubusercontent.com/supertone-inc/supertonic/main/py/helper.py

RUN mkdir -p /static
COPY --from=frontend /ui/out /static

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
