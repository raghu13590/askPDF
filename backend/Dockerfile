FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    curl \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements from the context (root)
COPY backend/requirements.txt /app/backend/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/backend/requirements.txt
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz

# Create data and model directories
RUN mkdir -p /data/audio /models/kokoro

# Clone Kokoro repo to get voices
RUN git clone --depth 1 https://github.com/hexgrad/kokoro /tmp/kokoro-repo && \
    mv /tmp/kokoro-repo/voices /models/kokoro/voices && \
    rm -rf /tmp/kokoro-repo

# Copy backend app code
COPY backend/app /app/app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
