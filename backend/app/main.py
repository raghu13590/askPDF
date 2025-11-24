import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .pdf_parser import pdf_bytes_to_text
from .nlp import split_into_sentences
from .tts import tts_sentence_to_wav

API_PREFIX = "/api"
AUDIO_DIR = "/data/audio"

app = FastAPI(title="PDF TTS")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(AUDIO_DIR, exist_ok=True)

@app.post(f"{API_PREFIX}/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    data = await file.read()
    text = pdf_bytes_to_text(data)
    sentences = split_into_sentences(text)
    return {"sentences": sentences}

@app.post(f"{API_PREFIX}/tts")
async def synthesize_sentence(payload: dict):
    text = payload.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' in payload.")
    path = tts_sentence_to_wav(text, AUDIO_DIR)
    rel = os.path.relpath(path, "/")
    url = f"/{rel}"
    return {"audioUrl": url}

# Serve audio files
app.mount("/data", StaticFiles(directory="/data"), name="data")

# Serve frontend build (Next.js export) from /static — mount last
if os.path.isdir("/static"):
    app.mount("/", StaticFiles(directory="/static", html=True), name="static")
