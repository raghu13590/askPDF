

"""
main.py
--------
FastAPI application for PDF Text-to-Speech (TTS) and Retrieval-Augmented Generation (RAG) services.

Features:
- Upload PDF files, extract sentences and bounding boxes, and trigger RAG indexing
- Synthesize audio for sentences using TTS
- List available TTS voices/styles
- Check RAG indexing status
- Serve static and audio files

Environment Variables:
- RAG_SERVICE_URL: URL for the RAG service (default: http://rag-service:8000)
"""
import os
import uuid
import hashlib
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


# Local imports
from .tts import tts_sentence_to_wav, list_voice_styles
from .pdf_service import PDFService



API_PREFIX = "/api"
AUDIO_DIR = "/data/audio"
# RAG service URL (can be overridden by environment variable)
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://rag-service:8000")


app = FastAPI(title="PDF TTS")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure audio and static directories exist
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs("/static", exist_ok=True)



pdf_service = PDFService(static_dir="/static", rag_service_url=RAG_SERVICE_URL)

@app.post(f"{API_PREFIX}/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    embedding_model: str = Form(...)
):
    """
    Upload a PDF file, extract sentences and bounding boxes, and trigger RAG indexing.
    
    Args:
        background_tasks (BackgroundTasks): FastAPI background task manager.
        file (UploadFile): PDF file to upload.
        embedding_model (str): Name of the embedding model to use.
    Returns:
        dict: Result of the upload and processing.
    """
    return await pdf_service.process_upload(file, embedding_model, background_tasks)


@app.get(f"{API_PREFIX}/voices")
async def get_voices():
    """
    List available TTS voices/styles.
    
    Returns:
        dict: Available voices/styles for TTS.
    """
    voices = list_voice_styles()
    return {"voices": voices}


@app.post(f"{API_PREFIX}/tts")
async def synthesize_sentence(payload: dict):
    """
    Synthesize audio for a sentence using TTS.
    
    Args:
        payload (dict): Should contain 'text', 'voice', and optional 'speed'.
    Returns:
        dict: URL to the generated audio file.
    Raises:
        HTTPException: If 'text' is missing in the payload.
    """
    text = payload.get("text")
    voice = payload.get("voice")
    speed = payload.get("speed", 1.0)
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' in payload.")
    path = tts_sentence_to_wav(text, AUDIO_DIR, voice_style=voice, speed=speed)
    rel = os.path.relpath(path, "/")
    url = f"/{rel}"
    return {"audioUrl": url}


@app.get(f"{API_PREFIX}/rag_status")
async def check_rag_status(collection_name: str):
    """
    Check the status of RAG indexing for a collection.
    
    Args:
        collection_name (str): Name of the collection to check status for.
    Returns:
        dict: Status information from the RAG service.
    """
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{RAG_SERVICE_URL}/status", params={"collection_name": collection_name})
            return resp.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}


# Serve audio files
app.mount("/data", StaticFiles(directory="/data"), name="data")

# Mount static files last to avoid shadowing API routes
app.mount("/", StaticFiles(directory="/static", html=True), name="static")
