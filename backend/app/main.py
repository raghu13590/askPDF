import os
import httpx
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .pdf_parser import extract_text_with_coordinates
from .nlp import split_into_sentences
from .tts import tts_sentence_to_wav, list_voice_styles

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
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...), embedding_model: str = Form(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    
    if not embedding_model:
        raise HTTPException(status_code=400, detail="Please provide an embedding_model.")
    
    # Generate unique ID for this upload
    upload_id = str(uuid.uuid4())
    pdf_filename = f"{upload_id}.pdf"
    pdf_path = os.path.join("/static", pdf_filename)
    os.makedirs("/static", exist_ok=True)
    
    content = await file.read()
    with open(pdf_path, "wb") as f:
        f.write(content)
        
    text, char_map = extract_text_with_coordinates(content)
    
    sentences = split_into_sentences(text)
    
    # Map sentences to bounding boxes by finding each sentence in the extracted text
    current_idx = 0
    enriched_sentences = []
    
    for s in sentences:
        s_text = s["text"]
        
        # 1. Create a "clean" version of the sentence for matching (no whitespace)
        s_text_clean = "".join(s_text.split())
        
        if not s_text_clean:
            s["bboxes"] = []
            enriched_sentences.append(s)
            continue
            
        # 2. Scan 'text' starting from current_idx to find the sequence of chars in s_text_clean
        match_start = -1
        match_end = -1
        s_ptr = 0
        
        temp_idx = current_idx
        
        # Search for the sentence in the text
        while temp_idx < len(text) and s_ptr < len(s_text_clean):
            char = text[temp_idx]
            
            # Skip whitespace in the source text
            if char.isspace():
                temp_idx += 1
                continue
                
            # Check for match
            if char == s_text_clean[s_ptr]:
                if match_start == -1:
                    match_start = temp_idx
                s_ptr += 1
                temp_idx += 1
            else:
                # Mismatch
                if match_start != -1:
                    # We were matching but failed. Reset and try again from next char.
                    # This handles cases where the same word appears multiple times.
                    temp_idx = match_start + 1
                    match_start = -1
                    s_ptr = 0
                else:
                    # Haven't found start yet, keep looking
                    temp_idx += 1
        
        # Check if we found the full sentence
        if match_start != -1 and s_ptr == len(s_text_clean):
            match_end = temp_idx
            # Extract bboxes for the matched range
            bboxes = char_map[match_start:match_end]
            s["bboxes"] = bboxes
            current_idx = match_end
        else:
            # If strict match fails, we skip this sentence (or could log a warning)
            # We no longer fallback to simple find() as it's unreliable with layout changes
            s["bboxes"] = []
            
        enriched_sentences.append(s)

    # Trigger RAG Indexing
    rag_url = os.getenv("RAG_SERVICE_URL", "http://rag-service:8000")
    
    async def call_rag(txt: str, metadata: dict, emb_model: str):
        async with httpx.AsyncClient() as client:
            try:
                await client.post(
                    f"{rag_url}/index", 
                    json={
                        "text": txt,
                        "embedding_model": emb_model,
                        "metadata": metadata
                    },
                    timeout=300.0  # 5 minutes - indexing with embeddings can take time
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"RAG Indexing failed: {repr(e)}", flush=True)

    background_tasks.add_task(call_rag, text, {"filename": file.filename, "upload_id": upload_id}, embedding_model)

    return {"sentences": enriched_sentences, "pdfUrl": f"/{pdf_filename}"}

@app.get(f"{API_PREFIX}/voices")
async def get_voices():
    voices = list_voice_styles()
    return {"voices": voices}

@app.post(f"{API_PREFIX}/tts")
async def synthesize_sentence(payload: dict):
    text = payload.get("text")
    voice = payload.get("voice") # No default here, let tts_sentence_to_wav handle it
    speed = payload.get("speed", 1.0)
    
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' in payload.")
        
    path = tts_sentence_to_wav(text, AUDIO_DIR, voice_style=voice, speed=speed)
    rel = os.path.relpath(path, "/")
    url = f"/{rel}"
    return {"audioUrl": url}

# Serve audio files
app.mount("/data", StaticFiles(directory="/data"), name="data")

# Mount static files last to avoid shadowing API routes
# Mount static files last to avoid shadowing API routes
os.makedirs("/static", exist_ok=True)
app.mount("/", StaticFiles(directory="/static", html=True), name="static")
