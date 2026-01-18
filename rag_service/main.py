from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import httpx
import httpx
from rag import index_document
from agent import app as agent_app
from fastapi.middleware.cors import CORSMiddleware
from vectordb.qdrant import QdrantAdapter

app = FastAPI(title="RAG Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class IndexRequest(BaseModel):
    text: str
    embedding_model: str
    metadata: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    question: str
    llm_model: str
    embedding_model: str
    collection_name: Optional[str] = None
    history: List[Dict[str, str]] = [] # list of {role: "user"|"assistant", content: "..."}

@app.post("/index")
async def index_endpoint(req: IndexRequest):
    try:
        result = await index_document(
            text=req.text, 
            embedding_model_name=req.embedding_model,
            metadata=req.metadata
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        # Convert history
        from langchain_core.messages import HumanMessage, AIMessage
        chat_history = []
        for msg in req.history:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))
        
        inputs = {
            "question": req.question,
            "chat_history": chat_history,
            "llm_model": req.llm_model,
            "embedding_model": req.embedding_model,
            "collection_name": req.collection_name,
            "context": "",
            "answer": ""
        }
        
        result = await agent_app.ainvoke(inputs)
        return {"answer": result["answer"], "context": result["context"]}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def status_endpoint(collection_name: str):
    """Check if a collection exists and is ready"""
    try:
        db = QdrantAdapter()
        exists = await db.collection_exists(collection_name)
        return {"status": "ready" if exists else "not_ready", "collection": collection_name}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/models")
async def get_models():
    """Fetch available models from DMR"""
    dmr_url = os.getenv("DMR_BASE_URL", "http://host.docker.internal:12434")
    try:
        if not dmr_url.endswith("/v1"):
            dmr_url = f"{dmr_url}/v1"
            
        async with httpx.AsyncClient() as client:
            # Assuming standard OpenAI endpoint /v1/models
            resp = await client.get(f"{dmr_url}/models")
            if resp.status_code == 200:
                data = resp.json()
                print(f"DMR Models Found: {data}", flush=True)
                return data
            else:
                print(f"DMR Fetch Failed {resp.status_code}: {resp.text}", flush=True)
                # Fallback or error
                return {"data": [{"id": "ai/qwen3:latest"}, {"id": "ai/nomic-embed-text-v1.5:latest"}]}
    except Exception as e:
        # Return defaults if connection fails (e.g. while building)
        return {"data": [{"id": "ai/qwen3:latest"}, {"id": "ai/nomic-embed-text-v1.5:latest"}], "error": str(e)}

@app.get("/health/model")
async def model_health_endpoint(model: str):
    """Specific check for a model's availability"""
    from models import is_model_ready
    ready = await is_model_ready(model)
    return {"model": model, "ready": ready}

@app.get("/health")
async def health():
    return {"status": "ok"}
