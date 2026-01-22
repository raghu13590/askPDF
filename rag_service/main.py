"""
main.py - FastAPI entrypoint for the RAG Service

This module provides endpoints for document indexing, chat with retrieval-augmented generation, model listing, and health checks.

Endpoints:
- POST /index: Index a document for retrieval.
- POST /chat: Chat with RAG using LLM and embedding models.
- GET /status: Check if a collection exists in the vector DB.
- GET /models: List available LLM and embedding models from the LLM API/server.
- GET /health/model: Check if a specific model is ready.
- GET /health: Service health check.

Dependencies:
- FastAPI
- httpx
- dotenv
- rag, agent, vectordb.qdrant (local modules)
"""

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage
from httpx import AsyncClient

from agent import app as agent_app
from rag import index_document
from vectordb.qdrant import QdrantAdapter
from models import check_chat_model_ready, check_embed_model_ready

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="RAG Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IndexRequest(BaseModel):
    """Request body for /index endpoint."""
    text: str
    embedding_model: str
    metadata: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    """Request body for /chat endpoint."""
    question: str
    llm_model: str
    embedding_model: str
    collection_name: Optional[str] = None
    history: List[Dict[str, str]] = []  # list of {role: "user"|"assistant", content: "..."}


@app.post("/index")
async def index_endpoint(req: IndexRequest):
    """
    Index a document for retrieval-augmented generation.
    Args:
        req (IndexRequest): Document text, embedding model, and optional metadata.
    Returns:
        Result of indexing operation.
    """
    try:
        result = await index_document(
            text=req.text,
            embedding_model_name=req.embedding_model,
            metadata=req.metadata,
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Chat endpoint for retrieval-augmented generation.
    Args:
        req (ChatRequest): User question, LLM/embedding models, chat history, collection name.
    Returns:
        Answer and context from the agent.
    """
    try:
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
            "answer": "",
        }

        result = await agent_app.ainvoke(inputs)
        return {"answer": result["answer"], "context": result["context"]}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def status_endpoint(collection_name: str):
    """
    Check if a collection exists and is ready in the vector database.
    Args:
        collection_name (str): Name of the collection to check.
    Returns:
        Status and collection name.
    """
    try:
        db = QdrantAdapter()
        exists = await db.collection_exists(collection_name)
        return {"status": "ready" if exists else "not_ready", "collection": collection_name}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/models")
async def get_models():
    """
    Fetch available LLM and embedding models from the LLM API/server (OpenAI-compatible).
    Returns:
        List of model IDs or fallback defaults if the LLM API/server is unavailable.
    """
    llm_api_url = os.getenv("LLM_API_URL")
    try:
        if not llm_api_url.endswith("/v1"):
            llm_api_url = f"{llm_api_url}/v1"

        async with AsyncClient() as client:
            resp = await client.get(f"{llm_api_url}/models")
            if resp.status_code == 200:
                data = resp.json()
                print(f"LLM API/server Models Found: {data}", flush=True)
                return data
            else:
                error_msg = f"LLM API/server Fetch Failed {resp.status_code}: {resp.text}"
                print(error_msg, flush=True)
                raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        error_msg = f"Error fetching models from LLM API/server: {str(e)}"
        print(error_msg, flush=True)
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/health/model")
async def model_health_endpoint(model: str):
    """
    Check if a specific model is available and ready.
    Args:
        model (str): Model ID to check.
    Returns:
        Model readiness status.
    """
    ready = await is_chat_model_ready(model)
    return {"model": model, "ready": ready}


@app.get("/health/is_chat_model_ready")
async def is_chat_model_ready(model: str):
    """
    Check if a chat/LLM model is ready (probes readiness).
    Args:
        model (str): Model ID to check.
    Returns:
        Model readiness status.
    """
    ready = await check_chat_model_ready(model)
    return {"model": model, "chat_model_ready": ready}


@app.get("/health/is_embed_model_ready")
async def is_embed_model_ready(model: str):
    """
    Check if an embedding model is ready (probes readiness).
    Args:
        model (str): Model ID to check.
    Returns:
        Model readiness status.
    """
    ready = await check_embed_model_ready(model)
    return {"model": model, "embed_model_ready": ready}


@app.get("/health")
async def health():
    """
    Service health check endpoint.
    Returns:
        Status OK if service is running.
    """
    return {"status": "ok"}
