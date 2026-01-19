"""
Model utilities for LLM API/OpenAI-compatible APIs.
"""

import os
from dotenv import load_dotenv
load_dotenv()
import httpx
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def _get_base_url() -> str:
    """Get the LLM API base URL, ensuring it ends with /v1."""
    base_url = os.getenv("LLM_API_URL")
    return base_url if base_url.endswith("/v1") else f"{base_url}/v1"

def get_llm(model_name: str, temperature: float = 0.0):
    """
    Return a configured ChatOpenAI client for the given model.
    """
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        base_url=_get_base_url(),
        api_key="sk-no-key-required"
    )

def get_embedding_model(model_name: str):
    """
    Return a configured OpenAIEmbeddings client for the given model.
    """
    return OpenAIEmbeddings(
        model=model_name,
        base_url=_get_base_url(),
        api_key="sk-no-key-required",
        check_embedding_ctx_length=False
    )

async def is_model_ready(model_name: str) -> bool:
    """
    Check if the model is ready in the LLM API/server by probing the API.
    Returns True if ready, False if not ready or not found.
    """
    base_url = _get_base_url()
    try:
        async with httpx.AsyncClient() as client:
            # 1. Check if model exists
            resp = await client.get(f"{base_url}/models", timeout=2.0)
            if resp.status_code != 200:
                return False
            model_ids = [m['id'] for m in resp.json().get('data', [])]
            if model_name not in model_ids:
                return False

            # 2. Probe with Chat Completion
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1
            }
            try:
                chat_resp = await client.post(f"{base_url}/chat/completions", json=payload, timeout=2.0)
                if chat_resp.status_code == 200:
                    return True
                if chat_resp.status_code == 503:
                    return False
            except Exception:
                pass

            # 3. Probe with Embeddings
            try:
                emb_resp = await client.post(
                    f"{base_url}/embeddings",
                    json={"model": model_name, "input": "hi"},
                    timeout=2.0
                )
                if emb_resp.status_code == 200:
                    return True
                if emb_resp.status_code == 503:
                    return False
            except Exception:
                pass

            # 4. Fallback: If model is listed and not 503, assume reachable
            return True
    except Exception:
        return False
