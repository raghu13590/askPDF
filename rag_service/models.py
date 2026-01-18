import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def get_llm(model_name: str, temperature: float = 0.0):
    """
    Returns a configured LLM client.
    Configured for OpenAI-compatible API (DMR).
    """
    base_url = os.getenv("DMR_BASE_URL", "http://host.docker.internal:12434")
    # Helper to clean URL if needed (e.g. ensure /v1 suffix if client requires it, 
    # but langgraph/langchain usually handles base_url + /chat/completions)
    # Typically ChatOpenAI expects base_url to be the root or root/v1
    
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        base_url=base_url,
        api_key="sk-no-key-required" # DMR usually doesn't need a real key
    )

def get_embedding_model(model_name: str):
    """
    Returns a configured Embedding model client.
    """
    base_url = os.getenv("DMR_BASE_URL", "http://host.docker.internal:12434")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"

    return OpenAIEmbeddings(
        model=model_name,
        base_url=base_url,
        api_key="sk-no-key-required",
        check_embedding_ctx_length=False # Disable check for local models
    )

import httpx
async def is_model_ready(model_name: str) -> bool:
    """
    Checks if the model is ready in DMR by performing a minimal probe.
    We are looking specifically for 503 (Loading Model) or 200 (Ready).
    """
    base_url = os.getenv("DMR_BASE_URL", "http://host.docker.internal:12434")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    
    try:
        async with httpx.AsyncClient() as client:
            # 1. Check if model exists in list first
            list_resp = await client.get(f"{base_url}/models", timeout=2.0)
            if list_resp.status_code != 200:
                return False
            
            models_data = list_resp.json()
            model_ids = [m['id'] for m in models_data.get('data', [])]
            if model_name not in model_ids:
                return False

            # 2. Probe with Chat Completion
            probe_payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1
            }
            
            try:
                probe_resp = await client.post(
                    f"{base_url}/chat/completions", 
                    json=probe_payload,
                    timeout=2.0 
                )
                if probe_resp.status_code == 200:
                    return True
                if probe_resp.status_code == 503:
                    return False
            except Exception:
                pass # Continue to next probe

            # 3. Probe with Embeddings (Backup)
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

            # 4. Fallback: If it's in the list and not explicitly returning 503, 
            # assume it's at least reachable and not "Loading".
            return True
    except Exception:
        return False
