import asyncio
from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from models import get_llm, get_embedding_model
from vectordb.qdrant import QdrantAdapter
import logging

logger = logging.getLogger(__name__)

async def invoke_with_retry(func, *args, **kwargs):
    """Retries a call if it fails with a 503 model loading error"""
    max_retries = 10
    delay = 5
    for i in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            err_str = str(e)
            if "503" in err_str and ("loading" in err_str.lower() or "unavailable" in err_str.lower()):
                logger.warning(f"Model is loading (503). Retrying in {delay}s... (Attempt {i+1}/{max_retries})")
                await asyncio.sleep(delay)
                continue
            raise e
    raise Exception("Max retries reached while waiting for model to load.")


class AgentState(TypedDict):
    question: str
    chat_history: List[BaseMessage]
    llm_model: str
    embedding_model: str
    collection_name: Annotated[str, "The name of the Qdrant collection to search"]
    context: str
    answer: str

def retrieve(state: AgentState):
    # Sync version (removed for brevity/cleanup if unused, but keeping structure)
    pass 

async def retrieve_node(state: AgentState):
    question = state["question"]
    emb_model_name = state["embedding_model"]
    
    embed_model = get_embedding_model(emb_model_name)
    # embed_query is usually sync in langchain integration unless using aembed_query
    query_vector = await invoke_with_retry(embed_model.aembed_query, question)
    
    collection_name = state.get("collection_name")
    if not collection_name:
        # Fallback to legacy naming
        base_model_name = emb_model_name.split(":")[0]
        safe_name = base_model_name.replace("-", "_").replace(".", "_").replace("/", "_")
        collection_name = f"rag_{safe_name}"

    
    db = QdrantAdapter()
    results = await db.search(collection_name, query_vector, limit=5)
    
    context = "\n\n".join([res["text"] for res in results])
    return {"context": context}

async def generate_node(state: AgentState):
    question = state["question"]
    context = state["context"]
    llm_name = state["llm_model"]
    
    llm = get_llm(llm_name)
    
    messages = [
        SystemMessage(content="You are a helpful assistant. Use the following context to answer the user's question. If you don't know, say so."),
        SystemMessage(content=f"Context:\n{context}"),
    ]
    # Add history
    messages.extend(state.get("chat_history", []))
    messages.append(HumanMessage(content=question))
    
    response = await invoke_with_retry(llm.ainvoke, messages)
    return {"answer": response.content}

# Define Graph
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()
