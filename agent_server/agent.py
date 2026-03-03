"""
RAG Agent - Corporate Affairs Chatbot on Databricks Apps

Migrated from Model Serving (ResponsesAgent class) to Databricks Apps
(@invoke/@stream async functions served by MLflow AgentServer).

Uses:
- Async @invoke/@stream decorators for the agent interface
- Databricks Vector Search for retrieval
- MLflow Prompt Registry for versioned prompt management
- Apps user authorization for per-user UC permissions
- DatabricksOpenAI (async) for Foundation Model API calls
"""

import os
import uuid
from typing import AsyncGenerator

import mlflow
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

try:
    mlflow.openai.autolog()
except Exception:
    pass

# ---------------------------------------------------------------------------
# Configuration - loaded from environment variables (set via .env or app.yaml)
# ---------------------------------------------------------------------------
LLM_ENDPOINT_NAME = os.environ.get("LLM_ENDPOINT", "databricks-claude-sonnet-4-5")
VECTOR_SEARCH_INDEX = os.environ.get("VECTOR_SEARCH_INDEX", "shidong_catalog.corp_affairs.docs_index")
PROMPT_NAME = os.environ.get("PROMPT_NAME", "shidong_catalog.corp_affairs.rag_prompt")
PROMPT_ALIAS = os.environ.get("PROMPT_ALIAS", "production")

# ---------------------------------------------------------------------------
# Lazy-initialised async OpenAI client
# ---------------------------------------------------------------------------
_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from databricks_openai import DatabricksOpenAI
        _openai_client = DatabricksOpenAI()
    return _openai_client


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
@mlflow.trace
def _retrieve_context(question: str) -> str:
    """Retrieve relevant documents from Vector Search.

    Uses the Databricks SDK WorkspaceClient which automatically picks up
    the app's service principal credentials (DATABRICKS_CLIENT_ID /
    DATABRICKS_CLIENT_SECRET / DATABRICKS_HOST) injected by Databricks Apps.
    """
    from databricks.sdk import WorkspaceClient

    try:
        w = WorkspaceClient()
        results = w.vector_search_indexes.query_index(
            index_name=VECTOR_SEARCH_INDEX,
            query_text=question,
            columns=["content", "source_file", "department"],
            num_results=5,
        )
        rows = results.result.data_array if results.result else []
        if not rows:
            return "No relevant documents found."
        return "\n\n".join(
            f"[Source: {row[1]} | Dept: {row[2]}]\n{row[0]}" for row in rows
        )
    except Exception as e:
        return f"Error retrieving documents: {e}"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
def _load_and_format_prompt(context: str, question: str) -> str:
    """Load the prompt from MLflow Prompt Registry and fill in variables.

    The prompt template contains {{context}} and {{question}} placeholders.
    Loading at query time with a short alias TTL (60s default) means prompt
    updates propagate without redeploying the agent.
    """
    try:
        prompt = mlflow.genai.load_prompt(
            f"prompts:/{PROMPT_NAME}@{PROMPT_ALIAS}"
        )
        return prompt.format(context=context, question=question)
    except Exception:
        return (
            "You are a helpful corporate affairs assistant. "
            "Answer the employee's question based ONLY on the provided context. "
            "If you don't know, say so. Cite your sources.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )


# ---------------------------------------------------------------------------
# Streaming entry point (primary logic)
# ---------------------------------------------------------------------------
@stream()
async def streaming(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    """Stream agent responses using RAG.

    Steps:
    1. Retrieve relevant documents from Vector Search
    2. Load and format prompt from MLflow Prompt Registry
    3. Stream the response from the LLM via Foundation Model API
    """
    user_message = ""
    for msg in request.input:
        if msg.role == "user":
            if isinstance(msg.content, str):
                user_message = msg.content
            elif isinstance(msg.content, list):
                user_message = " ".join(
                    item.text if hasattr(item, "text") else str(item)
                    for item in msg.content
                )

    context = _retrieve_context(user_message)
    formatted_prompt = _load_and_format_prompt(context=context, question=user_message)

    messages = [{"role": "user", "content": formatted_prompt}]

    client = _get_openai_client()
    item_id = f"msg_{uuid.uuid4().hex[:8]}"

    llm_stream = client.chat.completions.create(
        model=LLM_ENDPOINT_NAME,
        messages=messages,
        temperature=0.1,
        max_tokens=1000,
        stream=True,
    )

    full_text = ""
    for chunk in llm_stream:
        delta = chunk.choices[0].delta
        if delta.content:
            full_text += delta.content
            yield ResponsesAgentStreamEvent(
                type="response.output_text.delta",
                item_id=item_id,
                content_index=0,
                delta=delta.content,
            )

    yield ResponsesAgentStreamEvent(
        type="response.output_item.done",
        item={
            "id": item_id,
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": full_text}],
        },
    )


# ---------------------------------------------------------------------------
# Non-streaming entry point (collects stream output)
# ---------------------------------------------------------------------------
@invoke()
async def non_streaming(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    """Collect all streaming events and return the final response."""
    outputs = [
        event.item
        async for event in streaming(request)
        if event.type == "response.output_item.done"
    ]
    return ResponsesAgentResponse(output=outputs)
