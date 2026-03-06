"""
RAG Agent - Corporate Affairs Chatbot on Databricks Apps

Migrated from Model Serving (ResponsesAgent class) to Databricks Apps
(@invoke/@stream async functions served by MLflow AgentServer).

Uses:
- Async @invoke/@stream decorators for the agent interface
- Databricks Vector Search for retrieval
- MLflow Prompt Registry for versioned prompt management
- Apps user authorization for per-user UC permissions
- AsyncDatabricksOpenAI for non-blocking Foundation Model API calls
"""

import asyncio
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
        from databricks_openai import AsyncDatabricksOpenAI
        _openai_client = AsyncDatabricksOpenAI()
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
def _load_and_format_prompt(context: str, question: str) -> tuple[str, dict]:
    """Load the prompt from MLflow Prompt Registry and fill in variables.

    The prompt template contains {{context}} and {{question}} placeholders.
    Loading at query time with a short alias TTL (60s default) means prompt
    updates propagate without redeploying the agent.

    Returns (formatted_text, model_config) so callers can use the LLM
    parameters registered alongside the prompt version (temperature, etc.).
    """
    try:
        prompt = mlflow.genai.load_prompt(
            f"prompts:/{PROMPT_NAME}@{PROMPT_ALIAS}"
        )
        model_config = dict(prompt.model_config) if prompt.model_config else {}
        return prompt.format(context=context, question=question), model_config
    except Exception:
        return (
            "You are a helpful corporate affairs assistant. "
            "Answer the employee's question based ONLY on the provided context. "
            "If you don't know, say so. Cite your sources.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        ), {}


# ---------------------------------------------------------------------------
# Streaming entry point (primary logic)
# ---------------------------------------------------------------------------
def _extract_message_text(msg) -> str:
    """Extract plain text from a message's content field."""
    if isinstance(msg.content, str):
        return msg.content
    if isinstance(msg.content, list):
        return " ".join(
            item.text if hasattr(item, "text") else str(item)
            for item in msg.content
        )
    return str(msg.content) if msg.content else ""


@stream()
async def streaming(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    """Stream agent responses using RAG.

    Steps:
    1. Extract conversation history (multi-turn) and last user message
    2. Retrieve relevant documents from Vector Search (offloaded to thread)
    3. Load and format prompt from MLflow Prompt Registry
    4. Stream the response from the LLM via async Foundation Model API
    """
    # Build conversation history; use the last user message for retrieval
    history: list[dict[str, str]] = []
    last_user_message = ""
    for msg in request.input:
        text = _extract_message_text(msg)
        if msg.role in ("user", "assistant"):
            history.append({"role": msg.role, "content": text})
        if msg.role == "user" and text:
            last_user_message = text

    # Offload sync Vector Search SDK call to a thread to avoid blocking
    context = await asyncio.to_thread(_retrieve_context, last_user_message)
    formatted_prompt, model_config = _load_and_format_prompt(
        context=context, question=last_user_message
    )

    # System prompt from registry, followed by full conversation history
    messages: list[dict[str, str]] = [
        {"role": "system", "content": formatted_prompt},
        *history,
    ]

    client = _get_openai_client()
    item_id = f"msg_{uuid.uuid4().hex[:8]}"

    llm_stream = await client.chat.completions.create(
        model=LLM_ENDPOINT_NAME,
        messages=messages,
        temperature=model_config.get("temperature", 0.1),
        max_tokens=model_config.get("max_tokens", 1000),
        stream=True,
    )

    full_text = ""
    async for chunk in llm_stream:
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
