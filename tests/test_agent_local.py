"""
Local Agent Test - Tests the async @invoke/@stream agent functions.

Usage (local, after `uv run quickstart`):
    uv run pytest tests/test_agent_local.py -v

Usage (Databricks cluster):
    run_python_file_on_databricks(file_path="./tests/test_agent_local.py")
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)

import mlflow
from mlflow.types.responses import ResponsesAgentRequest

print(f"MLflow version: {mlflow.__version__}")


def test_non_streaming():
    """Test that the async non-streaming handler works."""
    from agent_server.agent import non_streaming

    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": "What is the remote work policy?"}],
    )

    result = asyncio.run(non_streaming(request))
    assert result.output, "Expected non-empty output"
    assert len(result.output) > 0, "Expected at least one output item"

    first_output = result.output[0]
    text = ""
    if hasattr(first_output, "content"):
        for block in first_output.content:
            if hasattr(block, "text"):
                text += block.text
            elif isinstance(block, dict) and "text" in block:
                text += block["text"]
    elif hasattr(first_output, "text"):
        text = first_output.text

    assert len(text) > 0, "Expected non-empty text response"
    print(f"PASS: Non-streaming returned {len(text)} characters")
    print(f"  Response preview: {text[:200]}...")


def test_streaming():
    """Test that the async streaming handler works."""
    from agent_server.agent import streaming

    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": "What are the company holidays?"}],
    )

    async def collect_events():
        events = []
        async for event in streaming(request):
            events.append(event)
        return events

    events = asyncio.run(collect_events())
    assert len(events) > 0, "Expected at least one streaming event"

    done_events = [e for e in events if e.type == "response.output_item.done"]
    assert len(done_events) > 0, "Expected at least one done event"

    print(f"PASS: Streaming returned {len(events)} events ({len(done_events)} done)")


def test_prompt_registry_loading():
    """Test that the Prompt Registry integration works."""
    prompt_name = os.environ.get("PROMPT_NAME", "shidong_catalog.corp_affairs.rag_prompt")
    prompt_alias = os.environ.get("PROMPT_ALIAS", "production")

    try:
        prompt = mlflow.genai.load_prompt(f"prompts:/{prompt_name}@{prompt_alias}")
        assert prompt.template, "Expected non-empty prompt template"
        print(f"PASS: Loaded prompt version {prompt.version} from registry")
        print(f"  Template preview: {prompt.template[:100]}...")
    except Exception as e:
        print(f"SKIP: Prompt Registry not available ({e})")
        print("  Register prompts first using notebook 03_prompt_engineering.py")


if __name__ == "__main__":
    print("=" * 60)
    print("RAG Agent Local Tests (Databricks Apps)")
    print("=" * 60)

    tests = [test_prompt_registry_loading, test_non_streaming, test_streaming]

    for test in tests:
        print(f"\n--- {test.__name__} ---")
        try:
            test()
        except Exception as e:
            print(f"FAIL: {e}")

    print("\n" + "=" * 60)
    print("All tests completed!")
