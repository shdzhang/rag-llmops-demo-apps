import asyncio
import logging

import mlflow
from dotenv import load_dotenv
from mlflow.genai.agent_server import get_invoke_function
from mlflow.genai.scorers import (
    Completeness,
    Fluency,
    RelevanceToQuery,
    Safety,
)
from mlflow.genai.simulators import ConversationSimulator
from mlflow.types.responses import ResponsesAgentRequest

# Load environment variables from .env if it exists
load_dotenv(dotenv_path=".env", override=True)
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)

# need to import agent for our @invoke-registered function to be found
from agent_server import agent  # noqa: F401

test_cases = [
    {
        "goal": "Understand the remote work and home office policies",
        "persona": "A new employee who just joined and wants to know about working from home.",
        "simulation_guidelines": [
            "Ask about remote work eligibility first, then follow up about home office equipment.",
            "Prefer short, direct questions",
        ],
    },
    {
        "goal": "Learn about parental leave and expense reimbursement",
        "persona": "An employee planning for a growing family who also needs to submit recent travel expenses.",
        "simulation_guidelines": [
            "Start with parental leave questions, then pivot to expense submission process.",
        ],
    },
]

simulator = ConversationSimulator(
    test_cases=test_cases,
    max_turns=5,
    user_model="databricks:/databricks-claude-sonnet-4-5",
)

# Get the invoke function that was registered via @invoke decorator in your agent
invoke_fn = get_invoke_function()
assert invoke_fn is not None, (
    "No function registered with the `@invoke` decorator found."
    "Ensure you have a function decorated with `@invoke()`."
)

# if invoke function is async, then we need to wrap it in a sync function
if asyncio.iscoroutinefunction(invoke_fn):

    def predict_fn(input: list[dict], **kwargs) -> dict:
        req = ResponsesAgentRequest(input=input)
        response = asyncio.run(invoke_fn(req))
        return response.model_dump()
else:

    def predict_fn(input: list[dict], **kwargs) -> dict:
        req = ResponsesAgentRequest(input=input)
        response = invoke_fn(req)
        return response.model_dump()


def evaluate():
    mlflow.genai.evaluate(
        data=simulator,
        predict_fn=predict_fn,
        scorers=[
            RelevanceToQuery(),
            Safety(),
            Fluency(),
            Completeness(),
        ],
    )
