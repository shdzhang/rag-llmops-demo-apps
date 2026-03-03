# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Agent Evaluation with MLflow GenAI
# MAGIC
# MAGIC This notebook evaluates the RAG agent using **MLflow GenAI Evaluate**:
# MAGIC 1. Creates an evaluation dataset
# MAGIC 2. Loads the **actual agent** (`rag_agent.py`) from the candidate model logged in NB04
# MAGIC 3. Applies built-in LLM-as-judge scorers against the real agent
# MAGIC 4. Enforces quality gates for promotion

# COMMAND ----------
# MAGIC %pip install mlflow>=3.1 databricks-sdk databricks-agents databricks-vectorsearch databricks-openai pandas
# MAGIC # databricks-openai is still required by rag_agent.py when loaded via mlflow.pyfunc.load_model
# MAGIC %restart_python

# COMMAND ----------
import mlflow
import pandas as pd

# --- Configuration (from DAB job parameters) ---
CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
MODEL_NAME = dbutils.widgets.get("model_name")
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")

JUDGE_LLM_ENDPOINT = dbutils.widgets.get("judge_llm_endpoint")
VS_ENDPOINT = dbutils.widgets.get("vector_search_endpoint")

UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

# Use the same experiment as notebook 04 so all runs are grouped together
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_NAME)

# LLM judge URI — "databricks:/" prefix required by MLflow GenAI scorers.
# Combined with reduced scorers (1 instead of 3) and 8 test cases, this keeps
# evaluation under 15-20 minutes.
JUDGE_LLM = f"databricks:/{JUDGE_LLM_ENDPOINT}"

# Quality gates (metric names match scorer output keys)
# Additional quality checks (tone, citation, safety) run in production via NB09 External Monitor.
QUALITY_THRESHOLDS = {
    "correctness": 0.5,         # 50% of responses must be correct
}

print(f"MLflow version: {mlflow.__version__}")
print(f"Evaluating: {UC_MODEL_NAME}@candidate")
print(f"Judge LLM: {JUDGE_LLM}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: Prepare Evaluation Dataset
# MAGIC
# MAGIC Uses `mlflow.genai.datasets` to create a tracked evaluation dataset.

# COMMAND ----------

# Define test cases with expected outputs (ground truth).
# Key principle: expected responses should state the ESSENTIAL FACTS only.
# The Correctness scorer checks whether the agent's answer is semantically
# consistent with the expected response. Overly detailed expectations cause
# false negatives because the judge penalises any missing detail.
#
# Questions are written the way real employees would ask them --
# informal, sometimes vague, sometimes with typos or extra context.
eval_data = [
    # --- Remote Work Policy ---
    {
        "inputs": {"question": "Hey, I just joined last month — can I work from home yet?"},
        "expectations": {
            "expected_response": (
                "Employees may work remotely up to 3 days per week after completing "
                "the 90-day probation period."
            ),
        },
    },
    {
        "inputs": {"question": "I need a better chair for my home setup. Does the company help with that?"},
        "expectations": {
            "expected_response": (
                "There is a one-time $500 equipment stipend for home office setup."
            ),
        },
    },
    # --- Parental Leave Policy ---
    {
        "inputs": {"question": "My wife and I are expecting in June. How much time off do I get as the primary caregiver?"},
        "expectations": {
            "expected_response": (
                "Primary caregivers receive 16 weeks of paid leave at full salary."
            ),
        },
    },
    # --- Expense Policy ---
    {
        "inputs": {"question": "I had a client dinner last week and need to get reimbursed. What's the process?"},
        "expectations": {
            "expected_response": (
                "Expense reports must be submitted through the Concur system within "
                "30 days. Receipts are required for expenses over $25."
            ),
        },
    },
    {
        "inputs": {"question": "I need to buy $1,500 worth of software licenses. Who needs to sign off?"},
        "expectations": {
            "expected_response": (
                "Expenses between $500 and $2,000 require director approval."
            ),
        },
    },
    # --- IT Security ---
    {
        "inputs": {"question": "IT is making me reset my password again — what are the actual requirements?"},
        "expectations": {
            "expected_response": (
                "Passwords must be at least 12 characters and include uppercase, "
                "lowercase, numbers, and symbols. They must be changed every 90 days."
            ),
        },
    },
    {
        "inputs": {"question": "I think I left my work laptop in an Uber. What should I do?"},
        "expectations": {
            "expected_response": (
                "Lost or stolen devices must be reported to IT within 1 hour."
            ),
        },
    },
    # --- Company Holidays ---
    {
        "inputs": {"question": "Is the office closed between Christmas and New Year's?"},
        "expectations": {
            "expected_response": (
                "Yes, there is a company-wide shutdown from December 26 to December 31."
            ),
        },
    },
]

eval_df = pd.DataFrame(eval_data)
print(f"Evaluation dataset: {len(eval_df)} test cases")
eval_df.head()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: Load the Actual Agent and Define Predict Function
# MAGIC
# MAGIC We load the **real agent** (`rag_agent.py`) from the candidate model logged in
# MAGIC NB04 — the exact same code that will be deployed to production. This ensures
# MAGIC evaluation tests the real code path, not a reimplementation.

# COMMAND ----------

import time
from databricks.vector_search.client import VectorSearchClient

# --- Pre-flight: verify Vector Search index has data ---
# TRIGGERED indexes don't auto-sync. If the index is empty we trigger a sync
# ourselves and wait for it to complete (up to 15 min total).
VS_INDEX = f"{CATALOG}.{SCHEMA}.docs_index"
_vsc = VectorSearchClient(disable_notice=True)
_vs_index = _vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)

print(f"Checking Vector Search index readiness: {VS_INDEX}")
_sync_triggered = False
for _attempt in range(30):
    try:
        _test = _vs_index.similarity_search(
            query_text="test", columns=["content"], num_results=1
        )
        _rows = _test.get("result", {}).get("data_array", [])
        if _rows:
            print(f"  Index is ONLINE with data (attempt {_attempt + 1})")
            break
        else:
            if not _sync_triggered:
                print(f"  Index has 0 rows - triggering sync...")
                try:
                    _vs_index.sync()
                    _sync_triggered = True
                    print(f"  Sync triggered. Waiting for data to appear...")
                except Exception as _se:
                    print(f"  Sync trigger skipped ({_se}) - may already be in progress")
                    _sync_triggered = True
            else:
                print(f"  Still 0 rows - waiting 30s (attempt {_attempt + 1})")
    except Exception as _e:
        print(f"  Index not ready: {_e} - waiting 30s (attempt {_attempt + 1})")
    time.sleep(30)
else:
    raise RuntimeError(
        f"Vector Search index {VS_INDEX} has no data after 15 minutes. "
        "Check the source table and Delta Sync pipeline status."
    )

# COMMAND ----------

# Load the actual agent model logged by NB04 (candidate version)
_loaded_model = mlflow.pyfunc.load_model(f"models:/{UC_MODEL_NAME}@candidate")
print(f"Loaded model: {UC_MODEL_NAME}@candidate")


def predict_fn(question: str) -> str:
    """
    Wraps the real agent for mlflow.genai.evaluate().

    Calls the actual rag_agent.py code via the logged model — same retrieval,
    same prompt loading, same LLM call that runs in production. The parameter
    name ('question') must match the key in eval_data['inputs'].
    """
    request = {"input": [{"role": "user", "content": question}]}
    response = _loaded_model.predict(request)

    output = response.get("output", [])
    if output and isinstance(output[0], dict):
        return output[0].get("text", str(output[0]))
    return str(output)


# Quick test - verify the loaded agent works before running full evaluation
test_response = predict_fn("What is the remote work policy?")
print(f"Test response: {test_response[:300]}...")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: Define Scorers
# MAGIC
# MAGIC Using MLflow's built-in scorers and custom judges.

# COMMAND ----------

import os
from mlflow.genai.scorers import Correctness

# Concurrency settings for mlflow.genai.evaluate()
# Default is 10 workers each, but Foundation Model endpoints may throttle.
# Adjust based on your endpoint's rate limits.
os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = "4"          # parallel test cases
os.environ["MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS"] = "2"   # parallel scorers per test case

# Scorers for offline evaluation.
# We use only Correctness here (the quality gate metric) to keep evaluation fast.
# Additional quality checks (professional tone, source citation, safety,
# groundedness) run automatically in production via the External Monitor (NB09).
scorers = [
    Correctness(model=JUDGE_LLM),
]

print(f"Configured {len(scorers)} scorer(s) (judge model: {JUDGE_LLM})")
print(f"Concurrency: {os.environ['MLFLOW_GENAI_EVAL_MAX_WORKERS']} data workers, "
      f"{os.environ['MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS']} scorer workers")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Run Evaluation
# MAGIC
# MAGIC `mlflow.genai.evaluate()` orchestrates:
# MAGIC - Running the predict function on each test case
# MAGIC - Applying all scorers
# MAGIC - Logging results to MLflow experiment

# COMMAND ----------

eval_results = mlflow.genai.evaluate(
    data=eval_df,
    predict_fn=predict_fn,
    scorers=scorers,
)

print("Evaluation complete!")
print(f"\nMetrics:")
for metric, value in eval_results.metrics.items():
    print(f"  {metric}: {value}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 5: Analyze Results

# COMMAND ----------

# Display per-row results (MLflow 3.x: use search_traces instead of eval_table)
results_df = mlflow.search_traces(run_id=eval_results.run_id)
display(results_df) if "display" in dir() else print(results_df)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 6: Quality Gate Check
# MAGIC
# MAGIC Verify the agent meets minimum quality thresholds before promotion.

# COMMAND ----------

metrics = eval_results.metrics

# Check quality gates
gate_results = {}
all_passed = True

for metric_name, threshold in QUALITY_THRESHOLDS.items():
    # Find matching metric (metrics may have prefixes like 'correctness/mean')
    matching_metrics = {k: v for k, v in metrics.items() if metric_name in k.lower() and "mean" in k.lower()}

    if matching_metrics:
        metric_key, value = next(iter(matching_metrics.items()))
        passed = value >= threshold
        gate_results[metric_name] = {
            "value": value,
            "threshold": threshold,
            "passed": passed,
        }
        if not passed:
            all_passed = False
        status = "PASS" if passed else "FAIL"
        print(f"  {metric_name}: {value:.3f} (threshold: {threshold}) [{status}]")
    else:
        print(f"  {metric_name}: metric not found in results (available: {list(metrics.keys())})")

print(f"\nOverall Quality Gate: {'PASSED' if all_passed else 'FAILED'}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 7: Conditional Promotion
# MAGIC
# MAGIC If quality gates pass, promote candidate to champion.

# COMMAND ----------

if all_passed:
    client = mlflow.MlflowClient()

    # Get candidate version
    candidate = client.get_model_version_by_alias(UC_MODEL_NAME, "candidate")
    version = candidate.version

    # Promote to champion
    client.set_registered_model_alias(
        name=UC_MODEL_NAME,
        alias="champion",
        version=version,
    )

    print(f"Model version {version} promoted to 'champion'!")
    print(f"  {UC_MODEL_NAME}@champion -> version {version}")

    # Set task value for deployment notebook
    if "dbutils" in dir():
        dbutils.jobs.taskValues.set(key="promoted_version", value=version)
        dbutils.jobs.taskValues.set(key="quality_gate_passed", value=True)
else:
    print("Quality gates FAILED - model NOT promoted.")
    print("Review the evaluation results above and improve the agent.")

    if "dbutils" in dir():
        dbutils.jobs.taskValues.set(key="quality_gate_passed", value=False)

    # Fail the notebook so the pipeline does not proceed to deployment
    raise Exception("Quality gate check failed - model NOT promoted to champion")
