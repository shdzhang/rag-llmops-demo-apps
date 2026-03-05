# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Agent Evaluation with MLflow GenAI
# MAGIC
# MAGIC Evaluates the RAG agent using **MLflow GenAI Evaluate**:
# MAGIC 1. Creates an evaluation dataset
# MAGIC 2. Imports the agent code **directly** from `agent_server/agent.py`
# MAGIC 3. Applies built-in LLM-as-judge scorers
# MAGIC 4. Enforces quality gates
# MAGIC
# MAGIC ## Apps LLMOps Model
# MAGIC
# MAGIC In the Apps deployment model, the agent code is the deployment artifact
# MAGIC (deployed via `databricks bundle deploy`), not an MLflow model version.
# MAGIC We evaluate the code directly — no `log_model()` / `load_model()` round-trip.

# COMMAND ----------
# MAGIC %pip install mlflow>=3.1 databricks-sdk databricks-agents databricks-vectorsearch databricks-openai pandas
# MAGIC %restart_python

# COMMAND ----------
import os
import sys
import mlflow
import pandas as pd

# --- Configuration (from DAB job parameters) ---
CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")

JUDGE_LLM_ENDPOINT = dbutils.widgets.get("judge_llm_endpoint")
VS_ENDPOINT = dbutils.widgets.get("vector_search_endpoint")

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_NAME)

JUDGE_LLM = f"databricks:/{JUDGE_LLM_ENDPOINT}"

QUALITY_THRESHOLDS = {
    "correctness": 0.5,
}

print(f"MLflow version: {mlflow.__version__}")
print(f"Judge LLM: {JUDGE_LLM}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: Prepare Evaluation Dataset

# COMMAND ----------

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
print(f"Static evaluation cases: {len(eval_df)}")

# --- Load regression cases from production feedback loop ---
EVAL_DATASET_TABLE = f"{CATALOG}.{SCHEMA}.eval_dataset"
try:
    regression_sdf = spark.table(EVAL_DATASET_TABLE)
    regression_pd = regression_sdf.select("question").distinct().toPandas()
    if not regression_pd.empty:
        static_questions = set(row["question"] for row in eval_data)
        new_cases = []
        for _, row in regression_pd.iterrows():
            q = row["question"]
            if q not in static_questions:
                new_cases.append({
                    "inputs": {"question": q},
                    "expectations": {},
                })
        if new_cases:
            regression_df = pd.DataFrame(new_cases)
            eval_df = pd.concat([eval_df, regression_df], ignore_index=True)
            print(f"Production regression cases added: {len(new_cases)}")
except Exception as _e:
    print(f"No production eval dataset yet ({type(_e).__name__}) — using static cases only")

print(f"Total evaluation cases: {len(eval_df)}")
eval_df.head()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: Pre-flight Check & Import Agent
# MAGIC
# MAGIC We import the agent code directly from `agent_server/agent.py` — the exact
# MAGIC same code that runs in the deployed Databricks App. No `log_model()` /
# MAGIC `load_model()` round-trip needed.

# COMMAND ----------

import time
from databricks.vector_search.client import VectorSearchClient

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
                except Exception as _se:
                    print(f"  Sync trigger skipped ({_se})")
                    _sync_triggered = True
            else:
                print(f"  Still 0 rows - waiting 30s (attempt {_attempt + 1})")
    except Exception as _e:
        print(f"  Index not ready: {_e} - waiting 30s (attempt {_attempt + 1})")
    time.sleep(30)
else:
    raise RuntimeError(
        f"Vector Search index {VS_INDEX} has no data after 15 minutes."
    )

# COMMAND ----------

# Set env vars so the agent code picks up the right config
try:
    os.environ["LLM_ENDPOINT"] = dbutils.widgets.get("llm_endpoint")
except Exception:
    os.environ.setdefault("LLM_ENDPOINT", "databricks-claude-sonnet-4-5")
os.environ["VECTOR_SEARCH_INDEX"] = VS_INDEX
os.environ.setdefault("PROMPT_NAME", f"{CATALOG}.{SCHEMA}.rag_prompt")
os.environ.setdefault("PROMPT_ALIAS", "production")

# Add the bundle root to sys.path so we can import agent_server
notebook_path = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook().getContext().notebookPath().get()
)
bundle_root = os.path.dirname(os.path.dirname(notebook_path))
if not bundle_root.startswith("/Workspace"):
    bundle_root = f"/Workspace{bundle_root}"
if bundle_root not in sys.path:
    sys.path.insert(0, bundle_root)

from agent_server.agent import _retrieve_context, _load_and_format_prompt, _get_openai_client, LLM_ENDPOINT_NAME

print(f"Agent imported directly from agent_server.agent")
print(f"  LLM endpoint: {LLM_ENDPOINT_NAME}")
print(f"  VS index: {VS_INDEX}")

# COMMAND ----------

def predict_fn(question: str) -> str:
    """
    Calls the same retrieval and LLM pipeline as the deployed agent.

    Runs _retrieve_context -> _load_and_format_prompt -> LLM call, matching
    the exact code path in agent_server/agent.py.
    """
    context = _retrieve_context(question)
    formatted_prompt = _load_and_format_prompt(context=context, question=question)
    messages = [{"role": "user", "content": formatted_prompt}]

    client = _get_openai_client()
    response = client.chat.completions.create(
        model=LLM_ENDPOINT_NAME,
        messages=messages,
        temperature=0.1,
        max_tokens=1000,
    )
    return response.choices[0].message.content


test_response = predict_fn("What is the remote work policy?")
print(f"Test response: {test_response[:300]}...")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: Define Scorers

# COMMAND ----------

from mlflow.genai.scorers import Correctness

os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = "4"
os.environ["MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS"] = "2"

scorers = [
    Correctness(model=JUDGE_LLM),
]

print(f"Configured {len(scorers)} scorer(s) (judge model: {JUDGE_LLM})")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Run Evaluation

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

results_df = mlflow.search_traces(run_id=eval_results.run_id)
if not results_df.empty:
    safe_cols = [c for c in ["trace_id", "state", "execution_duration", "status", "execution_time_ms"] if c in results_df.columns]
    print(results_df[safe_cols].to_string() if safe_cols else f"{len(results_df)} traces found")
else:
    print("No traces found")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 6: Quality Gate

# COMMAND ----------

metrics = eval_results.metrics

gate_results = {}
all_passed = True

for metric_name, threshold in QUALITY_THRESHOLDS.items():
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
# MAGIC ## Step 7: Record Results
# MAGIC
# MAGIC In the Apps model, quality gates inform deployment decisions but don't
# MAGIC drive model version promotion. The evaluation run ID is recorded for
# MAGIC traceability — CI/CD uses this to gate the `bundle deploy` step.

# COMMAND ----------

if all_passed:
    print(f"Quality gate PASSED — safe to deploy via `databricks bundle deploy`")
    if "dbutils" in dir():
        dbutils.jobs.taskValues.set(key="quality_gate_passed", value=True)
        dbutils.jobs.taskValues.set(key="eval_run_id", value=eval_results.run_id)
else:
    print("Quality gate FAILED — do NOT deploy.")
    if "dbutils" in dir():
        dbutils.jobs.taskValues.set(key="quality_gate_passed", value=False)
    raise Exception("Quality gate failed — agent does not meet minimum quality thresholds")
