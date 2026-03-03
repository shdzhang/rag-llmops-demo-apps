# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Agent Build: Log Agent to MLflow for Evaluation
# MAGIC
# MAGIC In the Apps deployment model, the agent code lives in `agent_server/agent.py`
# MAGIC and is deployed directly as a Databricks App (no `log_model` required for
# MAGIC serving). However, we still log a model to MLflow for:
# MAGIC 1. Tracking agent versions and configurations
# MAGIC 2. Running offline evaluation with `mlflow.genai.evaluate()`
# MAGIC 3. Maintaining the candidate/champion alias workflow
# MAGIC
# MAGIC This notebook logs the agent code and config, then sets the "candidate" alias.

# COMMAND ----------
# MAGIC %pip install mlflow>=3.1 databricks-sdk databricks-vectorsearch databricks-openai databricks-agents
# MAGIC %restart_python

# COMMAND ----------
import mlflow
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
)

# --- Configuration (from DAB job parameters) ---
CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
MODEL_NAME = dbutils.widgets.get("model_name")
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")

LLM_ENDPOINT = dbutils.widgets.get("llm_endpoint")
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
VS_INDEX = f"{CATALOG}.{SCHEMA}.docs_index"
PROMPT_NAME = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('prompt_base_name')}"

print(f"MLflow version: {mlflow.__version__}")
print(f"UC Model: {UC_MODEL_NAME}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: Set MLflow Experiment

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

try:
    mlflow.set_experiment(EXPERIMENT_NAME)
except Exception:
    mlflow.set_experiment(f"/Users/default/dev_{MODEL_NAME}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: Resolve Agent File Path

# COMMAND ----------

import os

try:
    notebook_path = (
        dbutils.notebook.entry_point.getDbutils()
        .notebook().getContext().notebookPath().get()
    )
    bundle_root = os.path.dirname(os.path.dirname(notebook_path))
    if not bundle_root.startswith("/Workspace"):
        bundle_root = f"/Workspace{bundle_root}"
    agent_file = os.path.join(bundle_root, "agent_server", "agent.py")
except Exception:
    agent_file = os.path.normpath(
        os.path.join(os.getcwd(), "..", "agent_server", "agent.py")
    )

print(f"Agent file: {agent_file}")
assert os.path.exists(agent_file), f"Agent file not found at {agent_file}"

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: Log the Agent to MLflow
# MAGIC
# MAGIC We log the agent for evaluation tracking and version management.
# MAGIC Deployment is handled separately by the Databricks App (notebook 07).

# COMMAND ----------

resources = [
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT),
    DatabricksVectorSearchIndex(index_name=VS_INDEX),
]

pip_requirements = [
    "mlflow>=3.1",
    "databricks-vectorsearch",
    "databricks-sdk",
    "databricks-openai",
]

input_example = {
    "input": [{"role": "user", "content": "What is the remote work policy?"}]
}

agent_config = {
    "llm_endpoint": LLM_ENDPOINT,
    "vector_search_index": VS_INDEX,
    "prompt_name": PROMPT_NAME,
    "prompt_alias": "production",
}

with mlflow.start_run(run_name="rag_agent_build") as run:
    mlflow.log_params({**agent_config, "agent_type": "Apps-AsyncResponsesAgent"})

    prompt_uri = f"prompts:/{PROMPT_NAME}@production"

    model_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model=agent_file,
        model_config=agent_config,
        resources=resources,
        pip_requirements=pip_requirements,
        input_example=input_example,
        prompts=[prompt_uri],
    )

    logged_run_id = run.info.run_id
    print(f"Agent logged successfully!")
    print(f"  Run ID: {logged_run_id}")
    print(f"  Model URI: {model_info.model_uri}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Register Model to Unity Catalog

# COMMAND ----------

uc_model_info = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=UC_MODEL_NAME,
)

print(f"Registered model: {uc_model_info.name}")
print(f"  Version: {uc_model_info.version}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 5: Set "Candidate" Alias for Evaluation

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()

client.set_registered_model_alias(
    name=UC_MODEL_NAME,
    alias="candidate",
    version=uc_model_info.version,
)

print(f"Alias 'candidate' set to version {uc_model_info.version}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 6: Output for Downstream Tasks

# COMMAND ----------

if "dbutils" in dir():
    dbutils.jobs.taskValues.set(key="logged_run_id", value=logged_run_id)
    dbutils.jobs.taskValues.set(key="model_version", value=uc_model_info.version)

print(f"\nReady for evaluation!")
print(f"  Run ID: {logged_run_id}")
print(f"  Model Version: {uc_model_info.version}")
print(f"  UC Model: {UC_MODEL_NAME}@candidate")
print(f"\nNote: Deployment is handled via Databricks Apps (notebook 07)")
