# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - App Deployment (Databricks Apps)
# MAGIC
# MAGIC This notebook deploys the agent as a **Databricks App** using Asset Bundles:
# MAGIC 1. Validates the champion model exists in UC
# MAGIC 2. Deploys the App using `databricks bundle deploy` + `databricks bundle run`
# MAGIC 3. Tests the deployed App endpoint
# MAGIC
# MAGIC ## Key Differences from Model Serving
# MAGIC
# MAGIC | Aspect | Model Serving (old) | Databricks Apps (new) |
# MAGIC |--------|--------------------|-----------------------|
# MAGIC | Deploy | `agents.deploy()` | `databricks bundle deploy` + `bundle run` |
# MAGIC | Server | MLflow Model Server | MLflow AgentServer (FastAPI) |
# MAGIC | Auth | OBO via `MODEL_SERVING_USER_CREDENTIALS` | App service principal + user auth |
# MAGIC | UI | Review App | Built-in Chat UI |
# MAGIC | Versioning | UC model versions | Git-based + Asset Bundles |

# COMMAND ----------
# MAGIC %pip install mlflow>=3.1 databricks-sdk
# MAGIC %restart_python

# COMMAND ----------
import mlflow
from mlflow import MlflowClient
from databricks.sdk import WorkspaceClient

# --- Configuration (from DAB job parameters) ---
CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
MODEL_NAME = dbutils.widgets.get("model_name")
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")
APP_NAME = dbutils.widgets.get("app_name")

UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient()
w = WorkspaceClient()

print(f"Deploying App: {APP_NAME}")
print(f"Model: {UC_MODEL_NAME}")
print(f"MLflow version: {mlflow.__version__}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: Verify Champion Model Exists

# COMMAND ----------

try:
    champion = client.get_model_version_by_alias(UC_MODEL_NAME, "champion")
    champion_version = champion.version
    print(f"Champion version: {champion_version}")
    print(f"  Run ID: {champion.run_id}")
except Exception:
    print("No 'champion' alias found - falling back to 'candidate'")
    champion = client.get_model_version_by_alias(UC_MODEL_NAME, "candidate")
    champion_version = champion.version
    print(f"Candidate version: {champion_version}")
    print(f"  Run ID: {champion.run_id}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: Deploy via Databricks Apps
# MAGIC
# MAGIC Databricks Apps deployment is handled by Asset Bundles:
# MAGIC - `databricks bundle deploy` uploads files and configures resources
# MAGIC - `databricks bundle run` starts/restarts the app
# MAGIC
# MAGIC For automated CI/CD, these commands run from a CI pipeline.
# MAGIC This notebook documents the process for reference.

# COMMAND ----------

print(f"""
Databricks Apps Deployment Instructions
=========================================

The agent is deployed as a Databricks App using Asset Bundles.
Run these commands from the project root directory:

  # 1. Validate bundle configuration
  databricks bundle validate

  # 2. Deploy the bundle (uploads files, configures resources)
  databricks bundle deploy

  # 3. Start/restart the app (REQUIRED after deploy)
  databricks bundle run corp_chatbot_app

After deployment, the app will be available at:
  https://<workspace-url>/apps/{APP_NAME}

The app includes:
  - Async AgentServer with @invoke/@stream handlers
  - Built-in Chat UI (no separate frontend needed)
  - MLflow tracing for observability
  - App-level resource grants (LLM endpoint, Vector Search)

Model Reference:
  UC Model: {UC_MODEL_NAME}@champion (v{champion_version})
  The app reads agent code directly from source — no model artifact download needed.
""")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: Check App Status (if already deployed)

# COMMAND ----------

try:
    app_info = w.apps.get(APP_NAME)
    app_status = app_info.app_status.state if app_info.app_status else "UNKNOWN"
    compute_status = app_info.compute_status.state if app_info.compute_status else "UNKNOWN"
    app_url = app_info.url or "Not available"

    print(f"App Status: {app_status}")
    print(f"Compute Status: {compute_status}")
    print(f"App URL: {app_url}")

    if "dbutils" in dir():
        dbutils.jobs.taskValues.set(key="app_url", value=app_url)
        dbutils.jobs.taskValues.set(key="app_name", value=APP_NAME)

except Exception as e:
    print(f"App '{APP_NAME}' not found or not yet deployed: {e}")
    print("Deploy using: databricks bundle deploy && databricks bundle run corp_chatbot_app")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Test the Deployed App (if running)

# COMMAND ----------

try:
    app_info = w.apps.get(APP_NAME)
    app_url = app_info.url

    if app_url and app_info.app_status and "RUNNING" in str(app_info.app_status.state):
        import json

        result = w.api_client.do(
            "POST",
            f"/apps/{APP_NAME}/invocations",
            body={"input": [{"role": "user", "content": "What is the company's remote work policy?"}]},
        )

        answer = ""
        for item in result.get("output", []):
            if item.get("type") == "message":
                for block in item.get("content", []):
                    if block.get("type") == "output_text":
                        answer += block.get("text", "")

        print("Test query successful!")
        print(f"Response: {answer[:500]}")
    else:
        print("App is not running yet. Deploy first, then re-run this cell.")
except Exception as e:
    print(f"Test query failed: {e}")
    print("Ensure the app is deployed and running.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Deployment Summary

# COMMAND ----------

print(f"""
Deployment Summary
==================
App Name:     {APP_NAME}
Model:        {UC_MODEL_NAME} v{champion_version}

What's included:
  - Async AgentServer with streaming support
  - Built-in Chat UI at the app URL
  - MLflow tracing (experiment: {EXPERIMENT_NAME})
  - App-level resource grants for LLM endpoint and Vector Search

To deploy from CLI:
  databricks bundle deploy
  databricks bundle run corp_chatbot_app

To test from CLI:
  TOKEN=$(databricks auth token | jq -r .access_token)
  curl -X POST <app-url>/invocations \\
    -H "Authorization: Bearer $TOKEN" \\
    -H "Content-Type: application/json" \\
    -d '{{"input": [{{"role": "user", "content": "Hello!"}}]}}'

Next Steps:
  1. Share the App URL with stakeholders
  2. Monitor via notebook 09_monitoring_dashboard.py
""")
