# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Model Registration & Alias Management
# MAGIC
# MAGIC This notebook manages model versions in Unity Catalog:
# MAGIC 1. Verifies the champion model after evaluation
# MAGIC 2. Adds descriptions and tags to model versions
# MAGIC 3. Manages model lifecycle via **aliases** (not deprecated stages)
# MAGIC
# MAGIC **Note**: Model promotion (candidate -> champion) is handled automatically
# MAGIC in notebook 05 when quality gates pass. This notebook is for manual
# MAGIC management and governance tasks.

# COMMAND ----------
# MAGIC %pip install mlflow>=3.1 databricks-sdk
# MAGIC %restart_python

# COMMAND ----------
import mlflow
from mlflow import MlflowClient

# --- Configuration (from DAB job parameters) ---
CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
MODEL_NAME = dbutils.widgets.get("model_name")
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")

UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient()

print(f"Managing model: {UC_MODEL_NAME}")
print(f"Experiment: {EXPERIMENT_NAME}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: View Current Model Versions & Aliases

# COMMAND ----------

# List all versions
model = client.get_registered_model(UC_MODEL_NAME)
print(f"Model: {model.name}")
print(f"Description: {model.description or '(none)'}")

# Show aliases (UC client returns aliases as dict: {"alias_name": "version_number"})
print(f"\nAliases:")
aliases = model.aliases or {}
if isinstance(aliases, dict):
    for alias_name, version in aliases.items():
        print(f"  @{alias_name} -> version {version}")
else:
    for alias in aliases:
        print(f"  {alias}")

# List recent versions
print(f"\nRecent versions:")
versions = client.search_model_versions(f"name='{UC_MODEL_NAME}'")
for v in sorted(versions, key=lambda x: int(x.version), reverse=True)[:5]:
    tags = v.tags if isinstance(v.tags, dict) else {}
    print(f"  v{v.version}: status={v.status}, tags={tags}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: Verify Champion Model

# COMMAND ----------

try:
    champion = client.get_model_version_by_alias(UC_MODEL_NAME, "champion")
    print(f"Champion model: version {champion.version}")
    print(f"  Created: {champion.creation_timestamp}")
    print(f"  Run ID: {champion.run_id}")
except Exception as e:
    print(f"No champion alias set: {e}")
    print("Run notebook 05 (evaluation) to promote a model to champion.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: Add Description & Tags to Model Version
# MAGIC
# MAGIC Tags are useful for governance, auditing, and filtering.

# COMMAND ----------

try:
    champion = client.get_model_version_by_alias(UC_MODEL_NAME, "champion")

    # Update model version description
    client.update_model_version(
        name=UC_MODEL_NAME,
        version=champion.version,
        description=(
            "Corporate Affairs RAG Chatbot - ResponsesAgent with "
            "MLflow Prompt Registry, Vector Search retrieval, and OBO support."
        ),
    )

    # Add governance tags
    client.set_model_version_tag(UC_MODEL_NAME, champion.version, "evaluated", "true")
    client.set_model_version_tag(UC_MODEL_NAME, champion.version, "evaluation_notebook", "05_agent_evaluation")
    client.set_model_version_tag(UC_MODEL_NAME, champion.version, "agent_type", "ResponsesAgent")
    client.set_model_version_tag(UC_MODEL_NAME, champion.version, "obo_enabled", "true")

    print(f"Updated version {champion.version} with description and tags")
except Exception as e:
    print(f"Could not update champion: {e}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Update Registered Model Description

# COMMAND ----------

client.update_registered_model(
    name=UC_MODEL_NAME,
    description=(
        "Corporate Affairs RAG Chatbot built with ResponsesAgent (MLflow 3.x). "
        "Uses Databricks Vector Search for document retrieval and MLflow Prompt Registry "
        "for versioned prompt management. Supports On-Behalf-Of (OBO) authentication "
        "for per-user data isolation. Deployed via databricks.agents.deploy()."
    ),
)

print(f"Updated model description for {UC_MODEL_NAME}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Manual Operations (uncomment as needed)
# MAGIC
# MAGIC ### Rollback to a Previous Version
# MAGIC ```python
# MAGIC # Point champion alias to a previous version
# MAGIC client.set_registered_model_alias(
# MAGIC     name=UC_MODEL_NAME,
# MAGIC     alias="champion",
# MAGIC     version="1",  # Previous known-good version
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ### Archive a Version
# MAGIC ```python
# MAGIC # Remove alias (version still exists but isn't referenced)
# MAGIC client.delete_registered_model_alias(
# MAGIC     name=UC_MODEL_NAME,
# MAGIC     alias="candidate",
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ### Delete a Model Version
# MAGIC ```python
# MAGIC client.delete_model_version(name=UC_MODEL_NAME, version="1")
# MAGIC ```
