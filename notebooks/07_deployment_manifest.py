# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - Deployment Manifest
# MAGIC
# MAGIC Logs a deployment record to the MLflow experiment as a run with metadata tags.
# MAGIC Answers: "What version is deployed in this environment right now?"
# MAGIC
# MAGIC Each deployment creates an MLflow run tagged with:
# MAGIC - Git commit SHA and branch
# MAGIC - Prompt name, version, and alias
# MAGIC - Evaluation run ID (from the quality gate)
# MAGIC - Target environment and app URL
# MAGIC - Timestamp

# COMMAND ----------
# MAGIC %pip install mlflow>=3.1 databricks-sdk
# MAGIC %restart_python

# COMMAND ----------
import os
import mlflow
from datetime import datetime
from databricks.sdk import WorkspaceClient

EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")
APP_NAME = dbutils.widgets.get("app_name")
CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
TARGET = dbutils.widgets.get("target")

PROMPT_NAME = f"{CATALOG}.{SCHEMA}.rag_prompt"
PROMPT_ALIAS = "production"

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_NAME)

w = WorkspaceClient()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Gather Deployment Metadata

# COMMAND ----------

git_sha = "unknown"
git_branch = "unknown"
try:
    import subprocess
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
except Exception:
    for env_var in ["GIT_COMMIT", "GITHUB_SHA", "CI_COMMIT_SHA"]:
        if os.environ.get(env_var):
            git_sha = os.environ[env_var]
            break
    for env_var in ["GIT_BRANCH", "GITHUB_REF_NAME", "CI_COMMIT_BRANCH"]:
        if os.environ.get(env_var):
            git_branch = os.environ[env_var]
            break

prompt_version = "unknown"
try:
    prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@{PROMPT_ALIAS}")
    prompt_version = str(getattr(prompt, "version", "unknown"))
except Exception as e:
    print(f"Could not load prompt version: {e}")

eval_run_id = "none"
try:
    eval_run_id = dbutils.jobs.taskValues.get(
        taskKey="agent_evaluation", key="eval_run_id", debugValue="none"
    )
except Exception:
    pass

app_url = "unknown"
try:
    app = w.apps.get(APP_NAME)
    app_url = getattr(app, "url", "unknown") or "unknown"
except Exception:
    pass

print(f"Git SHA:          {git_sha}")
print(f"Git branch:       {git_branch}")
print(f"Prompt:           {PROMPT_NAME}@{PROMPT_ALIAS} (v{prompt_version})")
print(f"Eval run ID:      {eval_run_id}")
print(f"Target:           {TARGET}")
print(f"App URL:          {app_url}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Log Deployment Run

# COMMAND ----------

with mlflow.start_run(run_name=f"deploy-{TARGET}-{git_sha[:8]}") as run:
    mlflow.set_tags({
        "deployment.target": TARGET,
        "deployment.git_sha": git_sha,
        "deployment.git_branch": git_branch,
        "deployment.prompt_name": PROMPT_NAME,
        "deployment.prompt_version": prompt_version,
        "deployment.prompt_alias": PROMPT_ALIAS,
        "deployment.eval_run_id": eval_run_id,
        "deployment.app_name": APP_NAME,
        "deployment.app_url": app_url,
        "deployment.timestamp": datetime.now().isoformat(),
        "mlflow.runName": f"deploy-{TARGET}-{git_sha[:8]}",
    })
    mlflow.log_params({
        "target": TARGET,
        "git_sha": git_sha,
        "prompt_version": prompt_version,
    })

    print(f"\nDeployment manifest logged as MLflow run: {run.info.run_id}")
    print(f"View in experiment: {EXPERIMENT_NAME}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Deployment History
# MAGIC
# MAGIC Query recent deployments for this environment.

# COMMAND ----------

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment:
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.`deployment.target` = '{TARGET}'",
        order_by=["start_time DESC"],
        max_results=10,
    )
    if not runs.empty:
        cols = [c for c in [
            "run_id", "start_time",
            "tags.deployment.git_sha", "tags.deployment.prompt_version",
            "tags.deployment.eval_run_id",
        ] if c in runs.columns]
        print(f"\nRecent deployments to {TARGET}:")
        print(runs[cols].to_string(index=False))
    else:
        print(f"No previous deployments found for target: {TARGET}")

# COMMAND ----------

if "dbutils" in dir():
    dbutils.jobs.taskValues.set(key="deployment_run_id", value=run.info.run_id)
    dbutils.jobs.taskValues.set(key="deployed_git_sha", value=git_sha)
print("Deployment manifest complete.")
