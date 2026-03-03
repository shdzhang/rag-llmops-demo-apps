# Troubleshooting Guide: RAG LLMOps Demo

Issues encountered during development and deployment, organized by category. Each entry includes the error, root cause, and fix.

---

## Table of Contents

1. [Configuration & Naming](#1-configuration--naming)
2. [Vector Search & Retrieval](#2-vector-search--retrieval)
3. [Agent Evaluation](#3-agent-evaluation)
4. [Model Serving & Deployment](#4-model-serving--deployment)
5. [MLflow 3.x Migration](#5-mlflow-3x-migration)
6. [Databricks Asset Bundles](#6-databricks-asset-bundles)
7. [Quick Reference](#7-quick-reference)

---

## 1. Configuration & Naming

### 1.1 Hardcoded Config Causes Entity-Not-Found Errors

**Error:**
```
Unity Catalog entity main.corporate_affairs.corporate_docs_index does not exist.
```

**Cause:** Catalog, schema, and resource names were hardcoded in both `rag_agent.py` and notebooks (01-09). When DAB `mode: development` prepends `dev_<username>_` to schemas, the hardcoded names no longer match.

**Fix:**
- **Agent:** Reads config dynamically via `mlflow.models.ModelConfig()`, passed from notebook 04's `log_model(model_config=...)`.
- **Notebooks:** All read from DAB job parameters via `dbutils.widgets.get("catalog_name")`, etc.

---

### 1.2 Resource Name Length Limits (63/64 chars)

**Errors:**
```
Tool name shidong_catalog__dev_shidong_zhang_corporate_affairs__corporate_docs_index is too long, truncating to 64 characters
```
Endpoint wait loops ran forever because the constructed name didn't match the actual (truncated) endpoint name.

**Cause:** Two separate limits collide with DAB `mode: development` prefixing:
- Vector Search tool names: **64-character** limit
- Model Serving endpoint names: **63-character** limit (truncated by `agents.deploy()`)

**Fix:**
1. Shortened all base resource names (`corporate_affairs` → `corp_affairs`, `corporate_docs_index` → `docs_index`, etc.)
2. Added `[:63]` truncation when constructing endpoint names:
```python
endpoint_name = f"agents_{UC_MODEL_NAME}".replace(".", "-")[:63]
```

---

### 1.3 UC Model Aliases and Tags Are Dicts, Not Lists

**Error:**
```
'str' object has no attribute 'alias'
```

**Cause:** Unity Catalog returns `model.aliases` as `{"alias_name": "version"}` and `v.tags` as `{"key": "value"}` -- both dicts, not lists of objects.

**Fix:**
```python
aliases = model.aliases or {}
for alias_name, version in aliases.items():
    print(f"  @{alias_name} -> version {version}")
```

---

## 2. Vector Search & Retrieval

### 2.1 Index "ONLINE" But Returns 0 Rows

**Symptom:** Index status was `ONLINE`, pipeline showed `COMPLETED`, but `similarity_search()` returned 0 results.

**Root Cause:** Change Data Feed (CDF) was enabled **after** the initial data write. The Delta Sync pipeline had no changes to pick up.

**Fix:**
- **Notebook 01:** Enable CDF *before* writing data via `CREATE TABLE IF NOT EXISTS ... TBLPROPERTIES (delta.enableChangeDataFeed = true)`. Table properties persist across data overwrites, so a single `CREATE TABLE` is sufficient.
- **Notebook 02:** Wait loop now checks for *actual data* (via `similarity_search`), not just `ONLINE` status. Auto-triggers a sync if index is empty.

---

### 2.2 VectorSearchClient OBO Authentication

**Error:**
```
VectorSearchClient.__init__() got an unexpected keyword argument 'workspace_client'
```

**Cause:** `VectorSearchClient` does not accept a `workspace_client` parameter. The correct OBO mechanism is `CredentialStrategy`.

**Fix:**
```python
from databricks.vector_search.client import VectorSearchClient, CredentialStrategy

try:
    # OBO: authenticate as calling user in Model Serving
    vsc = VectorSearchClient(
        credential_strategy=CredentialStrategy.MODEL_SERVING_USER_CREDENTIALS,
        disable_notice=True,
    )
except Exception:
    # Fallback for local dev / notebook
    vsc = VectorSearchClient(disable_notice=True)
```

---

## 3. Agent Evaluation

### 3.1 All Notebooks Must Share One MLflow Experiment

**Symptom:** MLflow UI showed stray experiments (e.g., `07_endpoint_deployment`) instead of one unified experiment.

**Cause:** Notebooks without `mlflow.set_experiment()` get an auto-created experiment named after the notebook path.

**Fix:** Added `experiment_name` as a DAB job parameter to all job YAMLs. Every notebook (04-09) now calls:
```python
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")
mlflow.set_experiment(EXPERIMENT_NAME)
```

> **Note:** Job YAMLs must reference `${resources.experiments.experiment.name}` (not `${var.experiment_name}`) so the DAB prefix is included. See also [6.3 Monitoring Job Creates a Stray Experiment](#63-monitoring-job-creates-a-stray-mlflow-experiment).

---

### 3.2 Quality Gate Fails: Correctness 0.000

**Symptom:** All evaluation questions scored 0 for correctness. Agent responses said "I'm unable to access..."

**Causes (cumulative):**
1. Vector Search index had no data (CDF ordering issue -- see 2.1)
2. `predict_fn` silently passed error strings as context instead of failing
3. `expected_response` values were overly detailed, penalizing partial matches

**Fixes:**
1. Fixed CDF ordering (see 2.1) + added pre-evaluation readiness check with auto-sync
2. `predict_fn` now raises `RuntimeError` on 0 retrieval results
3. Simplified `expected_response` to essential facts only

---

### 3.3 Evaluation Takes 10+ Hours

**Symptom:** `mlflow.genai.evaluate()` ran for 10+ hours before being cancelled.

**Cause:** 3 scorers x 12 test cases = 36 judge calls (+ 12 predict calls) hitting a rate-limited Claude Sonnet 4.5 endpoint, each sending large prompts.

**Fix:**
1. **1 scorer** (`Correctness` only) -- additional quality checks run in production via External Monitor (NB09)
2. **Claude Opus 4.1** as judge (`databricks:/databricks-claude-opus-4-1`)
3. **8 test cases** (down from 12)
4. Concurrency env vars: `MLFLOW_GENAI_EVAL_MAX_WORKERS=4`, `MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS=2`

Result: **~10-15 minutes** instead of 10+ hours.

---

### 3.4 No Champion Model for Deployment

**Error:**
```
Registered Model Alias 'champion' does not exist.
```

**Cause:** Quality gate in notebook 05 failed but the `raise Exception` was commented out, so the pipeline continued to deployment without a promoted model.

**Fix:** Uncommented the `raise Exception` so the pipeline stops on quality failure. Notebook 07 also has a fallback to `@candidate` alias.

---

## 4. Model Serving & Deployment

### 4.1 `agents.deploy()` ValueError Handling

**Errors:**
```
ValueError: Endpoint ... already serves model ...
ValueError: Endpoint ... is currently updating.
```

**Cause:** `agents.deploy()` raises `ValueError` when (a) the same version is already deployed, or (b) the endpoint is mid-update.

**Fix:**
- **Pre-deployment:** Wait loop checks `config_update` status, waits up to 20 min for any prior update to finish
- **Try-except:** Catches `"already serves"` and `"currently updating"` as non-fatal

---

### 4.2 Endpoint Readiness Check Loops Forever

**Symptom:** Endpoint was READY, but the wait loop never broke out (ran 960+ seconds).

**Cause:** `str(state.config_update)` returns `"EndpointStateConfigUpdate.NOT_UPDATING"`, but the code used exact string matching (`== "NOT_UPDATING"`).

**Fix:** Use substring checks:
```python
is_ready = "READY" in str(state.ready)
is_updating = "IN_PROGRESS" in str(state.config_update)
```

---

### 4.3 ResponsesAgent Requires Responses API Format

**Error:**
```
Model is missing inputs ['input']. Note that there were extra inputs: ['messages', 'max_tokens'].
```

**Cause:** `ResponsesAgent` uses the Responses API (`input`/`output`), not Chat Completions (`messages`/`choices`).

**Fix:** All endpoint queries use:
```python
# Request
body = {"input": [{"role": "user", "content": question}]}

# Response parsing
for item in result.get("output", []):
    if item.get("type") == "message":
        for block in item.get("content", []):
            if block.get("type") == "output_text":
                answer += block.get("text", "")
```

Also use `w.api_client.do()` instead of raw `requests.post()` to handle all auth methods (PAT, AAD, OAuth) transparently.

---

### 4.4 Inference Table JSON Path Mismatch

**Symptom:** Monitoring "Popular Questions" query returned `NULL` for all questions.

**Cause:** SQL used `$.messages[0].content` (Chat Completions) but Responses API stores input as `$.input[0].content`.

**Fix:**
```sql
COALESCE(
    get_json_object(request, '$.input[0].content'),
    get_json_object(request, '$.messages[0].content')
) AS question
```

---

## 5. MLflow 3.x Migration

### 5.1 Prompt Registry Requires 3-Level UC Names

**Error:**
```
RestException: INVALID_PARAMETER_VALUE: name is not a valid name.
```

**Fix:** Prompt names must be `{catalog}.{schema}.{prompt_name}` with underscores only (no hyphens).

---

### 5.2 `EvaluationResult.eval_table` Removed

**Error:**
```
'EvaluationResult' object has no attribute 'eval_table'
```

**Fix:** Use `mlflow.search_traces(run_id=eval_results.run_id)` instead.

---

### 5.3 LLM Judge Model URI Format

**Error:**
```
Malformed model uri 'databricks-claude-opus-4-1'
```

**Fix:** Scorers require `provider:/model-name` format: `"databricks:/databricks-claude-opus-4-1"`.

---

### 5.4 Link Prompts to MLflow Runs

**Symptom:** "Prompts" section in experiment UI was empty.

**Fix:** Add `prompts=[prompt_uri]` to `mlflow.pyfunc.log_model()`.

---

### 5.5 `get_open_ai_client()` Deprecated

**Fix:** Replace with:
```python
from databricks_openai import DatabricksOpenAI
client = DatabricksOpenAI()
```
Add `databricks-openai` to `%pip install` commands.

---

### 5.6 `mlflow.openai.autolog()` Fails During `log_model()` Validation

**Error:**
```
MlflowException: Failed to run user code from .../rag_agent.py.
Error: 'NoneType' object has no attribute '_multi_processor'
```

**Cause:** `mlflow.pyfunc.log_model()` loads the agent module in a validation subprocess where the OpenTelemetry `GLOBAL_TRACE_PROVIDER` is not initialized. `mlflow.openai.autolog()` tries to access `GLOBAL_TRACE_PROVIDER._multi_processor` and crashes. This started appearing after MLflow/openai SDK version updates — earlier versions tolerated missing providers.

**Fix:** Wrap `autolog()` in a try/except in the agent module:
```python
try:
    mlflow.openai.autolog()
except Exception:
    pass
```
The agent works correctly without autolog. Explicit `@mlflow.trace` decorators on agent methods still produce traces. Also use lazy initialization for the `DatabricksOpenAI` client (property getter instead of `__init__`).

---

## 6. Databricks Asset Bundles

### 6.1 Double Prefix in Job Names

**Symptom:** Job names like `[dev shidong_zhang] [dev] Data Preparation`.

**Cause:** `mode: development` already adds `[dev <username>]`. Job YAML also had `[${bundle.target}]`.

**Fix:** Remove `[${bundle.target}]` from all job YAML `name` fields.

---

### 6.2 Notebook Not Deployed (Silent Skip)

**Error:**
```
Unable to access the notebook ".../05_agent_evaluation" in the workspace.
```

**Cause:** Known DABs sync issue -- files silently skipped during upload.

**Fix:** Redeploy. If persistent, manually upload the notebook.

---

### 6.3 Monitoring Job Creates a Stray MLflow Experiment

**Symptom:** Two experiments in the MLflow UI: `[dev shidong_zhang] dev_corp_chatbot` (correct) and `dev_corp_chatbot` (stray, created by the monitoring job).

**Cause:** `monitoring.job.yml` used `${var.experiment_name}` for its `experiment_name` parameter. In `mode: development`, DAB prefixes **resource** names (e.g., `${resources.experiments.experiment.name}` → `[dev shidong_zhang] dev_corp_chatbot`) but does **not** prefix raw **variable** values (e.g., `${var.experiment_name}` → `dev_corp_chatbot`). When NB09 called `mlflow.set_experiment("dev_corp_chatbot")`, MLflow created a new experiment without the prefix.

**Fix:** Changed the default in `monitoring.job.yml` from `${var.experiment_name}` to `${resources.experiments.experiment.name}`, matching the other job YAMLs:
```yaml
# monitoring.job.yml
parameters:
  - name: "experiment_name"
    default: "${resources.experiments.experiment.name}"  # NOT ${var.experiment_name}
```

**Cleanup:** Delete the stray `dev_corp_chatbot` experiment from the MLflow UI.

---

### 6.4 Prompt Registration Fails: `SCHEMA_DOES_NOT_EXIST`

**Error:**
```
RestException: SCHEMA_DOES_NOT_EXIST: Schema 'shidong_catalog.corp_affairs' does not exist.
```

**Cause:** Same class of bug as 6.3. The `prompt_name` DAB variable was computed in `databricks.yml` using `${var.schema_name}` (raw value: `corp_affairs`). In `mode: development`, the actual schema is prefixed by DAB (e.g., `dev_shidong_zhang_corp_affairs`), but DAB variables can only reference other variables — not resources. So the prompt name pointed to a non-existent schema.

**Key insight:** DAB variables (`${var.X}`) can only interpolate other variables. They **cannot** reference resources (`${resources.schemas.X.name}`). The dev-prefixed schema name is only available via resource references in job parameter defaults.

**Fix:** Removed the computed `prompt_name` variable from `databricks.yml`. Instead:
1. `databricks.yml` defines `prompt_base_name` (just `"rag_prompt"`)
2. Job YAML passes it as `prompt_base_name` parameter
3. Notebooks construct the full name using their (correctly-prefixed) widget values:
```python
PROMPT_NAME = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('prompt_base_name')}"
```

**Rule of thumb:** Never construct 3-level UC names (`catalog.schema.object`) in `databricks.yml` variables. Always construct them in notebooks using `CATALOG` and `SCHEMA` from job parameters, which carry the correct DAB-prefixed values.

---

## 7. Quick Reference

### Querying a ResponsesAgent Endpoint
```python
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
result = w.api_client.do(
    "POST",
    f"/serving-endpoints/{endpoint_name}/invocations",
    body={"input": [{"role": "user", "content": "your question"}]},
)
for item in result.get("output", []):
    if item.get("type") == "message":
        for block in item.get("content", []):
            if block.get("type") == "output_text":
                print(block["text"])
```

### Endpoint Readiness Check (Enum-Safe)
```python
is_ready = "READY" in str(state.ready)
is_updating = "IN_PROGRESS" in str(state.config_update)
```

### Endpoint Name (63-char Limit)
```python
endpoint_name = f"agents_{UC_MODEL_NAME}".replace(".", "-")[:63]
```

### Vector Search OBO
```python
from databricks.vector_search.client import VectorSearchClient, CredentialStrategy
vsc = VectorSearchClient(
    credential_strategy=CredentialStrategy.MODEL_SERVING_USER_CREDENTIALS,
    disable_notice=True,
)
```
