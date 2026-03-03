# Troubleshooting Guide: RAG LLMOps Demo (Databricks Apps)

Issues encountered during development and deployment. Each entry includes the error, root cause, and fix.

---

## Table of Contents

1. [Configuration & Naming](#1-configuration--naming)
2. [Vector Search & Retrieval](#2-vector-search--retrieval)
3. [Agent Evaluation](#3-agent-evaluation)
4. [MLflow 3.x](#4-mlflow-3x)
5. [Databricks Asset Bundles](#5-databricks-asset-bundles)
6. [Databricks Apps Deployment](#6-databricks-apps-deployment)
7. [Databricks Apps Runtime](#7-databricks-apps-runtime)
8. [Notebook Smoke Tests (NB04)](#8-notebook-smoke-tests-nb05)
9. [Quick Reference](#9-quick-reference)
10. [Historical: Model Serving Issues](#10-historical-model-serving-issues)

---

## 1. Configuration & Naming

### 1.1 Hardcoded Config Causes Entity-Not-Found Errors

**Error:**
```
Unity Catalog entity main.corporate_affairs.corporate_docs_index does not exist.
```

**Cause:** Catalog, schema, and resource names were hardcoded. When DAB `mode: development` prepends `dev_<username>_` to schemas, hardcoded names don't match.

**Fix:**
- **Agent:** Reads config from environment variables (`LLM_ENDPOINT`, `VECTOR_SEARCH_INDEX`, etc.), set in `app.yaml` and `.env`.
- **Notebooks:** Read from DAB job parameters via `dbutils.widgets.get("catalog_name")`, etc.

---

### 1.2 Resource Name Length Limits (63/64 chars)

**Error:**
```
Tool name shidong_catalog__dev_shidong_zhang_corporate_affairs__corporate_docs_index is too long
```

**Cause:** Vector Search tool names have a 64-character limit. Long catalog/schema/index names combined with DAB prefixing exceed this.

**Fix:** Use short base names (`corp_affairs` instead of `corporate_affairs`, `docs_index` instead of `corporate_docs_index`).

---

## 2. Vector Search & Retrieval

### 2.1 Index "ONLINE" But Returns 0 Rows

**Symptom:** Index status was `ONLINE`, pipeline showed `COMPLETED`, but `similarity_search()` returned 0 results.

**Root Cause:** Change Data Feed (CDF) was enabled **after** the initial data write. The Delta Sync pipeline had no changes to pick up.

**Fix:**
- **Notebook 01:** Enable CDF *before* writing data via `CREATE TABLE IF NOT EXISTS ... TBLPROPERTIES (delta.enableChangeDataFeed = true)`.
- **Notebook 02 / NB04:** Wait loop checks for *actual data* (via `similarity_search`), not just `ONLINE` status. Auto-triggers sync if index is empty.

---

## 3. Agent Evaluation

### 3.1 All Notebooks Must Share One MLflow Experiment

**Symptom:** MLflow UI showed stray experiments created by individual notebooks.

**Cause:** Notebooks without `mlflow.set_experiment()` get auto-created experiments.

**Fix:** Added `experiment_name` as a DAB job parameter. Every notebook calls:
```python
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")
mlflow.set_experiment(EXPERIMENT_NAME)
```

> Job YAMLs must reference `${resources.experiments.experiment.name}` (not `${var.experiment_name}`) so the DAB prefix is included.

---

### 3.2 Quality Gate Fails: Correctness 0.000

**Symptom:** All evaluation questions scored 0 for correctness. Agent responses said "I'm unable to access..."

**Causes (cumulative):**
1. Vector Search index had no data (CDF ordering issue — see 2.1)
2. Agent couldn't authenticate to Vector Search (see 7.2)
3. `expected_response` values were overly detailed, penalizing partial matches

**Fixes:**
1. Fixed CDF ordering + added pre-evaluation readiness check with auto-sync
2. Used `WorkspaceClient` for VS retrieval (handles Apps OAuth M2M natively)
3. Simplified `expected_response` to essential facts only

---

### 3.3 Evaluation Takes 10+ Hours

**Symptom:** `mlflow.genai.evaluate()` ran for 10+ hours before being cancelled.

**Cause:** 3 scorers x 12 test cases = 36 judge calls hitting a rate-limited endpoint, each sending large prompts.

**Fix:**
1. **1 scorer** (`Correctness` only) — additional quality checks run in production via External Monitor (NB06)
2. **Claude Opus 4.1** as judge (`databricks:/databricks-claude-opus-4-1`)
3. **8 test cases** (down from 12)
4. Concurrency env vars: `MLFLOW_GENAI_EVAL_MAX_WORKERS=4`, `MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS=2`

Result: ~10-15 minutes instead of 10+ hours.

---

### 3.4 `AttributeError: 'str' object has no attribute 'name'` in NB04

**Error:**
```
AttributeError: 'str' object has no attribute 'name'
```

**Cause:** NB04 used `[w.name for w in dbutils.widgets.getAll()]` to list available widgets. In some Databricks Runtime versions, `dbutils.widgets.getAll()` returns plain strings, not objects with a `.name` attribute.

**Fix:** Simplified environment variable setting with a direct `try/except`:
```python
try:
    os.environ["LLM_ENDPOINT"] = dbutils.widgets.get("llm_endpoint")
except Exception:
    os.environ.setdefault("LLM_ENDPOINT", "databricks-claude-sonnet-4-5")
```

---

### 3.5 `PySpark ValueError` When Displaying `mlflow.search_traces()` Results

**Error:**
```
PySpark ValueError: Exception thrown when converting pandas.Series (object)
with name 'response' to Arrow Array
```

**Cause:** `mlflow.search_traces()` returns a pandas DataFrame with complex Python objects (nested dicts/lists) in columns like `response`. Databricks `display()` converts pandas DataFrames via PySpark using Apache Arrow, which cannot serialize these complex objects.

**Fix:** Print only safe scalar columns instead of using `display()`:
```python
results_df = mlflow.search_traces(run_id=eval_results.run_id)
safe_cols = [c for c in ["trace_id", "status", "execution_time_ms"] if c in results_df.columns]
print(results_df[safe_cols].to_string() if safe_cols else f"{len(results_df)} traces found")
```

**Applies to:** Any notebook that calls `display()` on `mlflow.search_traces()` output (NB04, NB06).

---

### 3.6 `mlflow.search_traces()` Column Names Don't Match Documentation

**Symptom:** NB06 analytics sections (volume, latency, errors) all print "No column found" — the code runs without errors but produces no useful output.

**Cause:** `mlflow.search_traces()` returns columns with different names than commonly documented:

| Expected | Actual |
|---|---|
| `status` | `state` |
| `execution_time_ms` | `execution_duration` |
| `timestamp_ms` | `request_time` |

The full column list: `trace_id`, `trace`, `client_request_id`, `state`, `request_time`, `execution_duration`, `request`, `response`, `trace_metadata`, `tags`, `spans`, `assessments`.

**Fix:** Use the actual column names with fallbacks:
```python
_state_col = "state" if "state" in df.columns else "status"
_latency_col = "execution_duration" if "execution_duration" in df.columns else "execution_time_ms"
_time_col = "request_time" if "request_time" in df.columns else "timestamp_ms"
```

**Note:** `execution_duration` may be a `timedelta64` type requiring conversion: `pd.to_timedelta(series).dt.total_seconds() * 1000` for milliseconds.

---

## 4. MLflow 3.x

### 4.1 Prompt Registry Requires 3-Level UC Names

**Error:**
```
RestException: INVALID_PARAMETER_VALUE: name is not a valid name.
```

**Fix:** Prompt names must be `{catalog}.{schema}.{prompt_name}` with underscores only (no hyphens).

---

### 4.2 `EvaluationResult.eval_table` Removed

**Error:**
```
'EvaluationResult' object has no attribute 'eval_table'
```

**Fix:** Use `mlflow.search_traces(run_id=eval_results.run_id)` instead.

---

### 4.3 LLM Judge Model URI Format

**Error:**
```
Malformed model uri 'databricks-claude-opus-4-1'
```

**Fix:** Scorers require `provider:/model-name` format: `"databricks:/databricks-claude-opus-4-1"`.

---

### 4.4 `get_open_ai_client()` Deprecated

**Fix:** Replace with:
```python
from databricks_openai import DatabricksOpenAI
client = DatabricksOpenAI()
```

---

### 4.5 `mlflow.openai.autolog()` Crashes on Module Import

**Error:**
```
AttributeError: 'NoneType' object has no attribute '_multi_processor'
```

**Cause:** When the agent module is imported outside the AgentServer (e.g., during local testing or NB04 evaluation), the OpenTelemetry `GLOBAL_TRACE_PROVIDER` is `None`. `mlflow.openai.autolog()` tries to access `GLOBAL_TRACE_PROVIDER._multi_processor`.

**Fix:** Wrap in try/except:
```python
try:
    mlflow.openai.autolog()
except Exception:
    pass
```
Autolog works correctly when running inside the AgentServer.

---

## 5. Databricks Asset Bundles

### 5.1 Double Prefix in Job Names

**Symptom:** Job names like `[dev shidong_zhang] [dev] Data Preparation`.

**Cause:** `mode: development` already adds `[dev <username>]`. Job YAML also had `[${bundle.target}]`.

**Fix:** Remove `[${bundle.target}]` from all job YAML `name` fields.

---

### 5.2 Notebook Not Deployed (Silent Skip)

**Error:**
```
Unable to access the notebook ".../04_agent_evaluation" in the workspace.
```

**Cause:** Known DABs sync issue — files silently skipped during upload.

**Fix:** Redeploy. If persistent, manually upload the notebook.

---

### 5.3 Monitoring Job Creates a Stray MLflow Experiment

**Symptom:** Two experiments in the MLflow UI: one with the `[dev ...]` prefix (correct) and one without (stray).

**Cause:** `monitoring.job.yml` used `${var.experiment_name}` for its parameter default. In `mode: development`, DAB prefixes **resource** names but does **not** prefix raw **variable** values. When NB06 called `mlflow.set_experiment("dev_corp_chatbot")`, MLflow created a new experiment without the prefix.

**Fix:** Changed the default from `${var.experiment_name}` to `${resources.experiments.experiment.name}`:
```yaml
parameters:
  - name: "experiment_name"
    default: "${resources.experiments.experiment.name}"  # NOT ${var.experiment_name}
```

---

### 5.4 Prompt Registration Fails: `SCHEMA_DOES_NOT_EXIST`

**Error:**
```
RestException: SCHEMA_DOES_NOT_EXIST: Schema 'shidong_catalog.corp_affairs' does not exist.
```

**Cause:** The prompt name was constructed using `${var.schema_name}` (raw value). In `mode: development`, the actual schema is `dev_<username>_corp_affairs`, but DAB variables can't reference resources.

**Fix:** Pass `prompt_base_name` (just `"rag_prompt"`) as a job parameter. Notebooks construct the full name:
```python
PROMPT_NAME = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('prompt_base_name')}"
```

**Rule:** Never construct 3-level UC names in `databricks.yml` variables. Always construct them in notebooks using `CATALOG` and `SCHEMA` from job parameters.

---

## 6. Databricks Apps Deployment

### 6.1 Production Target Requires `root_path`

**Error:**
```
Error: target with 'mode: production' must set 'workspace.root_path'
```

**Fix:** Add `root_path` to the prod target in `databricks.yml`:
```yaml
targets:
  prod:
    mode: production
    workspace:
      host: https://...
      root_path: /Workspace/Users/<user>/.bundle/${bundle.name}/${bundle.target}
```

---

### 6.2 App Deploy Fails: Vector Search Index Does Not Exist

**Error:**
```
Failed to retrieve UC table info for resource 'vector-search-index'
(shidong_catalog.corp_affairs.docs_index): TABLE_DOES_NOT_EXIST
```

**Cause:** The `databricks.yml` declares a `uc_securable` for the VS index, granting the app's service principal SELECT access. But the index doesn't exist yet because the data preparation job hasn't run.

**Fix:** Deploy in stages (see [deployment_guide.md](deployment_guide.md)):
1. Comment out the `vector-search-index` resource
2. Deploy the bundle
3. Run the data preparation job
4. Uncomment the resource and redeploy

---

### 6.3 Duplicate MLflow Experiments (quickstart vs bundle)

**Symptom:** Two experiments: `agents-on-apps` (from `uv run quickstart`) and `prod_corp_chatbot` (from `databricks bundle deploy`).

**Fix:** Update `.env` to use the bundle-managed experiment ID. Delete the unused experiment.

**Prevention:** Run `uv run quickstart` *after* `databricks bundle deploy`, or manually set `MLFLOW_EXPERIMENT_ID` in `.env`.

---

### 6.4 `requirements.txt` Must Contain `uv` (Not Dependencies)

**Cause:** Databricks Apps uses `pip install -r requirements.txt` to bootstrap the runtime. If it lists actual dependencies, they conflict with `uv`'s resolution from `pyproject.toml`.

**Fix:** `requirements.txt` must contain exactly one line: `uv`. All actual dependencies go in `pyproject.toml`.

---

### 6.5 `app.yaml` Must Be at Project Root

**Cause:** `app.yaml` is the Apps runtime configuration file, read from the root of `source_code_path`. It is not a DAB resource definition.

**Rule:** `app.yaml` = "how to run this app" (project root). `resources/*.yml` = "what infrastructure to create" (DAB includes).

---

### 6.6 `hatchling` Build Fails: "Unable to determine which files to ship"

**Error:**
```
ValueError: Unable to determine which files to ship inside the wheel using the following heuristics
```

**Cause:** Project name (`rag-llmops-demo-apps`) doesn't match any directory. Hatchling expects a matching package directory.

**Fix:** Add explicit package targets to `pyproject.toml`:
```toml
[tool.hatch.build.targets.wheel]
packages = ["agent_server", "scripts"]
```

---

### 6.7 OAuth Integration Quota Exceeded

**Error:**
```
QUOTA_EXCEEDED: Only 1000 OAuth custom application integrations can be created per account
```

**Cause:** Every Databricks App creates an OAuth integration. Shared/demo workspaces accumulate abandoned apps.

**Fix:**
1. Delete unused apps: `databricks apps delete <old-app-name>`
2. If broken app was partially created, delete it first, then redeploy
3. Long-term: workspace admin cleans up stale integrations

---

## 7. Databricks Apps Runtime

### 7.1 `ResponsesAgentResponseOutputItemMessage` ImportError

**Error:**
```
ImportError: cannot import name 'ResponsesAgentResponseOutputItemMessage' from 'mlflow.types.responses'
```

**Cause:** This class doesn't exist in the installed MLflow version.

**Fix:** Use a plain dict for the `item` field:
```python
yield ResponsesAgentStreamEvent(
    type="response.output_item.done",
    item={
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": full_text}],
    },
)
```

---

### 7.2 VectorSearchClient Auth Fails in Apps (OAuth M2M)

**Error:**
```
TypeError: Object of type ResponseInputTextParam is not JSON serializable
```
(Secondary symptom. Primary failure was `VectorSearchClient()` couldn't authenticate.)

**Cause:** Apps inject OAuth M2M credentials (`DATABRICKS_CLIENT_ID/SECRET/HOST`). `VectorSearchClient()` looks for a PAT or notebook token — neither exists in Apps.

**Fix:** Use `WorkspaceClient` instead (natively supports OAuth M2M):
```python
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
results = w.vector_search_indexes.query_index(
    index_name=INDEX_NAME,
    query_text=question,
    columns=[...],
    num_results=5,
)
rows = results.result.data_array if results.result else []
```

**Key insight:** In Apps, always prefer `WorkspaceClient()` over service-specific clients because it automatically handles OAuth M2M.

---

### 7.3 User Message Content Is a List, Not a String

**Error:**
```
TypeError: Object of type ResponseInputTextParam is not JSON serializable
```

**Cause:** In the Responses API, `msg.content` can be a `str` or `list[ResponseInputTextParam]`. The chat UI sends structured content. The agent assumed it was always a string.

**Fix:** Handle both:
```python
for msg in request.input:
    if msg.role == "user":
        if isinstance(msg.content, str):
            user_message = msg.content
        elif isinstance(msg.content, list):
            user_message = " ".join(
                item.text if hasattr(item, "text") else str(item)
                for item in msg.content
            )
```

---

### 7.4 Prompt Creation Fails: `RestException: NOT_FOUND` in NB03

**Error:**
```
RestException: NOT_FOUND: Prompt with name rag_prompt does not exist.
```

**Cause:** NB03 (`03_prompt_engineering.py`) calls `mlflow.genai.load_prompt()` to check if the prompt exists. When it doesn't, the exception handler matched only `RESOURCE_DOES_NOT_EXIST` but the actual error string was `NOT_FOUND`, causing the exception to re-raise instead of proceeding to create the prompt.

**Fix:** Broaden the exception handler to catch multiple error phrases:
```python
except Exception as e:
    err_str = str(e)
    if any(k in err_str for k in ("RESOURCE_DOES_NOT_EXIST", "NOT_FOUND", "does not exist")):
        print(f"Prompt not found — creating initial versions ({type(e).__name__})")
    else:
        raise
```

---

### 7.5 Prompt Loading Fails: Service Principal Lacks Schema Permissions

**Error:**
```
PERMISSION_DENIED: Permission denied to create prompt in schema corp_affairs.
```

**Cause:** `mlflow.genai.load_prompt()` requires schema permissions. The app's service principal only had SELECT on the VS index.

**Fix:** Add schema grants in `databricks.yml`:
```yaml
resources:
  schemas:
    corporate_schema:
      grants:
        - principal: "${resources.apps.corp_chatbot_app.service_principal_client_id}"
          privileges:
            - USE_SCHEMA
            - SELECT
            - EXECUTE
```

---

## 8. Notebook Smoke Tests (NB05/NB06)

### 8.1 `NotFound` When Querying App via `w.api_client.do()`

**Error:**
```
NotFound: Not Found
```

**Cause:** NB05 originally called `w.api_client.do("POST", f"/apps/{APP_NAME}/invocations", body=payload)`. The Databricks SDK sends this to `https://<workspace>/apps/<name>/invocations`, but the workspace API does not have a REST endpoint at that path — the `/apps/<name>/` path is a web proxy, not a REST API route.

**Fix:** See 8.3 for the final solution.

---

### 8.2 `JSONDecodeError` — App Returns Login Page, Not JSON

**Error:**
```
JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**Cause:** After switching to direct HTTP calls (`requests.post(f"{APP_URL}/invocations", ...)`), the app responded with HTTP 200 but the body was the Databricks **Sign In HTML page** (25KB of HTML). The `Content-Type` was `text/html`, not `application/json`.

**Debug evidence:**
```
status=200 content_type=text/html; charset=utf-8 body_len=25658
url=https://adb-xxx.azuredatabricks.net/login.html?...redirect_uri=...corp-chatbot-app.../.auth/callback
```

**Root cause:** Databricks Apps with **user authorization** require browser-based OAuth. Neither the notebook's workspace API token (`dbutils.notebook.entry_point...apiToken().get()`) nor the workspace proxy (`<workspace-url>/apps/<name>/`) accepts Bearer tokens. Both redirect to the OAuth login flow.

**Fix:** See 8.3.

---

### 8.3 Apps Cannot Be Queried Programmatically from Notebooks (OAuth)

**Problem:** When user authorization is enabled, Databricks Apps require browser-based OAuth for all requests — including `/invocations`. There is no programmatic HTTP endpoint that accepts workspace API tokens.

Attempted approaches (all failed):
1. Direct app URL + Bearer token → redirected to login page
2. Workspace proxy (`/apps/<name>/`) + Bearer token → same redirect
3. Adding `"stream": False` to payload → no effect on auth
4. SDK `w.api_client.do()` → 404 (path is a web proxy, not REST API)

**Solution:** NB05 now imports and calls the agent functions directly — the same code path the deployed app runs:
```python
from agent_server.agent import _retrieve_context, _load_and_format_prompt, _get_openai_client, LLM_ENDPOINT_NAME

def query_agent(question, max_output_tokens=500):
    context = _retrieve_context(question)
    formatted_prompt = _load_and_format_prompt(context=context, question=question)
    messages = [{"role": "user", "content": formatted_prompt}]
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=LLM_ENDPOINT_NAME, messages=messages,
        temperature=0.1, max_tokens=max_output_tokens,
    )
    return {"answer": response.choices[0].message.content}
```

This tests the full RAG pipeline (vector search → prompt → LLM) while the app status is verified separately via `w.apps.get(APP_NAME)`.

**Key takeaway:** For Databricks Apps with user authorization, programmatic testing must call the agent code directly. HTTP-level testing requires browser-based OAuth or disabling user authorization.

---

## 9. Quick Reference

### Querying the App (Browser Only)

Apps with user authorization can only be queried via browser (OAuth).
For programmatic testing from notebooks, call the agent code directly:
```python
from agent_server.agent import _retrieve_context, _load_and_format_prompt, _get_openai_client, LLM_ENDPOINT_NAME

context = _retrieve_context("your question")
prompt = _load_and_format_prompt(context=context, question="your question")
client = _get_openai_client()
response = client.chat.completions.create(
    model=LLM_ENDPOINT_NAME,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.1,
)
print(response.choices[0].message.content)
```

### Vector Search in Apps (use SDK)
```python
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
results = w.vector_search_indexes.query_index(
    index_name="catalog.schema.index",
    query_text="your query",
    columns=["col1", "col2"],
    num_results=5,
)
rows = results.result.data_array if results.result else []
```

### Extracting User Message (Responses API)
```python
for msg in request.input:
    if msg.role == "user":
        if isinstance(msg.content, str):
            user_message = msg.content
        elif isinstance(msg.content, list):
            user_message = " ".join(
                item.text if hasattr(item, "text") else str(item)
                for item in msg.content
            )
```

### Stream Done Event (plain dict, not class)
```python
yield ResponsesAgentStreamEvent(
    type="response.output_item.done",
    item={"id": item_id, "type": "message", "role": "assistant",
          "content": [{"type": "output_text", "text": full_text}]},
)
```

### Grant App SP Access to Schema (DAB)
```yaml
resources:
  schemas:
    my_schema:
      grants:
        - principal: "${resources.apps.my_app.service_principal_client_id}"
          privileges: [USE_SCHEMA, SELECT, EXECUTE]
```

---

## 10. Historical: Model Serving Issues

The following issues were encountered during the original Model Serving deployment. They are preserved here for reference but **do not apply to the current Apps architecture**.

### UC Model Aliases and Tags Are Dicts, Not Lists

**Error:** `'str' object has no attribute 'alias'`

**Context:** Unity Catalog returns `model.aliases` as `{"alias_name": "version"}` — dicts, not lists. This was relevant when the pipeline managed `@candidate`/`@champion` aliases. In the Apps architecture, model aliases are no longer used for deployment.

---

### VectorSearchClient OBO Authentication

**Error:** `VectorSearchClient.__init__() got an unexpected keyword argument 'workspace_client'`

**Context:** The correct OBO mechanism for Model Serving was `CredentialStrategy.MODEL_SERVING_USER_CREDENTIALS`. In the Apps architecture, this is replaced by `WorkspaceClient()` which handles OAuth M2M natively (see 7.2).

---

### `agents.deploy()` ValueError Handling

**Errors:** `Endpoint ... already serves model ...` / `Endpoint ... is currently updating.`

**Context:** `agents.deploy()` was the Model Serving deployment method. In the Apps architecture, deployment uses `databricks bundle deploy` + `bundle run`. These errors no longer apply.

---

### Endpoint Readiness Check Loops Forever

**Context:** Model Serving endpoints required polling `config_update` status with substring checks (`"READY" in str(state.ready)`). Apps use `w.apps.get(APP_NAME)` and check `app_status.state` instead (see NB05).

---

### ResponsesAgent Requires Responses API Format

**Error:** `Model is missing inputs ['input']. Note that there were extra inputs: ['messages', 'max_tokens'].`

**Context:** This applies to both Model Serving and Apps. The Responses API uses `input`/`output`, not `messages`/`choices`. Still relevant — see 8 (Quick Reference) for the correct format.

---

### Inference Table JSON Path Mismatch

**Context:** Inference tables used Chat Completions JSON paths (`$.messages[0].content`). In the Apps architecture, monitoring uses `mlflow.search_traces()` instead of inference table SQL queries. This issue no longer applies.

---

### No Champion Model for Deployment

**Error:** `Registered Model Alias 'champion' does not exist.`

**Context:** In Model Serving, deployment was gated by the `@champion` alias. In Apps, deployment is gated by the quality gate in NB04 (which raises an exception on failure) and triggered by CLI/CI-CD, not model promotion.
