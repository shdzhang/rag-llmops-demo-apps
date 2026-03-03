# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Prompt Engineering with MLflow Prompt Registry
# MAGIC
# MAGIC This notebook manages versioned prompt templates via **MLflow Prompt Registry**.
# MAGIC It is the **prompt artifact management** stage of the LLMOps pipeline.
# MAGIC
# MAGIC ## Role in the Pipeline
# MAGIC
# MAGIC | Stage | Notebook | Purpose |
# MAGIC |-------|----------|---------|
# MAGIC | **Prompt Management (here)** | 03 | Register, version, configure, smoke-test, alias |
# MAGIC | Agent Evaluation | 05 | End-to-end evaluation (retrieval + prompt + LLM) with quality gates |
# MAGIC
# MAGIC Systematic evaluation with `mlflow.genai.evaluate()` happens in NB05, which tests
# MAGIC the full RAG agent holistically. This notebook focuses on the prompt artifact itself.
# MAGIC
# MAGIC ## Workflow
# MAGIC
# MAGIC **Automated (DAB job / Run All):** Runs end-to-end idempotently. Skips registration if the template hasn't changed.
# MAGIC
# MAGIC **Iterative prompt engineering (manual):**
# MAGIC 1. **Edit** the prompt template below
# MAGIC 2. **Run All** → registers a new version only if the template changed
# MAGIC 3. Review the smoke test output
# MAGIC 4. Iterate until satisfied, then the alias update promotes to `@production`
# MAGIC
# MAGIC The deployed agent loads `prompts:/{name}@production`, so updating the alias
# MAGIC is a **hot-reload** — no redeployment needed.

# COMMAND ----------
# MAGIC %pip install mlflow>=3.1 databricks-sdk databricks-openai
# MAGIC %restart_python

# COMMAND ----------
import mlflow
from mlflow.entities.model_registry import PromptModelConfig

CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
LLM_ENDPOINT = dbutils.widgets.get("llm_endpoint")
PROMPT_NAME = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('prompt_base_name')}"

print(f"Prompt: {PROMPT_NAME}")
print(f"LLM endpoint: {LLM_ENDPOINT}")
print(f"MLflow version: {mlflow.__version__}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: Define Prompt Templates
# MAGIC
# MAGIC Edit `CURRENT_TEMPLATE` to iterate on your prompt. The idempotency check
# MAGIC compares the latest registered version with this text — a new version
# MAGIC is only created if the template actually changed.

# COMMAND ----------

# --- v1: Basic prompt (registered on first run only, for version history) ---
INITIAL_TEMPLATE = """\
You are a corporate affairs assistant. Answer employee questions based on the provided context.

Context: {{context}}

Question: {{question}}
"""

# --- Current prompt: edit this to iterate ---
CURRENT_TEMPLATE = """\
You are a knowledgeable corporate affairs assistant helping employees navigate \
company policies, procedures, and general corporate information.

## Instructions
- Answer the employee's question based ONLY on the provided context documents.
- If the context does not contain enough information, clearly state that and suggest \
who to contact (e.g., HR, Legal, Finance).
- Cite specific documents or sections when possible.
- Use a professional but approachable tone.
- For policy questions, always note the effective date if available.

## Context from Corporate Documents
{{context}}

## Employee Question
{{question}}
"""

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: Register Prompt (Idempotent)
# MAGIC
# MAGIC Only creates a new version if the template text has changed.

# COMMAND ----------

prompt_exists = False
current_prompt = None

try:
    existing = mlflow.genai.load_prompt(PROMPT_NAME)
    prompt_exists = True
    if existing.template.strip() == CURRENT_TEMPLATE.strip():
        print(f"Prompt '{PROMPT_NAME}' v{existing.version} already up to date — skipping")
        current_prompt = existing
    else:
        print(f"Template changed — registering new version")
        current_prompt = mlflow.genai.register_prompt(
            name=PROMPT_NAME,
            template=CURRENT_TEMPLATE,
            commit_message="Updated prompt template",
        )
        print(f"Registered v{current_prompt.version}")
except Exception as e:
    if "RESOURCE_DOES_NOT_EXIST" in str(e) or "not found" in str(e).lower():
        print("Prompt not found — creating initial versions")
    else:
        raise

# First run only: register v1 (for history) then current
if not prompt_exists and current_prompt is None:
    v1 = mlflow.genai.register_prompt(
        name=PROMPT_NAME,
        template=INITIAL_TEMPLATE,
        commit_message="v1: Basic corporate affairs prompt",
    )
    print(f"Registered initial v{v1.version}")

    current_prompt = mlflow.genai.register_prompt(
        name=PROMPT_NAME,
        template=CURRENT_TEMPLATE,
        commit_message="v2: Enhanced with structured instructions, citations, and fallback guidance",
    )
    print(f"Registered v{current_prompt.version}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: Attach Model Configuration
# MAGIC
# MAGIC Store recommended LLM parameters alongside the prompt version.
# MAGIC The agent reads these at runtime via `prompt.model_config`.

# COMMAND ----------

model_config = PromptModelConfig(
    model_name=LLM_ENDPOINT,
    temperature=0.1,
    max_tokens=1000,
)

mlflow.genai.set_prompt_model_config(
    name=PROMPT_NAME,
    version=current_prompt.version,
    model_config=model_config,
)

print(f"Model config attached to v{current_prompt.version}")

# Verify by loading with the prompts:/ URI format (same pattern the agent uses)
loaded = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}/{current_prompt.version}")
print(f"Verified: model_config = {loaded.model_config}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Smoke Test
# MAGIC
# MAGIC Quick sanity check that the prompt produces reasonable output.
# MAGIC This is **not** a systematic evaluation — that happens in NB05 where the
# MAGIC full RAG agent (retrieval + prompt + LLM) is scored with `mlflow.genai.evaluate()`.

# COMMAND ----------
from databricks_openai import DatabricksOpenAI

openai_client = DatabricksOpenAI()

sample_context = """
Document: Employee Handbook - Remote Work Policy (Effective: Jan 2025)
Section 3.2: Employees in eligible roles may work remotely up to 3 days per week.
Remote work arrangements must be approved by the employee's direct manager.
Equipment stipend of $500 is available for home office setup.
"""

sample_question = "How many days can I work from home?"

# Load using prompts:/ URI (same pattern as the deployed agent)
prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}/{current_prompt.version}")
formatted = prompt.format(context=sample_context, question=sample_question)

response = openai_client.chat.completions.create(
    model=prompt.model_config["model_name"],
    messages=[{"role": "user", "content": formatted}],
    temperature=prompt.model_config["temperature"],
    max_tokens=prompt.model_config["max_tokens"],
)

print(f"Prompt: v{current_prompt.version}")
print(f"Model:  {prompt.model_config['model_name']}")
print(f"\nQ: {sample_question}")
print(f"\nA: {response.choices[0].message.content}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 5: Set Production Alias
# MAGIC
# MAGIC Promote this version to `@production`. The deployed agent loads
# MAGIC `prompts:/{name}@production` so this takes effect immediately — no redeployment needed.

# COMMAND ----------

mlflow.genai.set_prompt_alias(
    name=PROMPT_NAME,
    alias="production",
    version=current_prompt.version,
)

print(f"Alias '@production' → v{current_prompt.version}")

# Verify alias-based loading (this is exactly what the agent does at runtime)
prod_prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@production")
print(f"Verified: prompts:/{PROMPT_NAME}@production → v{prod_prompt.version}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | API | Purpose |
# MAGIC |-----|---------|
# MAGIC | `mlflow.genai.register_prompt()` | Create a versioned prompt template |
# MAGIC | `mlflow.genai.set_prompt_model_config()` | Attach LLM parameters to a version |
# MAGIC | `mlflow.genai.load_prompt("prompts:/{name}/{version}")` | Load a specific version |
# MAGIC | `mlflow.genai.load_prompt("prompts:/{name}@alias")` | Load by alias (used by deployed agent) |
# MAGIC | `prompt.format(...)` | Fill template variables |
# MAGIC | `mlflow.genai.set_prompt_alias()` | Point alias to a version (hot-reload) |
# MAGIC
# MAGIC **Next:** NB04 logs the agent (which references this prompt), then NB05 evaluates
# MAGIC the full RAG pipeline with `mlflow.genai.evaluate()` and quality gates.
