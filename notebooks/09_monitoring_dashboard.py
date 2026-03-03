# Databricks notebook source
# MAGIC %md
# MAGIC # 09 - Production Monitoring
# MAGIC
# MAGIC This notebook sets up **production-grade monitoring** for the deployed agent:
# MAGIC
# MAGIC **Part A: Automated Quality Monitoring (MLflow External Monitor)**
# MAGIC - Runs LLM judges on a sample of production traces automatically
# MAGIC - Uses the same scorers from offline evaluation for consistency
# MAGIC - Results appear in the MLflow Experiment Traces tab
# MAGIC
# MAGIC **Part B: Inference Table Analytics**
# MAGIC - Query volume and trends
# MAGIC - Latency analysis (P50, P95, P99)
# MAGIC - Error rate monitoring
# MAGIC - Token usage and cost estimation
# MAGIC
# MAGIC For more details see:
# MAGIC - [Production quality monitoring](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/run-scorer-in-prod)
# MAGIC - [AI Gateway inference tables](https://docs.databricks.com/gcp/en/ai-gateway/inference-tables)

# COMMAND ----------
# MAGIC %pip install -U mlflow[databricks]>=3.1.1 databricks-agents databricks-sdk pandas
# MAGIC %restart_python

# COMMAND ----------
import os
import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient

# --- Configuration (from DAB job parameters) ---
CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
MODEL_NAME = dbutils.widgets.get("model_name")
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")

# App name (from DAB job parameters)
APP_NAME = dbutils.widgets.get("app_name")
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

# Inference table — for Databricks Apps, traces are logged to the MLflow
# experiment. If an AI Gateway inference table exists (e.g. from a previous
# Model Serving deployment), we can still query it for historical analytics.
_model_table_name = MODEL_NAME.replace("-", "_")
INFERENCE_TABLE = f"{CATALOG}.{SCHEMA}.`{_model_table_name}_payload`"

w = WorkspaceClient()

print(f"Monitoring app:     {APP_NAME}")
print(f"Inference table:    {INFERENCE_TABLE}")
print(f"Experiment:         {EXPERIMENT_NAME}")

# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC ## Part A: Automated Quality Monitoring
# MAGIC
# MAGIC MLflow's production monitoring automatically runs quality assessments on a
# MAGIC **sample** of production traffic, ensuring the agent maintains high quality
# MAGIC without manual intervention.
# MAGIC
# MAGIC This uses the **same scorers** from offline evaluation (notebook 05) in
# MAGIC production, giving you consistent quality measurement across the entire
# MAGIC application lifecycle -- dev to prod.

# COMMAND ----------
# MAGIC %md
# MAGIC ### Step 1: Define Assessment Scorers
# MAGIC
# MAGIC We use a mix of **built-in judges** (safety, groundedness, relevance) and
# MAGIC **custom guideline judges** (accuracy, professional tone) that mirror our
# MAGIC offline evaluation criteria.

# COMMAND ----------

from databricks.agents.monitoring import (
    AssessmentsSuiteConfig,
    GuidelinesJudge,
    BuiltinJudge,
    create_external_monitor,
    get_external_monitor,
    update_external_monitor,
)

# Set the experiment so the monitor attaches results to the right place
mlflow.set_experiment(EXPERIMENT_NAME)

# --- Built-in judges ---
# These are optimized, pre-built evaluators from Databricks
builtin_judges = [
    BuiltinJudge(name="safety"),
    BuiltinJudge(name="groundedness", sample_rate=0.4),
    BuiltinJudge(name="relevance_to_query"),
]

# --- Custom guideline judges ---
# These match our offline evaluation criteria from notebook 05
accuracy_guidelines = [
    """
    The response correctly references factual information based on these rules:
      - All factual information must be sourced from company policy documents
      - Policy details (dates, amounts, durations) must be accurate
      - If the information is not available, the response should clearly state that
      - AUTOMATIC FAIL if any fabricated policy information is presented as fact
    """,
]

professional_tone_guidelines = [
    """
    The response maintains a professional and helpful tone:
      - Should not be overly casual, use slang, or be dismissive
      - Should be clear and actionable
      - Should cite specific documents or sections when possible
      - For policy questions, should note the effective date if available
    """,
]

guideline_judges = [
    GuidelinesJudge(guidelines={
        "accuracy": accuracy_guidelines,
        "professional_tone": professional_tone_guidelines,
    }),
]

assessments = builtin_judges + guideline_judges

print(f"Configured {len(assessments)} assessment judges:")
for a in assessments:
    if isinstance(a, BuiltinJudge):
        rate = f" (sample_rate={a.sample_rate})" if hasattr(a, "sample_rate") and a.sample_rate else ""
        print(f"  - BuiltinJudge: {a.name}{rate}")
    elif isinstance(a, GuidelinesJudge):
        print(f"  - GuidelinesJudge: {list(a.guidelines.keys())}")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Step 2: Create or Update the External Monitor
# MAGIC
# MAGIC The monitor runs automatically every ~15 minutes after creation. Each run:
# MAGIC 1. Samples production traces at the configured rate
# MAGIC 2. Runs each judge on the sampled traces
# MAGIC 3. Attaches feedback to each trace in the MLflow Experiment
# MAGIC 4. Writes all traces to a Delta Table (`trace_logs_<experiment_id>`)

# COMMAND ----------

def get_or_create_monitor(sample_rate: float = 1.0):
    """Create or update the external monitor for production quality assessment."""
    config = AssessmentsSuiteConfig(
        sample=sample_rate,
        assessments=assessments,
    )
    try:
        existing = get_external_monitor(experiment_name=EXPERIMENT_NAME)
        print(f"Monitor already exists - updating with latest scorers...")
        updated = update_external_monitor(
            experiment_name=EXPERIMENT_NAME,
            assessments_config=config,
        )
        print(f"Monitor updated: {updated}")
        return updated
    except Exception as e:
        if "No monitor found" in str(e) or "does not exist" in str(e) or "NoMonitorFoundError" in type(e).__name__:
            print(f"No existing monitor — creating new external monitor...")
            monitor = create_external_monitor(
                catalog_name=CATALOG,
                schema_name=SCHEMA,
                assessments_config=config,
            )
            print(f"Monitor created: {monitor}")
            return monitor
        else:
            raise


# Create the monitor with 100% sampling (adjust for production cost)
# For high-traffic endpoints, use a lower sample rate (e.g., 0.1 for 10%)
monitor = get_or_create_monitor(sample_rate=1.0)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Monitor Status
# MAGIC
# MAGIC The monitoring job takes ~15-30 minutes for the initial run. After that,
# MAGIC it runs every 15 minutes. View results in:
# MAGIC - **MLflow UI -> Traces tab** (filter by experiment)
# MAGIC - **Delta Table**: `trace_logs_<experiment_id>` in the configured schema

# COMMAND ----------

print(f"""
Production Quality Monitor Active!
====================================
Experiment:    {EXPERIMENT_NAME}
Sample rate:   100% (adjust for production cost)
Judges:        {len(assessments)} configured

What happens next:
  1. The monitor runs automatically every ~15 minutes
  2. It samples production traces and runs all judges
  3. Results appear in the MLflow Experiment -> Traces tab
  4. A Delta Table with all traces is created in {CATALOG}.{SCHEMA}

To view results:
  - MLflow UI: Traces tab in the experiment
  - SQL: SELECT * FROM {CATALOG}.{SCHEMA}.trace_logs_<experiment_id>

To adjust sample rate for high-traffic production:
  update_external_monitor(
      experiment_name="{EXPERIMENT_NAME}",
      assessments_config=AssessmentsSuiteConfig(sample=0.1, assessments=assessments)
  )
""")

# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC ## Part B: Inference Table Analytics
# MAGIC
# MAGIC For Databricks Apps, traces are logged to the MLflow experiment. If an
# MAGIC AI Gateway inference table exists (e.g. from a prior Model Serving deployment),
# MAGIC these queries provide historical operational analytics.

# COMMAND ----------
# MAGIC %md
# MAGIC ### Check: Does the Inference Table Exist?

# COMMAND ----------

try:
    row_count = spark.sql(f"SELECT count(*) AS cnt FROM {INFERENCE_TABLE}").first()["cnt"]
    print(f"Inference table has {row_count} rows")
except Exception as e:
    print(f"Inference table not found or not yet populated: {e}")
    print("Deploy the agent (notebook 07) and send some queries first.")
    print("Skipping inference table analytics...")
    dbutils.notebook.exit("Inference table not ready - monitor created successfully")

# COMMAND ----------
# MAGIC %md
# MAGIC ### 1. Query Volume Trends

# COMMAND ----------

query_volume_df = spark.sql(f"""
    SELECT
      date_trunc('day', request_time) AS day,
      count(*) AS total_requests,
      count(CASE WHEN status_code = 200 THEN 1 END) AS successful,
      count(CASE WHEN status_code != 200 THEN 1 END) AS failed,
      round(count(CASE WHEN status_code = 200 THEN 1 END) * 100.0 / count(*), 1) AS success_rate_pct
    FROM {INFERENCE_TABLE}
    WHERE request_time >= date_sub(current_date(), 30)
    GROUP BY 1
    ORDER BY 1 DESC
""")

display(query_volume_df) if "display" in dir() else query_volume_df.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ### 2. Latency Analysis (P50, P95, P99)

# COMMAND ----------

latency_df = spark.sql(f"""
    SELECT
      date_trunc('day', request_time) AS day,
      count(*) AS requests,
      round(avg(execution_duration_ms), 0) AS avg_latency_ms,
      round(percentile(execution_duration_ms, 0.5), 0) AS p50_ms,
      round(percentile(execution_duration_ms, 0.95), 0) AS p95_ms,
      round(percentile(execution_duration_ms, 0.99), 0) AS p99_ms
    FROM {INFERENCE_TABLE}
    WHERE request_time >= date_sub(current_date(), 30)
      AND status_code = 200
    GROUP BY 1
    ORDER BY 1 DESC
""")

display(latency_df) if "display" in dir() else latency_df.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ### 3. Error Analysis

# COMMAND ----------

error_df = spark.sql(f"""
    SELECT
      status_code,
      count(*) AS count,
      round(count(*) * 100.0 / sum(count(*)) OVER(), 1) AS pct
    FROM {INFERENCE_TABLE}
    WHERE request_time >= date_sub(current_date(), 7)
    GROUP BY 1
    ORDER BY 2 DESC
""")

display(error_df) if "display" in dir() else error_df.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ### 4. Token Usage & Cost Estimation

# COMMAND ----------

token_df = spark.sql(f"""
    SELECT
      date_trunc('day', request_time) AS day,
      count(*) AS requests,
      sum(get_json_object(response, '$.usage.prompt_tokens')) AS total_input_tokens,
      sum(get_json_object(response, '$.usage.completion_tokens')) AS total_output_tokens,
      sum(get_json_object(response, '$.usage.total_tokens')) AS total_tokens
    FROM {INFERENCE_TABLE}
    WHERE request_time >= date_sub(current_date(), 30)
      AND status_code = 200
    GROUP BY 1
    ORDER BY 1 DESC
""")

display(token_df) if "display" in dir() else token_df.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ### 5. Popular Questions

# COMMAND ----------

questions_df = spark.sql(f"""
    SELECT
      COALESCE(
        get_json_object(request, '$.input[0].content'),
        get_json_object(request, '$.messages[0].content')
      ) AS question,
      count(*) AS frequency
    FROM {INFERENCE_TABLE}
    WHERE request_time >= date_sub(current_date(), 7)
      AND status_code = 200
    GROUP BY 1
    ORDER BY 2 DESC
    LIMIT 20
""")

display(questions_df) if "display" in dir() else questions_df.show(truncate=False)

# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC ## Summary

# COMMAND ----------

print(f"""
Production Monitoring Setup Complete!
=======================================

Part A - Automated Quality Monitor:
  App:         {APP_NAME}
  Experiment:  {EXPERIMENT_NAME}
  Judges:      safety, groundedness, relevance, accuracy, professional_tone
  Sample rate: 100%
  Frequency:   Every ~15 minutes (automatic)
  Results:     MLflow Experiment -> Traces tab

Part B - Inference Table Analytics:
  Table:       {INFERENCE_TABLE}
  Metrics:     Volume, latency, errors, tokens, popular questions

Recommended next steps:
  1. Review traces in the MLflow Experiment UI
  2. Create Databricks SQL Alerts for error rate > 5% and P95 latency > 10s
  3. Adjust sample rate for high-traffic production (e.g., 0.1 for 10%)
  4. Add custom guidelines as your agent evolves
""")
