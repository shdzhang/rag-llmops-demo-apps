# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Production Monitoring (Trace-Based)
# MAGIC
# MAGIC Production monitoring for the Databricks Apps agent. In the Apps model,
# MAGIC **MLflow traces** are the primary observability signal (no inference tables).
# MAGIC
# MAGIC **Part A: Automated Quality Monitoring (MLflow External Monitor)**
# MAGIC - Runs LLM judges on a sample of production traces automatically
# MAGIC - Uses the same scorers from offline evaluation for consistency
# MAGIC - Results appear in the MLflow Experiment Traces tab
# MAGIC
# MAGIC **Part B: Trace-Based Analytics**
# MAGIC - Query volume and trends via `mlflow.search_traces()`
# MAGIC - Latency analysis (P50, P95, P99)
# MAGIC - Error rate monitoring
# MAGIC - Token usage tracking
# MAGIC
# MAGIC For more details see:
# MAGIC - [Production quality monitoring](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/run-scorer-in-prod)

# COMMAND ----------
# MAGIC %pip install -U mlflow[databricks]>=3.1.1 databricks-agents databricks-sdk pandas
# MAGIC %restart_python

# COMMAND ----------
import os
import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient
from datetime import datetime, timedelta

# --- Configuration (from DAB job parameters) ---
CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")
APP_NAME = dbutils.widgets.get("app_name")

w = WorkspaceClient()

print(f"Monitoring app:     {APP_NAME}")
print(f"Experiment:         {EXPERIMENT_NAME}")

# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC ## Part A: Automated Quality Monitoring
# MAGIC
# MAGIC MLflow's production monitoring runs quality assessments on a **sample** of
# MAGIC production traces, using the **same scorers** from offline evaluation.

# COMMAND ----------
# MAGIC %md
# MAGIC ### Step 1: Define Assessment Scorers

# COMMAND ----------

from databricks.agents.monitoring import (
    AssessmentsSuiteConfig,
    GuidelinesJudge,
    BuiltinJudge,
    create_external_monitor,
    get_external_monitor,
    update_external_monitor,
)

mlflow.set_experiment(EXPERIMENT_NAME)

builtin_judges = [
    BuiltinJudge(name="safety"),
    BuiltinJudge(name="groundedness", sample_rate=0.4),
    BuiltinJudge(name="relevance_to_query"),
]

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


monitor = get_or_create_monitor(sample_rate=1.0)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Monitor Status

# COMMAND ----------

print(f"""
Production Quality Monitor Active!
====================================
Experiment:    {EXPERIMENT_NAME}
Sample rate:   100% (adjust for production cost)
Judges:        {len(assessments)} configured

The monitor runs automatically every ~15 minutes.
Results appear in MLflow Experiment -> Traces tab.
""")

# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC ## Part B: Trace-Based Analytics
# MAGIC
# MAGIC In the Apps model, MLflow traces (via `mlflow.openai.autolog()`) are the
# MAGIC primary observability signal. We query them via `mlflow.search_traces()`.

# COMMAND ----------
# MAGIC %md
# MAGIC ### 1. Recent Trace Summary

# COMMAND ----------

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    print(f"Experiment '{EXPERIMENT_NAME}' not found. Skipping trace analytics.")
    dbutils.notebook.exit("Experiment not found - monitor created successfully")

experiment_id = experiment.experiment_id

traces_df = mlflow.search_traces(
    experiment_ids=[experiment_id],
    max_results=500,
)

if traces_df.empty:
    print("No traces found yet. Send some queries to the app first.")
    print(f"  App URL: https://{APP_NAME}-*.databricksapps.com")
    dbutils.notebook.exit("No traces yet - monitor created successfully")

print(f"Found {len(traces_df)} traces in experiment {experiment_id}")
print(f"Available columns: {list(traces_df.columns)}")
safe_cols = [c for c in ["trace_id", "state", "execution_duration", "request_time"] if c in traces_df.columns]
print(traces_df[safe_cols].head(10).to_string() if safe_cols else f"{len(traces_df)} traces found")

# COMMAND ----------
# MAGIC %md
# MAGIC ### 2. Query Volume by Day

# COMMAND ----------

if "request_time" in traces_df.columns:
    traces_df["timestamp"] = pd.to_datetime(traces_df["request_time"])
elif "timestamp_ms" in traces_df.columns:
    traces_df["timestamp"] = pd.to_datetime(traces_df["timestamp_ms"], unit="ms")

_state_col = "state" if "state" in traces_df.columns else "status" if "status" in traces_df.columns else None

if "timestamp" in traces_df.columns and _state_col:
    volume_df = (
        traces_df
        .assign(day=traces_df["timestamp"].dt.date)
        .groupby("day")
        .agg(
            total_requests=("trace_id", "count"),
            errors=(_state_col, lambda x: (x == "ERROR").sum()),
        )
        .assign(success_rate_pct=lambda df: round((1 - df["errors"] / df["total_requests"]) * 100, 1))
        .sort_index(ascending=False)
    )
    display(volume_df) if "display" in dir() else print(volume_df)
else:
    print("Timestamp or state column not available for volume analysis")

# COMMAND ----------
# MAGIC %md
# MAGIC ### 3. Latency Analysis

# COMMAND ----------

if "execution_duration" in traces_df.columns:
    latency_col = "execution_duration"
elif "execution_time_ms" in traces_df.columns:
    latency_col = "execution_time_ms"
elif "duration_ms" in traces_df.columns:
    latency_col = "duration_ms"
else:
    latency_col = None

if latency_col:
    latency_series = pd.to_timedelta(traces_df[latency_col]).dt.total_seconds() * 1000 if traces_df[latency_col].dtype == "timedelta64[ns]" else pd.to_numeric(traces_df[latency_col], errors="coerce")
    latency_series = latency_series.dropna()
    if not latency_series.empty:
        print(f"Latency Statistics ({len(latency_series)} traces, column={latency_col}):")
        print(f"  Mean:   {latency_series.mean():.0f}ms")
        print(f"  Median: {latency_series.median():.0f}ms")
        print(f"  P95:    {latency_series.quantile(0.95):.0f}ms")
        print(f"  P99:    {latency_series.quantile(0.99):.0f}ms")
        print(f"  Max:    {latency_series.max():.0f}ms")
    else:
        print("No latency data available")
else:
    print("No latency column found in traces. Available columns:", list(traces_df.columns))

# COMMAND ----------
# MAGIC %md
# MAGIC ### 4. Error Analysis

# COMMAND ----------

_state_col_err = "state" if "state" in traces_df.columns else "status" if "status" in traces_df.columns else None

if _state_col_err:
    error_counts = traces_df[_state_col_err].value_counts()
    total = len(traces_df)
    print("Status Distribution:")
    for status_val, count in error_counts.items():
        pct = count / total * 100
        print(f"  {status_val}: {count} ({pct:.1f}%)")

    error_traces = traces_df[traces_df[_state_col_err] == "ERROR"]
    if not error_traces.empty:
        print(f"\nRecent errors ({len(error_traces)} total):")
        for _, row in error_traces.head(5).iterrows():
            trace_id = row.get("trace_id", "unknown")
            print(f"  Trace {trace_id}")
else:
    print("No state/status column in traces")

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

Part B - Trace-Based Analytics:
  Source:      mlflow.search_traces()
  Traces:      {len(traces_df) if 'traces_df' in dir() else 'N/A'}
  Metrics:     Volume, latency, errors

Recommended next steps:
  1. Review traces in the MLflow Experiment UI
  2. Create Databricks SQL Alerts for error rate and latency
  3. Adjust sample rate for high-traffic production (e.g., 0.1 for 10%)
""")
