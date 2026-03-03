# Architecture Overview

## LLMOps Lifecycle

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Data Prep  │───>│   Prompt    │───>│ Agent Build │───>│  Evaluate   │───>│   Deploy    │───>│   Monitor   │
│              │    │  Registry   │    │  & Log      │    │  & Promote  │    │  & Test     │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
  Notebooks:         Notebook:          Notebook:          Notebooks:         Notebooks:         Notebook:
  01, 02             03                 04                 05, 06             07, 08             09
```

## Component Details

### 1. Data Preparation (01, 02)
- Ingest documents into Delta tables
- Chunk and create embeddings
- Build Vector Search index

### 2. Prompt Engineering (03)
- Register prompts in **MLflow Prompt Registry**
- Version prompts with commit messages
- Attach model configuration (temperature, max_tokens)
- Set `@production` alias for deployment

### 3. Agent Build (04)
- **ResponsesAgent** (MLflow 3.x) - modern agent interface
- File-based logging with `mlflow.pyfunc.log_model(python_model="agent/rag_agent.py")`
- Declare resources: `DatabricksServingEndpoint`, `DatabricksVectorSearchIndex`
- Register to Unity Catalog with `@candidate` alias

### 4. Evaluation (05)
- **mlflow.genai.evaluate()** with built-in scorers
- Offline scorer: `Correctness` (LLM judge: Claude Opus 4.1) -- keeps evaluation fast (~10-15 min)
- Additional quality checks (tone, citation, safety, groundedness) run in production via External Monitor (NB09)
- Quality gates: Minimum correctness threshold
- Auto-promote to `@champion` on pass

### 5. Deployment & Testing (07, 08)
- **databricks.agents.deploy()** for managed deployment
- Enables AI Gateway, Inference Tables, Review App
- Built-in OBO via `CredentialStrategy.MODEL_SERVING_USER_CREDENTIALS` for per-user Vector Search permissions
- Optional: Full OBO via Databricks Apps for custom web UI (see `docs/obo_setup_guide.md`)
- Endpoint testing: basic queries, edge cases, latency benchmarks

### 6. Monitoring (09)
- **MLflow External Monitor** for automated quality assessment on production traces
  - Built-in judges: safety, groundedness, relevance
  - Custom guideline judges: accuracy, professional tone
  - Configurable sampling rate (100% default, lower for high traffic)
- AI Gateway Inference Tables for operational analytics
  - Latency percentiles (P50, P95, P99)
  - Token usage and cost estimation
  - Error rate and popular questions

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Agent Framework | MLflow 3.x ResponsesAgent |
| LLM | Databricks Foundation Models (Claude Sonnet 4.5 for RAG, Opus 4.1 for judge) |
| Retrieval | Databricks Vector Search |
| Prompt Management | MLflow Prompt Registry |
| Evaluation | MLflow GenAI Evaluate |
| Model Registry | Unity Catalog |
| Deployment | databricks.agents.deploy() |
| Monitoring | MLflow External Monitor + AI Gateway Inference Tables |
| Orchestration | Databricks Asset Bundles (DABs) |

### Orchestration (End-to-End Job)
- Single-trigger `end_to_end` job chains all stages via `run_job_task`
- DAB `mode: development` adds `[dev <username>]` prefix to all resources
- Individual jobs can also be run independently for debugging

## Key Design Decisions

1. **ResponsesAgent over ChatAgent**: MLflow 3.x's Responses API interface (uses `input`/`output`, not `messages`/`choices`)
2. **File-based logging**: Agent code is logged as a file, not pickled - more portable and debuggable
3. **Prompt Registry over local files**: Version control, aliases, and hot-reload without redeployment
4. **agents.deploy() over manual endpoint creation**: Automatic AI Gateway, inference tables, and review app
5. **UC aliases over stages**: `@candidate` / `@champion` instead of deprecated stage transitions
6. **Built-in scorers over custom evaluation**: MLflow-native evaluation with full tracking integration
