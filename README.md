# RAG LLMOps Demo (Databricks Apps)

End-to-end LLMOps demonstration for a RAG chatbot agent deployed on **Databricks Apps**, showcasing the complete lifecycle from data preparation through production monitoring.

Migrated from Model Serving (`ResponsesAgent` class with `agents.deploy()`) to Databricks Apps (async `@invoke`/`@stream` functions with `databricks bundle deploy`).

## Key Features

| Feature | Implementation |
|---------|---------------|
| **Agent Framework** | MLflow AgentServer with async `@invoke`/`@stream` |
| **Prompt Management** | MLflow Prompt Registry with versioning and aliases |
| **Retrieval** | Databricks Vector Search (managed embeddings + similarity search) |
| **Evaluation** | `mlflow.genai.evaluate()` with built-in scorers |
| **Deployment** | Databricks Apps via Asset Bundles (`databricks bundle deploy`) |
| **Auth** | App service principal + per-user authorization via `get_user_workspace_client()` |
| **Chat UI** | Built-in chat UI (streaming, markdown, authentication) |
| **Monitoring** | MLflow External Monitor + tracing |
| **Orchestration** | Databricks Asset Bundles (DABs) |

## Project Structure

```
rag-llmops-demo-apps/
├── agent_server/                   # Agent server code (deployed as App)
│   ├── agent.py                    # Async @invoke/@stream agent with RAG
│   ├── start_server.py             # AgentServer startup
│   ├── utils.py                    # User auth helpers
│   └── evaluate_agent.py           # Evaluation runner
├── scripts/                        # Development scripts
│   ├── quickstart.py               # Setup script (auth, env, experiment)
│   └── start_app.py                # Local dev server launcher
├── notebooks/                      # Databricks notebooks (run sequentially)
│   ├── 01_data_ingestion.py        # Ingest documents to Delta
│   ├── 02_vector_index_creation.py # Create Vector Search index
│   ├── 03_prompt_engineering.py    # Register prompts in MLflow Prompt Registry
│   ├── 04_agent_build.py           # Log agent to MLflow + UC for eval tracking
│   ├── 05_agent_evaluation.py      # Evaluate with mlflow.genai.evaluate()
│   ├── 06_model_registration.py    # Model governance (aliases, tags)
│   ├── 07_app_deployment.py        # Deploy as Databricks App
│   ├── 08_inference_testing.py     # App testing & latency benchmarks
│   └── 09_monitoring_dashboard.py  # Production monitoring queries
├── resources/                      # DAB resource definitions
│   ├── model_artifacts.yml         # UC model, schema
│   ├── data_preparation.job.yml    # Data pipeline job
│   ├── build_evaluate.job.yml      # Build + evaluate pipeline job
│   ├── deploy.job.yml              # App deployment job
│   ├── monitoring.job.yml          # Scheduled monitoring job
│   └── end_to_end.job.yml          # Orchestrator job (runs all stages)
├── data/knowledge_base/            # Sample corporate documents
├── docs/                           # Documentation
├── app.yaml                        # Databricks App configuration
├── databricks.yml                  # DAB bundle configuration
├── pyproject.toml                  # Python dependencies
├── requirements.txt                # "uv" (required for Apps runtime)
└── README.md
```

## Quick Start

### Prerequisites

- Databricks workspace with Unity Catalog enabled
- Databricks CLI configured (`databricks auth login`)
- Python 3.11+
- `uv` package manager
- Node.js 20+ (for chat UI)

### 1. Set Up Local Environment

```bash
cd rag-llmops-demo-apps
uv run quickstart
```

### 2. Test Locally

```bash
uv run start-app
# Open http://localhost:8000 for the chat UI
```

### 3. Deploy to Databricks Apps

```bash
databricks bundle deploy -t dev
databricks bundle run corp_chatbot_app -t dev
```

### 4. Run the Full Pipeline

```bash
databricks bundle run end_to_end -t dev
```

## LLMOps Lifecycle

```
Data Prep -> Prompt Registry -> Agent Build -> Evaluate -> Deploy App -> Test -> Monitor
  (01, 02)      (03)              (04)         (05, 06)     (07)        (08)    (09)
```

1. **Data Preparation**: Ingest docs, create Vector Search index
2. **Prompt Engineering**: Register versioned prompts, set `@production` alias
3. **Agent Build**: Log agent to MLflow + UC for evaluation tracking
4. **Evaluation**: Run `mlflow.genai.evaluate()`, enforce quality gates
5. **App Deployment**: Deploy as Databricks App with built-in Chat UI
6. **Monitoring**: MLflow External Monitor (automated judges) + tracing

## Technologies

- **MLflow 3.x**: AgentServer, Prompt Registry, GenAI Evaluate, Tracing
- **Databricks Foundation Models**: Claude Sonnet 4.5 via AI Gateway
- **Databricks Vector Search**: Managed embedding + retrieval
- **Databricks Apps**: Async FastAPI server with built-in chat UI
- **Unity Catalog**: Model registry, governance, permissions
- **Databricks Asset Bundles**: Infrastructure as Code
