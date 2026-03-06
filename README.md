# RAG LLMOps Demo (Databricks Apps)

End-to-end LLMOps demonstration for a RAG chatbot agent deployed on **Databricks Apps**, showcasing the complete lifecycle from data preparation through production monitoring.

## Apps LLMOps: The Key Insight

In Databricks Apps, **source code is the deployment artifact** — not an MLflow model version. This eliminates the `log_model()` -> `register_model()` -> `promote @champion` -> `agents.deploy()` pipeline. Instead:

```
edit code  ->  evaluate(code)  ->  git commit  ->  bundle deploy
```

Prompts hot-reload via MLflow Prompt Registry (no redeploy needed). Quality gates run in NB04 (including edge cases) and block deployment if thresholds fail.

## Key Features

| Feature | Implementation |
|---------|---------------|
| **Agent Framework** | MLflow AgentServer with async `@invoke`/`@stream` |
| **Prompt Management** | MLflow Prompt Registry with versioning and aliases |
| **Retrieval** | Databricks Vector Search (managed embeddings + similarity search) |
| **Evaluation** | `mlflow.genai.evaluate()` — evaluates agent code directly |
| **Deployment** | Databricks Apps via Asset Bundles (`databricks bundle deploy`) |
| **Auth** | App service principal + per-user authorization |
| **Chat UI** | Built-in chat UI (streaming, markdown, authentication) |
| **Monitoring** | MLflow External Monitor + trace-based analytics |
| **Orchestration** | Databricks Asset Bundles (DABs) + CI/CD |

## Project Structure

```
rag-llmops-demo-apps/
├── .github/workflows/
│   └── deploy.yml                  # CI/CD: validate → eval → deploy prod
├── agent_server/                   # Agent code (deployed as App)
│   ├── agent.py                    # Async @invoke/@stream agent with RAG
│   ├── start_server.py             # AgentServer startup
│   └── evaluate_agent.py           # Local evaluation runner (ConversationSimulator)
├── scripts/                        # Development & operations scripts
│   ├── quickstart.py               # Setup script (auth, env, experiment)
│   ├── start_app.py                # Local dev server launcher
│   └── rollback.sh                 # Rollback to a previous git commit
├── notebooks/                      # Databricks notebooks
│   ├── 01_data_ingestion.py        # Ingest documents to Delta
│   ├── 02_vector_index_creation.py # Create Vector Search index
│   ├── 03_prompt_engineering.py    # Register prompts in MLflow Prompt Registry
│   ├── 04_agent_evaluation.py      # Evaluate agent + quality gates + regression cases
│   └── 05_monitoring_dashboard.py  # Monitoring + alerting + feedback loop
├── resources/                      # DAB resource definitions
│   ├── data_preparation.job.yml    # Data pipeline job
│   ├── build_evaluate.job.yml      # Prompt + evaluate pipeline job
│   └── monitoring.job.yml          # Production monitoring job (scheduled)
├── tests/                          # Unit tests
│   └── test_agent_local.py         # Local agent function tests
├── docs/                           # Documentation
│   ├── architecture.md             # Architecture overview + design decisions
│   ├── deployment_guide.md         # Phased deployment + CI/CD guide
│   └── troubleshooting_guide.md    # Known issues and fixes
├── app.yaml                        # Databricks App runtime config
├── databricks.yml                  # DAB bundle config (dev/prod)
├── pyproject.toml                  # Python dependencies
├── requirements.txt                # "uv" bootstrap for Apps runtime
└── README.md
```

## Quick Start

### Prerequisites

- Databricks workspace with Unity Catalog enabled
- Databricks CLI configured (`databricks auth login`)
- Python 3.11+
- `uv` package manager

### 1. Set Up Local Environment

```bash
cd rag-llmops-demo-apps
uv run quickstart
```

### 2. Deploy Infrastructure & Data

```bash
# Deploy bundle (creates schema, experiment, app resource, jobs)
databricks bundle deploy -t prod

# Ingest documents and create Vector Search index (run once)
databricks bundle run data_preparation -t prod
```

Wait for the data preparation job to complete (~5-10 min) before proceeding.

### 3. Evaluate & Deploy App

```bash
# Evaluate agent quality (prompt engineering + quality gates)
databricks bundle run build_evaluate -t prod

# Start the app
databricks bundle run corp_chatbot_app -t prod

# Set up production monitoring
databricks bundle run monitoring -t prod
```

### 4. Test Locally (Optional)

```bash
uv run start-app
# Open http://localhost:8000 for the chat UI
```

## LLMOps Lifecycle

```
Data Prep        Develop & Evaluate     CI/CD Pipeline         Monitor
  (01, 02)         (03, 04)               (.github/workflows)    (05)
  [run once]       [inner loop]           [on merge to main]     [scheduled 6h]

  Ingest docs,     Register prompts,     validate → eval →      Health check,
  create VS index  evaluate quality      dev → prod             alerting,
                   + edge cases                                  feedback loop
                   + regression cases
```

### Environments

| Environment | Target | Deployed by |
|------------|--------|-------------|
| dev | `-t dev` | Developer (manual) or CI/CD (eval on merge) |
| prod | `-t prod` | CI/CD (manual approval after eval passes) |

### DAB Jobs

| Job | Notebooks | When to run |
|-----|-----------|-------------|
| `data_preparation` | 01, 02 | Once, or when documents change |
| `build_evaluate` | 03, 04 | Each code/prompt change (CI quality gate) |
| `monitoring` | 05 | Scheduled every 6h (health check + alerting + feedback loop) |

See [deployment guide](docs/deployment_guide.md) for CI/CD workflow and rollback procedures.

## What Changed from Model Serving

| Before (Model Serving) | After (Apps) |
|------------------------|--------------|
| `mlflow.pyfunc.log_model()` -> register -> promote `@champion` | Edit code, evaluate, git commit |
| `agents.deploy(model_name, version)` | `databricks bundle deploy` |
| Inference tables for monitoring | MLflow traces via `autolog()` |
| 4 DAB jobs (data, build, deploy, monitor) | 3 independent DAB jobs + CI/CD deploy |
| Old NB04-07 (log, register, deploy) | Removed — renumbered to 01-05 with feedback loop |
| Single environment | dev / prod with CI/CD promotion |
| No deployment tracking | MLflow runs with git SHA + prompt version |

## Why Apps Over Model Serving

Databricks Apps shifts agents from MDLC (Model Development Lifecycle) to SDLC (Software Development Lifecycle):

- **Code is the artifact** — no `log_model()` / `register_model()` / `@champion` promotion cycle
- **Async architecture** — `@invoke`/`@stream` with better concurrency per instance
- **Developer-friendly** — local debugging, git versioning, IDE/AI coding tool support
- **AgentServer integration** — compatible with Playground, Multi-Agent Supervisor, and OneChatBot
- **Hot-reload prompts** — MLflow Prompt Registry changes take effect without redeployment

Model Serving remains the GA path for agents and is not deprecated. Choose Apps when you want faster iteration and SDLC patterns; choose Model Serving when you need inference tables or GA status for agents. See [architecture docs](docs/architecture.md) for a detailed comparison and current limitations.

## Technologies

- **MLflow 3.x**: AgentServer, Prompt Registry, GenAI Evaluate, Tracing
- **Databricks Foundation Models**: Claude Sonnet 4.5 via AI Gateway
- **Databricks Vector Search**: Managed embedding + retrieval
- **Databricks Apps**: Async FastAPI server with built-in chat UI
- **Unity Catalog**: Governance and permissions
- **Databricks Asset Bundles**: Infrastructure as Code
