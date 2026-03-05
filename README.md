# RAG LLMOps Demo (Databricks Apps)

End-to-end LLMOps demonstration for a RAG chatbot agent deployed on **Databricks Apps**, showcasing the complete lifecycle from data preparation through production monitoring.

## Apps LLMOps: The Key Insight

In Databricks Apps, **source code is the deployment artifact** — not an MLflow model version. This eliminates the `log_model()` -> `register_model()` -> `promote @champion` -> `agents.deploy()` pipeline. Instead:

```
edit code  ->  evaluate(code)  ->  git commit  ->  bundle deploy
```

Prompts hot-reload via MLflow Prompt Registry (no redeploy needed). Quality gates run in NB04 and block deployment if thresholds fail.

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
├── agent_server/                   # Agent code (deployed as App)
│   ├── agent.py                    # Async @invoke/@stream agent with RAG
│   ├── start_server.py             # AgentServer startup
│   └── evaluate_agent.py           # Local evaluation runner (ConversationSimulator)
├── scripts/                        # Development scripts
│   ├── quickstart.py               # Setup script (auth, env, experiment)
│   └── start_app.py                # Local dev server launcher
├── notebooks/                      # Databricks notebooks
│   ├── 01_data_ingestion.py        # Ingest documents to Delta
│   ├── 02_vector_index_creation.py # Create Vector Search index
│   ├── 03_prompt_engineering.py    # Register prompts in MLflow Prompt Registry
│   ├── 04_agent_evaluation.py      # Evaluate agent directly + quality gates
│   ├── 05_inference_testing.py     # Post-deploy smoke tests (direct function import)
│   └── 06_monitoring_dashboard.py  # Trace-based production monitoring
├── resources/                      # DAB resource definitions
│   ├── data_preparation.job.yml    # Data pipeline job
│   ├── build_evaluate.job.yml      # Prompt + evaluate pipeline job
│   └── monitoring.job.yml          # Smoke tests + monitoring job (scheduled)
├── tests/                          # Unit tests
│   └── test_agent_local.py         # Local agent function tests
├── docs/                           # Documentation
│   ├── architecture.md             # Architecture overview + design decisions
│   ├── deployment_guide.md         # Phased deployment + CI/CD guide
│   └── troubleshooting_guide.md    # Known issues and fixes
├── app.yaml                        # Databricks App runtime config
├── databricks.yml                  # DAB bundle configuration
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

### 2. Test Locally

```bash
uv run start-app
# Open http://localhost:8000 for the chat UI
```

### 3. Deploy to Databricks

```bash
databricks bundle deploy -t prod
databricks bundle run corp_chatbot_app -t prod
```

### 4. Run Individual Jobs

```bash
databricks bundle run data_preparation -t prod   # Run once to set up data
databricks bundle run build_evaluate -t prod      # Evaluate agent quality
databricks bundle run monitoring -t prod          # Smoke tests + monitoring
```

## LLMOps Lifecycle

```
Data Prep            Develop & Evaluate      Deploy (CLI/CI-CD)      Monitor
  (01, 02)             (03, 04)                                       (05, 06)
  [run once]           [inner loop]           bundle deploy            [scheduled 6h]

  Ingest docs,         Register prompts,      bundle deploy,          Smoke tests,
  create VS index      evaluate quality       run app                 trace monitoring
```

### DAB Jobs

| Job | Notebooks | When to run |
|-----|-----------|-------------|
| `data_preparation` | 01, 02 | Once, or when documents change |
| `build_evaluate` | 03, 04 | Each code/prompt change (inner dev loop) |
| `monitoring` | 05, 06 | Scheduled every 6h (post-deploy validation) |

**Deployment is a CLI/CI/CD step**, not a DAB job. See [deployment guide](docs/deployment_guide.md) for CI/CD workflow.

## What Changed from Model Serving

| Before (Model Serving) | After (Apps) |
|------------------------|--------------|
| `mlflow.pyfunc.log_model()` -> register -> promote `@champion` | Edit code, evaluate, git commit |
| `agents.deploy(model_name, version)` | `databricks bundle deploy` |
| Inference tables for monitoring | MLflow traces via `autolog()` |
| 4 DAB jobs (data, build, deploy, monitor) | 3 independent DAB jobs (data, evaluate, monitor) |
| Old NB04-07 (log, register, deploy) | Removed — notebooks renumbered to 01-06 |

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
