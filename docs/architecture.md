# Architecture Overview

## Apps LLMOps Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Data Prep   │───>│  Develop &  │───>│   Deploy    │───>│  Monitor &  │
│  (run once)  │    │  Validate   │    │  (CLI/CICD) │    │  Validate   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
  Notebooks:         Notebooks:         CLI commands:      Notebooks:
  01, 02             03, 04             bundle deploy      05, 06
                                        bundle run app
```

### DAB Jobs (independent, no orchestrator)

| Job | Notebooks | When to run |
|-----|-----------|-------------|
| `data_preparation` | 01, 02 | Once, or when documents change |
| `build_evaluate` | 03, 04 | Each code/prompt change (inner dev loop) |
| `monitoring` | 05, 06 | Scheduled every 6h (post-deploy validation) |

Deployment is a CLI/CI/CD step (`databricks bundle deploy`), not a DAB job. Jobs are run independently — there is no end-to-end orchestrator because the deploy step (CLI) breaks the chain.

## Component Details

### 1. Data Preparation (01, 02)
- Ingest documents into Delta tables
- Chunk and create embeddings
- Build Vector Search index with managed embeddings

### 2. Prompt Engineering (03)
- Register prompts in **MLflow Prompt Registry**
- Version prompts with commit messages
- Attach model configuration (temperature, max_tokens)
- Set `@production` alias — hot-reloads without redeployment

### 3. Agent Evaluation (04)
- Imports agent code **directly** from `agent_server/agent.py`
- Calls the same retrieval + LLM pipeline that runs in production
- **mlflow.genai.evaluate()** with `Correctness` scorer
- Quality gates: fails the pipeline if thresholds not met
- Additional quality checks (tone, safety, groundedness) run in production via External Monitor (NB06)

### 4. Deployment (CLI/CI-CD)
- `databricks bundle deploy -t prod` uploads code and config
- `databricks bundle run corp_chatbot_app -t prod` starts the app
- App resources (experiment, LLM endpoint, VS index) declared in `databricks.yml`
- App service principal gets auto-provisioned grants
- Git commit = version; rollback = redeploy previous commit

### 5. Smoke Tests & Monitoring (05, 06)
- **NB05:** Post-deploy smoke tests
  - Verifies app status via SDK, then calls agent code directly
  - Basic queries, edge cases, latency benchmarks
  - Uses direct function import (Apps with user auth require browser OAuth)
- **NB06:** Trace-based production monitoring
  - **MLflow External Monitor** for automated quality assessment
    - Built-in judges: safety, groundedness, relevance
    - Custom guideline judges: accuracy, professional tone
    - Configurable sampling rate
  - **Trace analytics** via `mlflow.search_traces()`
    - Query volume, latency (P50/P95/P99), error rates

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Agent Framework | MLflow AgentServer with async `@invoke`/`@stream` |
| LLM | Databricks Foundation Models (Claude Sonnet 4.5) |
| LLM Judge | Databricks Foundation Models (Claude Opus 4.1) |
| Retrieval | Databricks Vector Search (via `WorkspaceClient`) |
| Prompt Management | MLflow Prompt Registry |
| Evaluation | MLflow GenAI Evaluate (NB04 — direct import, no `log_model`) |
| Deployment | Databricks Apps via Asset Bundles |
| Monitoring | MLflow External Monitor + `mlflow.search_traces()` |
| Auth | App service principal (OAuth M2M) |
| Orchestration | Databricks Asset Bundles (DABs) + CI/CD |

## Agents on Apps: What to Know

Key considerations when deploying agents as Databricks Apps (vs. Model Serving Endpoints).

### Why Apps

- **Targets software engineers, not data scientists** — the strategic shift is from model development lifecycle (MDLC) to software development lifecycle (SDLC). Agents are software, not models.
- **AgentServer replaces the managed scoring server** — MLflow AgentServer is the Apps equivalent of the scoring server on Model Serving Endpoints.
- **Developer experience advantages** — fast iteration, local debugging, git versioning, async architecture, AI coding tool support (Claude Code, Cursor), TypeScript support.
- **Async architecture improves concurrency** — async code allows more concurrent requests per instance than sync model serving.
- **Apps platform is GA** — only the agent-specific integration layer (AgentServer) is Public Preview. The underlying Apps infrastructure is production-grade.
- **REST API via `DatabricksOpenAI` client** — agent apps are exposed as REST endpoints using the `app/` prefix.

### When to Use Model Serving Instead

- **Customer requires GA today** — Model Serving is the GA path for agents. No removal or deprecation is planned while Apps matures.
- **Need inference tables** — Apps don't produce inference tables. Use MLflow tracing (`mlflow.openai.autolog()`) instead.
- **Need AI Gateway at the agent level** — AI Gateway support is available at the underlying LLM endpoint layer, not at the agent/app level.

### Current Limitations to Be Aware Of

| Area | Limitation | Workaround |
|------|-----------|------------|
| **Scaling** | No scale-to-zero yet | App Spaces (MicroVM) is on the roadmap |
| **Scaling** | No auto-scaling of core count | Horizontal scaling uses fixed cores |
| **Timeouts** | HTTP proxy timeout is 120s (vs 297s on MSE) | Keep responses under 120s; use resume-stream patterns for long queries |
| **Observability** | No system metrics (CPU, memory) today | OTel-based observability stack is coming |
| **Observability** | No inference tables | Use `mlflow.search_traces()` for trace analytics |
| **Governance** | No UC lineage for agent apps | Git lineage (commit = version) is the alternative |
| **Governance** | No workspace permission model like MSE | Apps require OAuth token; use user authorization |
| **Integration** | Only AgentServer-based apps integrate with Playground, MAS, and OneChatBot | Fully custom apps miss these integrations |

### What's Coming

- **Horizontal scaling + zero-downtime deployments** — both in progress
- **App Spaces** — MicroVM-based, with scale-to-zero support
- **OTel observability + Insights Dashboard** — out-of-the-box metrics, logs, and admin dashboard
- **MLflow traces in Unity Catalog** — governed trace data in Delta tables instead of MLflow Experiment
- **AppKit** — common development patterns SDK for Apps
- **Git-backed apps** — direct git integration for deployments
- **Neon Auth** — external auth backend-as-a-service for apps

### Key Resources

- [MLflow AgentServer Documentation](https://mlflow.org/docs/latest/genai/agent-server.html)
- [Databricks App Templates Repository](https://github.com/databricks/app-templates)
- [Databricks Apps Documentation](https://docs.databricks.com/en/apps/index.html)
- [Author agents in Apps](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/author-agent)
- [Migrate agents from Model Serving to Apps](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/migrate-agent-to-apps)

## Key Design Decisions

1. **Direct evaluation over `log_model()`**: Import agent code directly instead of logging/loading a model — faster feedback, no serialization overhead
2. **Prompt Registry with hot-reload**: Prompt changes take effect without redeployment
3. **`WorkspaceClient` over `VectorSearchClient`**: Natively handles Apps OAuth M2M authentication
4. **Deployment as CI/CD, not a job**: Avoids circular dependency (job deploying the bundle that defines the job)
5. **Trace-based monitoring over inference tables**: Apps don't produce inference tables; MLflow traces via `autolog()` are the primary signal
6. **Independent jobs, no orchestrator**: Deployment is CLI/CI/CD, breaking the linear chain; each job runs on its own schedule

## Comparison with Model Serving Architecture

| Aspect | Model Serving (previous) | Apps (current) |
|--------|-------------------------|----------------|
| Deployment artifact | MLflow model version | Source code (git) |
| Versioning | UC model versions + aliases | Git commits/tags |
| Rollback | Revert `@champion` alias | Redeploy previous commit |
| Agent build step | `log_model()` → register → `@candidate` | Edit code directly |
| Evaluation | Load logged model, evaluate | Import agent functions, evaluate |
| Deployment | `agents.deploy(model, version)` | `databricks bundle deploy` |
| Auth (Vector Search) | OBO via `CredentialStrategy` | `WorkspaceClient()` (OAuth M2M) |
| Monitoring data source | Inference tables (automatic) | MLflow traces (via autolog) |
| Pipeline stages | 4 chained jobs (data, build, deploy, monitor) | 3 independent jobs + CLI deploy |
| Notebooks | 01-09 (all used) | 01-06 (contiguous) |
| Config management | `ModelConfig` baked into model | Env vars in `app.yaml` |
