# Architecture Overview

## Apps LLMOps Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Data Prep   │───>│  Develop &  │───>│   Deploy    │───>│  Monitor    │
│  (run once)  │    │  Validate   │    │  (CI/CD)    │    │  (ongoing)  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
  Notebooks:         Notebooks:         CI/CD pipeline:    Notebook:
  01, 02             03, 04             dev → prod         05
                                                                ↓
                                                     ┌──────────────────┐
                                                     │ Feedback Loop:   │
                                                     │ traces → eval    │
                                                     │ dataset (Delta)  │
                                                     └────────┬─────────┘
                                                              ↓
                                                     Back to NB04 as
                                                     regression test cases
```

### Environments

| Environment | Target | Mode | Purpose |
|------------|--------|------|---------|
| **dev** | `-t dev` | `development` | Personal sandbox, rapid iteration, CI eval on merge |
| **prod** | `-t prod` | `production` | Production, promoted after eval passes + manual approval |

Each environment has its own app, experiment, and schema — completely isolated.

### DAB Jobs

| Job | Notebooks | When to run |
|-----|-----------|-------------|
| `data_preparation` | 01, 02 | Once, or when documents change |
| `build_evaluate` | 03, 04 | Each code/prompt change (inner dev loop) |
| `monitoring` | 05 | Scheduled every 6h (health check + alerting + feedback loop) |

Deployment is a CI/CD step (see `.github/workflows/deploy.yml`), not a DAB job. CI evaluates in dev, then promotes to prod with manual approval.

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

### 3. Agent Evaluation (04) — Three-Tier Strategy
- **Tier 1 (local, seconds)**: `uv run pytest` — unit tests with mocked dependencies
- **Tier 2 (CI, minutes)**: NB04 — evaluates against real endpoints with quality gate
  - Imports agent code **directly** from `agent_server/agent.py`
  - Static eval dataset + **production regression cases** from `{catalog}.{schema}.eval_dataset`
  - **mlflow.genai.evaluate()** with `Correctness` scorer
  - Quality gates: fails the pipeline if thresholds not met
- **Tier 3 (production, continuous)**: External Monitor judges (NB05)
  - Feeds back low-quality traces as regression test cases (closes the loop)

### 4. Deployment (CI/CD Pipeline)
- **CI/CD workflow** (`.github/workflows/deploy.yml`) automates the full pipeline:
  - PR: validate bundle only (fast, no cost)
  - Merge to main: validate → evaluate (dev) → deploy prod (manual approval)
  - Prod promotion requires manual approval via GitHub Environment protection rules
- App resources (experiment, LLM endpoint, VS index) declared in `databricks.yml`
- App service principal gets auto-provisioned grants
- Git commit = version; rollback = `scripts/rollback.sh` or `git revert` + CI re-run

### 5. Production Monitoring (05)
- **App health check**: Verifies app status via SDK before analyzing traces
- **Part A — MLflow External Monitor**: Automated quality assessment
  - Built-in judges: safety, groundedness, relevance
  - Custom guideline judges: accuracy, professional tone
  - Configurable sampling rate
- **Part B — Trace analytics** via `mlflow.search_traces()`
  - Query volume, latency (P50/P95/P99), error rates
  - **Threshold-based alerting**: error rate > 5%, P95 latency > 30s → alerts written to `{catalog}.{schema}.monitoring_alerts` Delta table
- **Part C — Production feedback loop**: Exports error traces to `{catalog}.{schema}.eval_dataset` Delta table, consumed by NB04 as regression test cases

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

## Deployment Version Tracking

Apps-based agents use git as the versioning mechanism instead of UC model versions.
`start_server.py` calls `setup_mlflow_git_based_version_tracking()` at startup,
which records the current git SHA in each MLflow trace. This provides:

- **Trace-to-commit mapping**: Every trace is tagged with the exact git commit that produced it, enabling precise debugging and rollback decisions.
- **Prompt version correlation**: Combined with MLflow Prompt Registry version metadata, you can see which code + prompt combination generated any given response.
- **No extra infrastructure**: Works automatically — no CI metadata injection or custom tagging required.

The git SHA is visible in the MLflow Experiment UI under the trace's system tags.

## Key Design Decisions

1. **Direct evaluation over `log_model()`**: Import agent code directly instead of logging/loading a model — faster feedback, no serialization overhead
2. **Prompt Registry with hot-reload**: Prompt changes take effect without redeployment
3. **`WorkspaceClient` over `VectorSearchClient`**: Natively handles Apps OAuth M2M authentication
4. **Deployment as CI/CD, not a job**: Avoids circular dependency (job deploying the bundle that defines the job)
5. **Trace-based monitoring over inference tables**: Apps don't produce inference tables; MLflow traces via `autolog()` are the primary signal
6. **Two environments (dev/prod)**: Dev validates before production, each with isolated resources
7. **Production feedback loop**: Error traces automatically become regression test cases — eval dataset grows with real usage
8. **Threshold-based alerting to Delta**: Monitoring alerts stored in a Delta table consumable by DBSQL Alerts or Notification Destinations

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
| Pipeline stages | 4 chained jobs (data, build, deploy, monitor) | 3 independent jobs + CI/CD deploy |
| Notebooks | 01-09 (all used) | 01-05 (contiguous) |
| Config management | `ModelConfig` baked into model | Env vars in `app.yaml` |
| Environments | Single target | dev / prod (isolated) |
| Deployment tracking | Model version in UC | Git history + `databricks apps get` |
| Eval dataset | Static | Static + production regression feedback |
| Alerting | None | Threshold-based alerts to Delta table |
| Rollback mechanism | Revert model alias | `scripts/rollback.sh` (code) or alias change (prompts) |
