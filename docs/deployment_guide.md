# Deployment Guide: RAG LLMOps Demo (Databricks Apps)

## Apps LLMOps Architecture

In the Apps deployment model, the **source code is the deployment artifact**
(not an MLflow model version). This changes the pipeline fundamentally:

| Aspect | Model Serving (old) | Apps (current) |
|--------|--------------------|--------------------|
| Deployment artifact | MLflow model version | Source code (git) |
| Versioning | UC model versions + aliases | Git commits/tags |
| Rollback | Revert `@champion` alias | Redeploy previous commit |
| Evaluation gates | Model version N passes -> deploy | Code at commit X passes -> deploy |
| Config changes | Baked into model via `ModelConfig` | Env vars in `app.yaml` |
| Prompt changes | Hot-reload via Prompt Registry | Hot-reload via Prompt Registry |

## Pipeline Overview

```
┌──────────────────────────────────────────────────────────┐
│                 Apps LLMOps Lifecycle                     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Stage 1: DATA INFRASTRUCTURE (run once)                 │
│    01_data_ingestion -> 02_vector_index_creation          │
│                                                          │
│  Stage 2: DEVELOP & VALIDATE (inner loop)                │
│    03_prompt_engineering (hot-reload, no redeploy)        │
│    agent_server/agent.py (edit locally)                   │
│    04_agent_evaluation (quality gate)                     │
│    ↻ iterate until quality gate passes                    │
│                                                          │
│  Stage 3: DEPLOY (CI/CD or CLI)                          │
│    databricks bundle deploy -t prod                      │
│    databricks bundle run corp_chatbot_app -t prod        │
│                                                          │
│  Stage 4: MONITOR & VALIDATE (ongoing, scheduled)        │
│    05_inference_testing (smoke tests)                     │
│    06_monitoring_dashboard (trace-based monitoring)       │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### DAB Jobs (independent, no orchestrator)

| Job | Notebooks | When to run |
|-----|-----------|-------------|
| `data_preparation` | 01, 02 | Once, or when documents change |
| `build_evaluate` | 03, 04 | Each code/prompt change (inner dev loop) |
| `monitoring` | 05, 06 | Scheduled every 6h (post-deploy validation) |

**Deployment is a CLI/CI/CD step**, not a DAB job. Jobs run independently — there is no end-to-end orchestrator because the deploy step (CLI) breaks the chain.

## First-Time Deployment (4 Phases)

### Phase 1: Deploy infrastructure (VS index grant commented out)

```bash
# Comment out the vector-search-index resource in databricks.yml first
databricks bundle deploy -t prod
```

Creates UC schema, MLflow experiment, App resource, and all job definitions.

### Phase 2: Create data infrastructure

```bash
databricks bundle run data_preparation -t prod
```

Runs NB01 + NB02: ingests documents, creates Vector Search endpoint and index.
Wait for completion (~5-10 minutes).

### Phase 3: Redeploy with full grants

Uncomment the `vector-search-index` resource in `databricks.yml`, then:

```bash
databricks bundle deploy -t prod
```

### Phase 4: Evaluate, deploy app, and monitor

```bash
# Evaluate agent quality
databricks bundle run build_evaluate -t prod

# Start the app
databricks bundle run corp_chatbot_app -t prod

# Run smoke tests + set up monitoring
databricks bundle run monitoring -t prod
```

## Subsequent Deployments

Once all infrastructure exists, the workflow is:

```bash
# 1. Evaluate quality (runs prompts + evaluation)
databricks bundle run build_evaluate -t prod

# 2. Deploy code changes + restart app
databricks bundle deploy -t prod
databricks bundle run corp_chatbot_app -t prod

# 3. Validate + monitor
databricks bundle run monitoring -t prod
```

Or run all jobs sequentially:

```bash
databricks bundle deploy -t prod && \
  databricks bundle run data_preparation -t prod && \
  databricks bundle run build_evaluate -t prod && \
  databricks bundle run monitoring -t prod
```

## CI/CD Pipeline

For automated deployments, use the following GitHub Actions workflow:

```yaml
# .github/workflows/deploy.yml
name: Deploy RAG Agent

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
  DATABRICKS_CLIENT_ID: ${{ secrets.DATABRICKS_SP_CLIENT_ID }}
  DATABRICKS_CLIENT_SECRET: ${{ secrets.DATABRICKS_SP_CLIENT_SECRET }}
  TARGET: prod

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: databricks/setup-cli@main
      - run: databricks bundle validate -t ${{ env.TARGET }}

  evaluate:
    needs: validate
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: databricks/setup-cli@main
      - name: Deploy bundle (infra + jobs)
        run: databricks bundle deploy -t ${{ env.TARGET }}
      - name: Run evaluation pipeline
        run: databricks bundle run build_evaluate -t ${{ env.TARGET }}
        # NB04 enforces quality gates -- job fails if thresholds not met

  deploy:
    needs: evaluate
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: databricks/setup-cli@main
      - name: Deploy and start app
        run: |
          databricks bundle deploy -t ${{ env.TARGET }}
          databricks bundle run corp_chatbot_app -t ${{ env.TARGET }}

  smoke-test:
    needs: deploy
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: databricks/setup-cli@main
      - name: Run smoke tests + monitoring
        run: databricks bundle run monitoring -t ${{ env.TARGET }}
```

### Key CI/CD Patterns

- **PR builds**: Run `validate` only (fast, no cost)
- **Main branch**: Full pipeline (validate -> evaluate -> deploy -> smoke test)
- **Quality gate**: NB04 raises an exception if thresholds fail, blocking deployment
- **Rollback**: Revert the git commit and re-run the pipeline
- **Prompt-only changes**: Update prompt in NB03, run `build_evaluate` -- no
  redeploy needed since prompts hot-reload via Prompt Registry

## Migration from Model Serving: Key Considerations

If you are migrating an existing agent from Model Serving Endpoints (MSE) to Apps,
keep these points in mind:

### Code Changes Required

1. **Agent class to async functions**: Replace `ResponsesAgent.predict_stream()` with `@invoke()` and `@stream()` decorated async functions.
2. **Authentication**: Replace OBO/`CredentialStrategy` patterns with `WorkspaceClient()` — it natively handles Apps OAuth M2M.
3. **Vector Search client**: Switch from `VectorSearchClient()` (needs `model-serving-user-context` header) to `w.vector_search_indexes.query_index()` (no special auth handling needed).
4. **Model config**: Replace `ModelConfig` (baked into logged model) with environment variables in `app.yaml`.
5. **Inference table monitoring**: Replace inference table queries with `mlflow.search_traces()` — Apps don't produce inference tables.

### What Stays the Same

- `ResponsesAgent` schema (input/output format) — works across both deployment modes
- MLflow Prompt Registry — same SDK, same hot-reload behavior
- Evaluation with `mlflow.genai.evaluate()` — same scorers, same dataset format
- Unity Catalog for governance — schemas, catalogs, volumes all work the same

### Architecture Patterns

**AgentServer is required for full platform integration.** Only AgentServer-based apps integrate with:
- Playground (test UI in the workspace)
- Multi-Agent Supervisor (MAS)
- OneChatBot and other first-party consumers

If you build a fully custom FastAPI app (without AgentServer), you get full flexibility but lose these integrations. Choose based on your needs:

| Pattern | Integration Level | Flexibility |
|---------|------------------|-------------|
| AgentServer + `@invoke`/`@stream` | Full (Playground, MAS, etc.) | Agent logic only |
| Custom FastAPI + your own routes | None with Databricks tools | Full control |
| Hybrid (AgentServer + custom routes) | Full + custom endpoints | Best of both |

### Timeout Considerations

- Apps HTTP proxy has a **120-second timeout** (vs. 297s on MSE)
- For long-running queries, implement resume-stream patterns or increase concurrency
- Async architecture helps: more concurrent requests per instance than sync MSE

## Target-Specific Notes

### Dev (`-t dev`)

- `mode: development` prefixes resource names with `[dev <username>]`
- Schema becomes `dev_<username>_corp_affairs`
- App name stays as-is (not prefixed)
- Safe to redeploy frequently

### Prod (`-t prod`)

- `mode: production` requires `workspace.root_path`
- No name prefixing -- resources use exact names
- App name is the production name visible to users
- Deploy carefully, test in dev first

## Troubleshooting

See [troubleshooting_guide.md](troubleshooting_guide.md) for all known issues.

Common deployment errors:
- **TABLE_DOES_NOT_EXIST** on VS index: Run data_preparation first (Phase 2)
- **QUOTA_EXCEEDED** on OAuth: Delete unused apps to free quota
- **root_path required**: Add `root_path` to prod target in `databricks.yml`
