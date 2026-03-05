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

### DAB Jobs

| Job | Notebooks | When to run |
|-----|-----------|-------------|
| `data_preparation` | 01, 02 | Once, or when documents change |
| `build_evaluate` | 03, 04 | Each code/prompt change (inner dev loop) |
| `monitoring` | 05, 06 | Scheduled every 6h (post-deploy validation + alerting) |
| `deployment_manifest` | 07 | After each deploy (tracks git SHA, prompt version) |

**Deployment is a CI/CD step** (see `.github/workflows/deploy.yml`). The pipeline evaluates in dev, then promotes to prod with manual approval.

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

The actual GitHub Actions workflow is at `.github/workflows/deploy.yml`. It
implements a dev-validated promotion model:

```
PR opened        →  validate (dev + prod bundles)
Merge to main    →  validate → evaluate (dev) → deploy prod (manual approval)
```

### Pipeline Flow

| Stage | Job | Target | What Happens |
|-------|-----|--------|-------------|
| 1 | `validate` | dev + prod | Bundle validation (fast, no cost) |
| 2 | `evaluate` | dev | Deploy infra, run `build_evaluate` (quality gate) |
| 3 | `deploy-prod` | prod | **Manual approval**, then deploy + start app + smoke test + manifest |

### Key CI/CD Patterns

- **PR builds**: Run `validate` only (fast, no cost)
- **Main branch**: Evaluate in dev, then promote to prod
- **Prod promotion**: Requires manual approval via GitHub Environment protection rules
- **Quality gate**: NB04 raises an exception if thresholds fail, blocking the pipeline
- **Deployment manifest**: NB07 logs git SHA, prompt version, and eval run ID to MLflow after each deploy
- **Rollback**: `scripts/rollback.sh <target> <commit-sha>` or `git revert` + CI re-run
- **Prompt-only changes**: Update prompt in NB03, run `build_evaluate` — no
  redeploy needed since prompts hot-reload via Prompt Registry

### GitHub Secrets Required

| Secret | Description |
|--------|-------------|
| `DATABRICKS_HOST` | Workspace URL |
| `DATABRICKS_SP_CLIENT_ID` | Service principal client ID |
| `DATABRICKS_SP_CLIENT_SECRET` | Service principal secret |

## Rollback Procedures

### Code Rollback

Use the provided script to roll back to any previous git commit:

```bash
./scripts/rollback.sh prod abc1234
./scripts/rollback.sh dev HEAD~1
```

This checks out the commit, runs `databricks bundle deploy`, starts the app, and
runs smoke tests. You end up in detached HEAD state — `git checkout main` to return.

Alternatively, `git revert <commit>` and push to trigger CI/CD re-deployment.

### Prompt Rollback

Prompt changes are independent of code deploys. To roll back a prompt:

```bash
# List prompt versions
mlflow prompts list-versions <catalog>.<schema>.rag_prompt

# Set @production alias to a previous version (takes effect immediately)
mlflow prompts set-alias <catalog>.<schema>.rag_prompt production <version>
```

No app restart needed — the agent loads the prompt by alias on each request.

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
- Use for local iteration and experimentation

### Prod (`-t prod`)

- `mode: production` requires `workspace.root_path`
- No name prefixing — resources use exact names
- App name is the production name visible to users
- Only deployed after dev evaluation + manual approval

## Troubleshooting

See [troubleshooting_guide.md](troubleshooting_guide.md) for all known issues.

Common deployment errors:
- **TABLE_DOES_NOT_EXIST** on VS index: Run data_preparation first (Phase 2)
- **QUOTA_EXCEEDED** on OAuth: Delete unused apps to free quota
- **root_path required**: Add `root_path` to prod target in `databricks.yml`
