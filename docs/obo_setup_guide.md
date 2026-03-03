# On-Behalf-Of (OBO) Setup Guide

## What is OBO?

On-Behalf-Of (OBO) authentication allows your deployed agent to make API calls
**as the end user** rather than as the service principal. This enables:

- **Per-user data isolation**: Vector Search queries respect the user's Unity Catalog permissions
- **Audit trail**: All actions are logged under the user's identity
- **Compliance**: Row-level security and column masking apply per user

## Architecture

```
End User -> Databricks App (OBO proxy) -> Agent Endpoint
                |                              |
                |-- User's OAuth token ------->|
                |                              |-- Vector Search (as user)
                |                              |-- LLM endpoint (as service)
```

## Two Levels of OBO

### Level 1: Agent-Level OBO (agents.deploy) - Built-In

This is the mode used in notebook `07_endpoint_deployment.py`. The agent uses
`CredentialStrategy.MODEL_SERVING_USER_CREDENTIALS` at query time so that
Vector Search queries run as the calling user.

```python
# In rag_agent.py (_retrieve_context method):
from databricks.vector_search.client import VectorSearchClient, CredentialStrategy

try:
    # OBO: use the calling user's credentials when running in Model Serving
    vsc = VectorSearchClient(
        credential_strategy=CredentialStrategy.MODEL_SERVING_USER_CREDENTIALS,
        disable_notice=True,
    )
except Exception:
    # Fallback for local dev / notebook testing (no user credentials available)
    vsc = VectorSearchClient(disable_notice=True)
```

- Vector Search queries respect the **user's** Unity Catalog permissions
- Falls back to default credentials when running locally or in tests
- No additional infrastructure needed

### Level 2: Full OBO via Databricks Apps

For end-to-end user identity propagation (e.g., a custom web UI), deploy as
a **Databricks App** that forwards the user's OAuth token to the serving endpoint.

#### Step 1: Create the App

```python
# app.py - FastAPI app with OBO support
import os
from fastapi import FastAPI, Request
from databricks.sdk import WorkspaceClient
from databricks.sdk.config import Config

app = FastAPI()

@app.post("/chat")
async def chat(request: Request):
    # Get the user's token from the forwarded header
    user_token = request.headers.get("x-forwarded-access-token")

    if user_token:
        # Create a workspace client authenticated as the user
        config = Config(
            host=os.environ["DATABRICKS_HOST"],
            token=user_token,
        )
        user_client = WorkspaceClient(config=config)
    else:
        # Fallback to service principal
        user_client = WorkspaceClient()

    # Query the ResponsesAgent endpoint on behalf of the user
    # Note: ResponsesAgent uses "input" (not "messages") and "output" (not "choices")
    body = await request.json()
    endpoint = os.environ["MODEL_SERVING_ENDPOINT"]
    result = user_client.api_client.do(
        "POST",
        f"/serving-endpoints/{endpoint}/invocations",
        body={"input": body.get("input", [{"role": "user", "content": body.get("question", "")}])},
    )

    return result
```

#### Step 2: Configure app.yml

```yaml
command:
  - uvicorn
  - app:app
  - --host=0.0.0.0
  - --port=8000

permissions:
  - permission: CAN_USE
    group_name: users

env:
  - name: DATABRICKS_HOST
    value: "${workspace.host}"
  - name: MODEL_SERVING_ENDPOINT
    value: "agents_<catalog>-<schema>-<model>"  # Replace with your actual endpoint name
```

#### Step 3: Deploy with DAB

Add to `databricks.yml`:

```yaml
resources:
  apps:
    chatbot_app:
      name: "corp-chatbot"
      source_code_path: ./app
      config:
        command:
          - uvicorn
          - app:app
          - --host=0.0.0.0
          - --port=8000
      permissions:
        - user_name: users
          level: CAN_USE
```

#### Step 4: Verify OBO

```python
# Test that user identity is passed through
import requests

response = requests.post(
    "https://<app-url>/chat",
    json={"input": [{"role": "user", "content": "What is the remote work policy?"}]},
    headers={"Authorization": f"Bearer {user_token}"},
)
```

## Security Considerations

1. **Token Validation**: Always validate the `x-forwarded-access-token` before using it
2. **Scopes**: The OBO token has the same permissions as the user
3. **Audit Logging**: All API calls made via OBO are logged under the user's identity
4. **Token Expiry**: OBO tokens have the same expiry as the user's session token

## When to Use Each Level

| Scenario | Recommended Level |
|----------|-----------------|
| All users see the same data | Level 1: `agents.deploy()` with `CredentialStrategy` |
| Per-user UC permissions on Vector Search | Level 1: `agents.deploy()` with `CredentialStrategy` |
| Custom web UI with user identity propagation | Level 2: Databricks App |
| Regulatory compliance requiring per-user audit trails | Level 2: Databricks App |
| Quick POC / demo | Level 1: `agents.deploy()` (simplest) |
