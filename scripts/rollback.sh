#!/bin/bash
#
# Rollback a Databricks Apps agent to a previous git commit.
#
# Usage:
#   ./scripts/rollback.sh <target> <commit-sha>
#   ./scripts/rollback.sh prod abc1234
#   ./scripts/rollback.sh dev HEAD~1
#
# For prompt-only rollback (no redeploy needed):
#   mlflow prompts set-alias <catalog>.<schema>.rag_prompt production <version>

set -euo pipefail

TARGET="${1:?Usage: $0 <target> <commit-sha>}"
COMMIT="${2:?Usage: $0 <target> <commit-sha>}"

ORIGINAL_BRANCH=$(git rev-parse --abbrev-ref HEAD)
ORIGINAL_COMMIT=$(git rev-parse HEAD)

echo "=== Rollback ==="
echo "  Target:          $TARGET"
echo "  Rolling back to: $COMMIT"
echo "  Current commit:  $ORIGINAL_COMMIT"
echo ""

git checkout "$COMMIT"

echo "Deploying bundle..."
databricks bundle deploy -t "$TARGET"

echo "Starting app..."
databricks bundle run corp_chatbot_app -t "$TARGET"

echo "Running smoke tests..."
databricks bundle run monitoring -t "$TARGET" || echo "WARNING: Smoke tests failed after rollback"

echo ""
echo "=== Rollback complete ==="
echo "  Deployed commit: $(git rev-parse HEAD)"
echo "  Target:          $TARGET"
echo ""
echo "You are now in detached HEAD state."
echo "To return to your branch: git checkout $ORIGINAL_BRANCH"
