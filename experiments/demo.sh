#!/usr/bin/env bash
# same unstable config 3 ways: nothing / rule table / llm agent. same seed so same batches
# usage: OLLAMA_HOST=http://yourbox:11434 ./experiments/demo.sh [ollama|gemini]
set -euo pipefail
POLICY=${1:-ollama}
ARGS="--lr 3e-2 --warmup 100 --clip 0 --steps 600"

uv run gradwarden train $ARGS --run-dir runs/demo-none
uv run gradwarden watch $ARGS --policy rules --run-dir runs/demo-rules
uv run gradwarden watch $ARGS --policy "$POLICY" --run-dir "runs/demo-$POLICY"
