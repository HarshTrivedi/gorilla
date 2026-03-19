#!/bin/bash
# Gantry-based alternative to serve_vllm_model.sh.
# Launches a vLLM inference server on Beaker using the public vllm Docker image.
# Usage: set the same env vars as serve_vllm_model.sh, then run this script.

# Gantry must be run from the git repo root (BFCLv4/), not this subdirectory.
cd "$(dirname "$0")/.."

if [ -z "$MODEL_NAME" ]; then
  echo "MODEL_NAME environment variable is not set. Using default: allenai/general-tool-use-dev."
  MODEL_NAME="allenai/general-tool-use-dev"
fi

if [ -z "$MODEL_REVISION" ]; then
  echo "Error: MODEL_REVISION environment variable is not set."
  exit 1
fi

if [ -z "$EXTRA_ARGS" ]; then
  echo "EXTRA_ARGS environment variable is not set. Using default: ''"
  EXTRA_ARGS=""
fi

SERVE_CMD="vllm serve $MODEL_NAME --revision $MODEL_REVISION --tensor-parallel-size 1 --trust-remote-code --dtype bfloat16 --gpu-memory-utilization 0.9 $EXTRA_ARGS"

gantry run \
  --workspace ai2/general-tool-use \
  --budget ai2/oe-adapt \
  --description "vLLM Server for $MODEL_NAME: $MODEL_REVISION" \
  --cluster ai2/saturn-cirrascale \
  --gpus 1 \
  --priority high \
  --not-preemptible \
  --docker-image vllm/vllm-openai:v0.16.0 \
  --no-python \
  --host-networking \
  --env-secret "HF_TOKEN=HF_TOKEN" \
  "$@" -- bash -c "$SERVE_CMD"
