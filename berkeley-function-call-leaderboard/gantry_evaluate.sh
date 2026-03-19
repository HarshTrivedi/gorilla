#!/bin/bash
# Gantry-based alternative to evaluate_model.sh.
# Uses the current git commit (must be pushed) instead of a pre-built Docker image.
# Usage: set the same env vars as evaluate_model.sh, then run this script.

# Gantry must be run from the git repo root (BFCLv4/), not this subdirectory.
cd "$(dirname "$0")/.."

if [ -z "$MODEL_REVISION" ]; then
  echo "MODEL_REVISION environment variable is not set. This is important to track the model you're evaluating!"
  exit 1
fi

if [ -z "$MODEL_NAME" ]; then
  echo "MODEL_NAME environment variable is not set. Defaulting to 'allenai/general-tool-use-dev'."
  MODEL_NAME="allenai/general-tool-use-dev"
fi

if [ -z "$NUM_THREADS" ]; then
  echo "NUM_THREADS environment variable is not set. Using default: 1."
  NUM_THREADS=1
fi

if [ -z "$TEST_CATEGORY" ]; then
  echo "TEST_CATEGORY environment variable is not set. Using default: all categories."
  TEST_CATEGORY="all"
fi

if [ -z "$USE_THINKING" ]; then
  echo "USE_THINKING environment variable is not set. Using default: 0"
  USE_THINKING=0
fi

echo "Configuration:"
echo "  NUM_THREADS: $NUM_THREADS"
echo "  TEST_CATEGORY: $TEST_CATEGORY"
echo "  MODEL_NAME: $MODEL_NAME"
echo "  LOCAL_MODEL_PATH: $LOCAL_MODEL_PATH"
echo "  MODEL_REVISION: $MODEL_REVISION"
echo "  MAX_TOKENS: $MAX_TOKENS"
echo "  USE_THINKING: $USE_THINKING"
echo "  OPENAI_BASE_URL: $OPENAI_BASE_URL"
echo "  EXTRA_VLLM_ARGS: $EXTRA_VLLM_ARGS"

# Build the bfcl generate command
GENERATE_CMD="bfcl generate --model \$MODEL_NAME --test-category \$TEST_CATEGORY --allow-overwrite --num-threads \$NUM_THREADS --temperature 0 --backend vllm"

if [ -n "$LOCAL_MODEL_PATH" ]; then
  GENERATE_CMD+=" --local-model-path \$LOCAL_MODEL_PATH"
fi

if [ -n "$EXTRA_VLLM_ARGS" ]; then
  GENERATE_CMD+=" --extra-vllm-args \"$EXTRA_VLLM_ARGS\""
fi

EVAL_CMD="bfcl evaluate --model \$MODEL_NAME --test-category \$TEST_CATEGORY"

FULL_CMD="cd berkeley-function-call-leaderboard && $GENERATE_CMD && $EVAL_CMD && python convert_bfcl_scores_to_beaker_metrics.py --overall_csv score/data_overall.csv && cp -r score /results/"

# Build gantry args
GANTRY_ARGS=(
  --description "BFCLv4 Evaluation ($TEST_CATEGORY) for $MODEL_NAME: $MODEL_REVISION"
  --cluster ai2/saturn
  --gpus 1
  --priority high
  --not-preemptible
  --docker-image ghcr.io/allenai/pytorch:2.5.1-cuda12.4-python3.11
  --install "pip install -e berkeley-function-call-leaderboard/"
  --env "MODEL_NAME=$MODEL_NAME"
  --env "MODEL_REVISION=$MODEL_REVISION"
  --env "NUM_THREADS=$NUM_THREADS"
  --env "TEST_CATEGORY=$TEST_CATEGORY"
  --env "USE_THINKING=$USE_THINKING"
  --env-secret "HF_TOKEN=HF_TOKEN"
  --env-secret "OPENAI_API_KEY=ljm_OPENAI_API_KEY"
  --env-secret "SERPAPI_API_KEY=SG_PERSONAL_SERPAPI_API_KEY"
)

if [ -n "$LOCAL_MODEL_PATH" ]; then
  GANTRY_ARGS+=(--weka "oe-adapt-default:/weka/oe-adapt-default")
  GANTRY_ARGS+=(--weka "oe-training-default:/weka/oe-training-default")
  GANTRY_ARGS+=(--env "LOCAL_MODEL_PATH=$LOCAL_MODEL_PATH")
fi

if [ -n "$MAX_TOKENS" ]; then
  GANTRY_ARGS+=(--env "MAX_TOKENS=$MAX_TOKENS")
fi

if [ -n "$OPENAI_BASE_URL" ]; then
  GANTRY_ARGS+=(--env "OPENAI_BASE_URL=$OPENAI_BASE_URL")
fi

gantry run "${GANTRY_ARGS[@]}" -- bash -c "$FULL_CMD"
