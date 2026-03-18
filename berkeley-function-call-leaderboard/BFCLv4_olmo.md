# BFCLv4 Evaluation Guide

## Beaker Evals

Build the Docker image (or find an existing one on the workspace):
```bash
sh ./build_image.sh
```

### External vllm server (BaseHandler)

**Step 1: Launch the server**
```bash
MODEL_NAME=allenai/Olmo-3-7B-Instruct \
MODEL_REVISION=main \
EXTRA_ARGS="--enable-auto-tool-choice --tool-call-parser olmo3" \
./serve_vllm_model.sh
```

**Step 2: Get the server IP**

Retrieve the node name from the Beaker job and the associated IP address from TailScale VPN.

**Step 3: Launch the eval**
```bash
MODEL_REVISION=main \
MODEL_NAME=allenai/Olmo-3-7B-Instruct \
NUM_THREADS=25 \
TEST_CATEGORY="single_turn,multi_turn,memory" \
OPENAI_BASE_URL="http://100.79.74.83:8000/v1/" \
./evaluate_model.sh
```

### Internal vllm server (OSSHandler)

For OLMo3 (note the `-local` **suffix** in the model name):
```bash
MODEL_REVISION=main \
MODEL_NAME=allenai/Olmo-3-7B-Instruct-local \
NUM_THREADS=100 \
TEST_CATEGORY="single_turn,multi_turn,memory" \
EXTRA_VLLM_ARGS="--gpu-memory-utilization 0.9 --enable-auto-tool-choice --tool-call-parser olmo3" \
./evaluate_model.sh
```

---

## Local Experimentation

Follow the `README.md` to set up and activate the conda environment before running the commands below. Install via `pyproject.toml`, **not** the `bfcl_eval` package from PyPI.

### Third-party API services (e.g. OpenAI)

Make sure `OPENAI_API_KEY` and `OPENAI_BASE_URL` are set in `.env` or exported as environment variables.

```bash
bfcl generate --model gpt-5-nano-2025-08-07-FC && bfcl evaluate --model gpt-5-nano-2025-08-07-FC
```

To run a partial eval with fixed run-ids (useful for debugging):

```bash
bfcl generate --model gpt-5-nano-2025-08-07-FC --test-category simple_python --run-ids \
  && bfcl evaluate --model gpt-5-nano-2025-08-07-FC --test-category simple_python --partial-eval
```

### External vllm server (BaseHandler)

**Step 1: Start the vllm server**

For OLMo3:
```bash
vllm serve allenai/Olmo-3-7B-Instruct \
  --gpu-memory-utilization 0.9 \
  --enable-auto-tool-choice \
  --tool-call-parser olmo3
```

For Qwen3:

> **Note:** For Qwen models, the preferred method is the OSSHandler (see below), which starts the vllm server internally. The BaseHandler path for Qwen is untested.

```bash
# Non-thinking mode
vllm serve Qwen/Qwen3-0.6B \
  --chat-template ./qwen3_nonthinking.jinja \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

```bash
# Reasoning mode
vllm serve Qwen/Qwen3-0.6B \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

**Step 2: Point the client at the server**

Set the server endpoint in `.env` or export it as an environment variable:

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
```

**Step 3: Run the eval**

For OLMo3:
```bash
bfcl generate --model allenai/Olmo-3-7B-Instruct \
  --backend vllm \
  --skip-server-setup \
  && bfcl evaluate --model allenai/Olmo-3-7B-Instruct
```

For Qwen3:
```bash
# Omit --test-category to evaluate all categories (slower)
bfcl generate --model Qwen/Qwen3-0.6B-FC \
  --test-category simple_python \
  --backend vllm \
  --skip-server-setup \
  && bfcl evaluate --model Qwen/Qwen3-0.6B-FC
```

### Internal vllm server (OSSHandler)

The OSSHandler starts a vllm server internally as part of the same process.

For Qwen3:
```bash
bfcl generate --model Qwen/Qwen3-0.6B-FC --backend vllm \
  && bfcl evaluate --model Qwen/Qwen3-0.6B-FC
```

For OLMo3 (note the `-local` **suffix** in the model name):
```bash
bfcl generate --model allenai/Olmo-3-7B-Instruct-local \
  --backend vllm \
  --extra-vllm-args "--gpu-memory-utilization 0.9 --enable-auto-tool-choice --tool-call-parser olmo3" \
  && bfcl evaluate --model allenai/Olmo-3-7B-Instruct-local
```
