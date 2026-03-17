# BFCLv4 Evaluation Guide

## Quick Start

### Qwen3 (Local FC OSS mode -- starts local vllm server internally)

```bash
bfcl generate --model Qwen/Qwen3-0.6B-FC --test-category simple_python --backend vllm && bfcl evaluate --model Qwen/Qwen3-0.6B-FC --test-category simple_python
```

### GPT (partial eval with run IDs)

```bash
bfcl generate --model gpt-5-nano-2025-08-07-FC --test-category simple_python --run-ids && bfcl evaluate --model gpt-5-nano-2025-08-07-FC --test-category simple_python --partial-eval
```

---

### Qwen3 (non-thinking / reasoning)

Start local vllm server
Set the server endpoint in `.env` or export environment variables:

```bash
LOCAL_SERVER_ENDPOINT=...
LOCAL_SERVER_PORT=...

# Non-thinking mode
vllm serve Qwen/Qwen3-0.6B \
  --chat-template ./qwen3_nonthinking.jinja \
  --enable-auto-tool-choice \
  --tool-call-parser hermes

# Reasoning mode
vllm serve Qwen/Qwen3-0.6B \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

Then run BFCL eval against the local server:

```bash
bfcl generate --model Qwen/Qwen3-0.6B-FC \
  --test-category simple_python \
  --backend vllm \
  --skip-server-setup \
&& bfcl evaluate --model Qwen/Qwen3-0.6B-FC
```

### OLMo-3 7B

Start local vllm server
Set the server endpoint in `.env` or export environment variables:

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1

vllm serve allenai/Olmo-3-7B-Instruct \
  --gpu-memory-utilization 0.9 \
  --enable-auto-tool-choice \
  --tool-call-parser olmo3
```

Then run BFCL against the local server:

```bash
bfcl generate --model allenai/Olmo-3-7B-Instruct \
  --backend vllm \
  --skip-server-setup \
&& bfcl evaluate --model allenai/Olmo-3-7B-Instruct
```

---

## Notes

- Set `OPENAI_API_KEY` and `OPENAI_BASE_URL` in `.env` when using OpenAI models
- Make sure the relevant model is actively hosted before running generate/evaluate commands.
