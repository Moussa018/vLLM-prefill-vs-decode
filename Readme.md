# vLLM: prefill vs decode

A progressive project to understand the two inference phases of an LLM with vLLM.

## Concepts

| Phase | What happens | Bottleneck | Metric |
|-------|--------------|------------|--------|
| **Prefill** | the whole prompt is processed at once, the KV cache is filled | compute-bound | TTFT (time to first token) |
| **Decode** | one token is generated per forward pass, the KV cache is re-read each time | memory-bandwidth-bound | ms/token (TPOT) |

## Installation (NVIDIA GPU required)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Step 1: simple generation

```bash
python 01_simple_generate.py
```

## Step 2: measure prefill and decode separately

```bash
python 02_prefill_vs_decode.py
python 02_prefill_vs_decode.py --prompt-lens 256 1024 3072 --decode-tokens 256
```

What to observe: prefill gets more expensive as the prompt grows (but its throughput in tokens/s stays high),
while decode stays roughly constant per token and is much slower in tokens/s.

## Next steps

- **Step 3**: multiple parallel requests (continuous batching), and their effect on TTFT and throughput
- **Step 4**: chunked prefill (`enable_chunked_prefill=True`) so large prompts don't block other requests' decode
- **Step 5**: disaggregated prefill/decode (two vLLM instances, KV cache transfer via the KV connector)