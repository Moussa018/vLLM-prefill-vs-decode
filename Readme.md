# vLLM: prefill vs decode

A progressive project to understand how an LLM serves requests, and why vLLM is built the way it is.
Each step is a small script you can run and measure.

---

# Part 1: the big picture (simple version)

## 1.1 What does an LLM actually do?

An LLM does **one thing**: given a text, it predicts the **next token** (a token is roughly a word or a piece of a word).

```
"The cat sits on the"  ->  LLM  ->  "mat"
```

To write a full answer, we repeat: add the predicted token to the text, ask again, and so on.

```
"The cat sits on the"         -> "mat"
"The cat sits on the mat"     -> "."
"The cat sits on the mat."    -> <end>
```

So generating an answer of 200 tokens means calling the model **200 times in a row**. This is the root of every
performance problem in LLM serving: generation is **sequential**, you cannot produce token 10 before token 9.

## 1.2 The two phases, with an analogy

Imagine a student who must answer a question about a 10-page document.

1. **Reading the document** (all pages are in front of them, they can read as fast as they can think):
   this is the **prefill**. The whole prompt is processed at once.
2. **Writing the answer word by word**, each time glancing back at the notes taken during reading:
   this is the **decode**. One word at a time, and every word requires looking at all the notes again.

| | Prefill (reading) | Decode (writing) |
|---|---|---|
| Input | the whole prompt (e.g. 1000 tokens) | 1 new token |
| Parallel? | yes, all tokens together | no, strictly one after another |
| What limits it | raw computing power of the GPU | speed of reading memory |
| Metric the user feels | **TTFT**: time before the first token appears | **TPOT / ITL**: time between two tokens |

## 1.3 What is the KV cache?

The "notes" of the analogy. For each token already seen, the model computes two small vectors (a **K**ey and
a **V**alue) at every layer. To produce the next token, the model must look at the K and V of **all previous tokens**.

Recomputing them at every step would be wasteful, so we **store them in GPU memory**: this is the **KV cache**.

- Prefill **fills** the KV cache (for the whole prompt).
- Decode **reads** the KV cache at every step and **appends** one new entry.

## 1.4 Why is the GPU used so differently in the two phases?

A GPU has two resources:

- **Compute** (how many multiplications per second it can do)
- **Memory bandwidth** (how many bytes per second it can read from its memory)

At every forward pass the GPU must read **all the model weights** (e.g. 1 GB for a 0.5B model in 16 bits).

- In **prefill**, 1000 tokens share that single read of the weights: lots of computing for each byte read.
  The GPU is busy computing -> **compute-bound**.
- In **decode**, one token alone pays for the whole read of the weights: almost no computing for each byte read.
  The GPU mostly waits for memory -> **memory-bandwidth-bound**.

> **Key idea of the project**: decode wastes the GPU's compute power. The trick to use it
> is to decode **many requests at the same time** (batching), because they all share the same weight read.
> That is the subject of step 3.

## 1.5 Where the project is going

```
Step 1  one generation                     -> see the two phases exist
Step 2  measure prefill and decode         -> see they behave very differently
Step 3  many requests at once (batching)   -> exploit the idle GPU during decode
Step 4  chunked prefill                    -> stop big prompts from freezing other users (done)
Step 5  disaggregated prefill/decode       -> give each phase its own GPU
```

---

# Part 2: going deeper

## 2.1 Architecture of one forward pass

```
 tokens -> [embedding] -> [layer 1] -> [layer 2] -> ... -> [layer L] -> [logits] -> next token
                              |
                  attention: Q (new tokens) x K,V (all tokens, from KV cache)
                  MLP:       big matrix multiplications with the weights
```

Each layer has two costly parts:

- **MLP / projections**: matrix multiplications with the **weights**. Cost in memory: read the weights once per pass,
  whatever the number of tokens. Cost in compute: proportional to the number of tokens.
- **Attention**: each new token compares its Query with the Keys of **all previous tokens**, then mixes their Values.
  Cost in memory: read the KV cache of every request. This part grows with the context length and
  is **not shared** between requests (each request has its own KV cache).

## 2.2 Compute-bound vs memory-bound, with numbers

The useful measure is **arithmetic intensity**: how many operations per byte read from memory.

For a model with P parameters in 16 bits (2 bytes each), a pass over T tokens costs about:

- compute: `2 * P * T` operations
- memory: `2 * P` bytes (the weights, read once)
- intensity: `(2 * P * T) / (2 * P) = T` operations per byte

So **the intensity is simply the number of tokens processed in the pass**.

A GPU has a "break-even point": its peak compute divided by its memory bandwidth. For an A100 that is
roughly `312 TFLOP/s / 2 TB/s ~ 150 operations per byte`.

| Situation | Tokens in the pass (T) | Compared to ~150 | Regime |
|---|---|---|---|
| Decode, 1 request | 1 | far below | memory-bound (GPU ~99% idle) |
| Decode, batch of 64 | 64 | below | still memory-bound, but 64x more useful work |
| Decode, batch of ~150+ | 150+ | about equal | switching to compute-bound |
| Prefill, 1000-token prompt | 1000 | far above | compute-bound |

This one table explains steps 2, 3 and 4:

- step 2: prefill (T large) and decode (T = 1) behave very differently;
- step 3: batching raises T in the decode, almost for free, until ~150;
- step 4: a big prefill (T large) mixed with decodes makes the whole pass slow.

## 2.3 The KV cache: size and why it limits everything

Size per token = `2 (K and V) * layers * kv_heads * head_dim * bytes`.
For Qwen2.5-0.5B (24 layers, 2 KV heads, head dim 64, 16 bits) that is about **12 KB per token**.
A 1000-token request therefore holds about 12 MB; larger models need orders of magnitude more.

GPU memory is split in three: **weights + KV cache + temporary activations**.
The number of requests that can run together is limited by how many KV caches fit in memory.

**The problem vLLM solved (PagedAttention).** Old systems reserved one big contiguous memory block per request,
sized for the *maximum possible* length. Most of it was never used (fragmentation), so few requests fitted.
vLLM cuts the KV cache into small fixed-size **blocks** (like pages in an operating system), allocated
one by one as the request grows, and freed as soon as it ends. Almost no waste -> many more requests in the
batch -> much higher throughput.

## 2.4 The scheduler: continuous batching

At each step the vLLM scheduler decides which requests take part in the forward pass, under two budgets:
free KV cache blocks, and a maximum number of tokens per step.

- A request that finishes frees its blocks **immediately**.
- A waiting request enters **as soon as there is room**.

Older "static" batching waited for the whole batch to finish, leaving the GPU idle while a long request
completed. See step 3 for details.

## 2.5 Metrics cheat sheet

| Metric | Definition | Mostly depends on |
|---|---|---|
| **TTFT** | time from request arrival to first token | prefill cost + queue waiting time |
| **TPOT / ITL** | time between two consecutive tokens | decode step time (weights read + KV cache read) |
| **Throughput** | total tokens/s produced by the GPU | batch size, memory available for KV cache |
| **Latency** | TTFT + (n - 1) * TPOT | everything above |

The central tension of LLM serving: **raising throughput (bigger batches) tends to worsen latency**.
The steps of this project explore how to reduce that trade-off.

---

# Part 3: the steps

## Installation (NVIDIA GPU required)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Step 1: simple generation

```bash
python 01_simple_generate.py
```

One call to `generate()`. Behind it vLLM runs a prefill (the prompt) and then a decode (the 64 new tokens).
Goal: check that the setup works and see prompt tokens vs generated tokens.

## Step 2: measure prefill and decode separately

```bash
python 02_prefill_vs_decode.py
python 02_prefill_vs_decode.py --prompt-lens 256 1024 3072 --decode-tokens 256
```

**How it measures** without touching vLLM internals:

- `t1` = time with `max_tokens=1` -> about the prefill (plus one token)
- `tN` = time with `max_tokens=N` -> prefill + (N-1) decode steps
- decode per token = `(tN - t1) / (N - 1)`

Prefix caching is disabled so every repetition really redoes the prefill.

**What to observe**: prefill gets more expensive as the prompt grows (but its throughput in tokens/s stays high),
while decode stays roughly constant per token and is much slower in tokens/s. This is section 2.2 in practice.

## Step 3: continuous batching

```bash
python 03_batching.py
python 03_batching.py --batch-sizes 1 8 32 128 --prompt-len 512
```

Sends N requests at once and measures throughput.

### Why batching is almost free in decode

From section 2.2: a decode step reads all the weights to produce 1 token. With 16 requests in the same
pass, the weights are still read **once**, and 16 tokens come out. The step cost barely changes, the number
of tokens produced is multiplied by the batch size.

| Batch size | Decode step time | Tokens per step | Throughput |
|-----------:|-----------------:|----------------:|-----------:|
| 1  | ~10 ms | 1  | ~100 tok/s |
| 16 | ~11 ms | 16 | ~1450 tok/s |
| 64 | ~14 ms | 64 | ~4500 tok/s |

*(illustrative numbers, not measurements)*

Throughput grows almost linearly with N until the GPU stops being memory-bound (around the break-even point
of section 2.2), then flattens. The step time also slowly increases because the attention part reads one
KV cache per request, and that is not shared.

### Static vs continuous batching

- **Static batching**: group N requests and wait until *all* of them finish. A request that generates
  20 tokens keeps its slot idle while another generates 500.
- **Continuous batching** (vLLM): the scheduler decides **at every step** who takes part. A finished request
  frees its slot, a waiting request takes it:

```
step 1: [A B C D]
step 2: [A B C D]
step 3: [A B C D]   <- B finishes (EOS or max_tokens)
step 4: [A E C D]   <- E, which was waiting, takes B's place right away
```

### What the script measures

For each batch size N, one `llm.generate()` call with N prompts, repeated 3 times (median kept):

| Column | Meaning |
|--------|---------|
| `total (s)` | wall-clock time to finish all N requests |
| `global tok/s` | `N * decode_tokens / total`: tokens produced by the whole GPU per second |
| `per-req tok/s` | `decode_tokens / total`: speed seen by one user |
| `speedup vs seq.` | `N * time(N=1) / time(N)`: gain compared to running the N requests one after another |

### What to observe

1. **Global throughput** rises strongly with N, then plateaus.
2. **Per-request speed** drops slightly: each user shares the GPU with the others.
3. **Total time for N requests** is far below N times the time of a single request.

### Limits (and why step 4 exists)

- When a new request arrives, its **prefill** (compute-bound, long) runs on the same GPU as the ongoing
  decodes. Every request already decoding waits for that heavy step, so the delay between its tokens (ITL) spikes.
- A busy server makes new requests wait in the queue, which increases their TTFT.
- Per-request TTFT is not measured here: vLLM's offline API does not expose it reliably (it needs server mode).

## Step 4: chunked prefill

```bash
python 04_chunked_prefill.py
python 04_chunked_prefill.py --budgets 8192 2048 512 128 --big-prompt 6000 --users 16
```

### The problem (left open by step 3)

Take a server where 16 users are chatting: every step produces 1 token for each of them, a step takes ~10 ms,
so each user sees a new token every ~10 ms. Now a 6000-token prompt arrives (the "intruder").

Without any protection, the scheduler puts the **whole prefill in one step**. From section 2.2, a pass over
6000 tokens is compute-bound and takes far longer than a decode step (tens or hundreds of ms). During that
single step **nobody else gets a token**:

```
without chunking:
step 1   [chat x16]                      10 ms   -> 16 tokens
step 2   [chat x16]                      10 ms   -> 16 tokens
step 3   [chat x16 + PREFILL 6000]      ~150 ms  -> 16 tokens   <- every user waits 150 ms: ITL spike
step 4   [chat x16]                      10 ms   -> 16 tokens
```

*(illustrative numbers, not measurements)*

The users feel it as the text "freezing" for a moment. The average ITL looks fine, but the **tail** (p99, max) is bad.

### The solution: cut the prefill in slices

Each step has a **token budget** (`max_num_batched_tokens`): the maximum number of tokens processed in one forward pass.
The decodes take 1 token each (16 of the budget), and the rest of the budget is given to the prefill:

```
with a budget of 512:
step 3   [chat x16 + prefill slice 1 (496 tokens)]    ~20 ms
step 4   [chat x16 + prefill slice 2 (496 tokens)]    ~20 ms
...
step 15  [chat x16 + prefill slice 13 (last tokens)]  ~15 ms   -> intruder gets its 1st token
```

Every step is now light. The chat users still get one token at every step, only a bit slower (20 ms instead of 10 ms),
instead of one 150 ms freeze. This is what `enable_chunked_prefill=True` does, and the scheduler logic is simply:
*"fill the step with the decodes first, then spend the remaining budget on prefill tokens"*.

Bonus: mixing decode tokens (memory-bound) and prefill tokens (compute-bound) in the same pass uses the GPU
better than separate passes, since the weights are read once for both (section 2.2 again).

### The trade-off

| Budget | Chat users' ITL | Intruder's TTFT | Steps for the prefill |
|---|---|---|---|
| big (>= prompt length) | rare but **huge spike** | **best** (everything in one step) | 1 |
| medium (512) | small, regular slowdown | a bit longer | ~13 |
| tiny (128) | smoothest | **much longer** | ~50 |

A budget too small also wastes efficiency (many small passes, each re-reading the weights). In practice
you pick a budget (vLLM's default is 2048 to 8192 depending on the version/hardware) that keeps the
spike acceptable without delaying the prefill too much. It is the same central tension as section 2.5:
**protecting the latency of the existing users costs latency to the new request**.

### How the script measures it

The offline `llm.generate()` returns only at the end, so we cannot see the individual tokens. Instead
the script drives the engine by hand (`LLMEngine`: `add_request()` + `step()` in a loop) and takes a timestamp
every time a request gets a new token.

1. 16 chat users start (64-token prompt, 256 tokens to generate).
2. After 30 steps (steady decode), one request with a 6000-token prompt is added.
3. For each chat user we record the gap between two consecutive tokens (ITL), separately **before** the intruder
   and **while its prefill is running**.
4. The same scenario is replayed for each budget, each in a **separate process** (clean GPU state).

The `max_num_batched_tokens` knob is what changes between runs: the biggest value (8192) lets the whole
prompt fit in one step, so it behaves like "no chunking"; the small values force the slicing.

| Column | Meaning |
|---|---|
| `budget` | `max_num_batched_tokens`: tokens allowed in one step |
| `ITL before p50` | normal gap between two tokens, before the intruder (the reference) |
| `ITL p50 / p99 / max` | gap between two tokens of chat users while the intruder is being prefilled |
| `heaviest step` | duration of the longest engine step during that period |
| `intruder steps` | number of steps needed to finish the intruder's prefill |
| `intruder TTFT` | time between the arrival of the intruder and its first token |

### What to observe

1. With a big budget, **`ITL max` and `heaviest step` are large** (about the duration of the whole prefill), while `ITL p50` stays normal.
   The average hides the problem: look at the tail.
2. When the budget decreases, `ITL max` drops toward the reference value.
3. At the same time, `intruder steps` and `intruder TTFT` go up. This is the price.
4. `ITL p50` slightly increases for small budgets: each step now carries prefill tokens too.

> Note: written for the vLLM `LLMEngine` API (`from_engine_args`, `add_request`, `step`). If your version changed
> these signatures, adapt the few lines in `run_scenario()`. In recent vLLM (V1 engine) chunked prefill is
> always on, and `max_num_batched_tokens` is the real control, which is why the script varies it.

### Limits (and why step 5 exists)

Even with chunking, the prefill and the decodes **still share the same GPU**: the decodes are slowed down
(a little, regularly) by the prefill tokens in their steps. Chunking reduces the spike but does not remove the
interference. To remove it completely, the two phases must run on different GPUs.

## Next steps

- **Step 5**: disaggregated prefill/decode (two vLLM instances, KV cache transfer via the KV connector).
  Idea: since the phases need different things (compute vs memory bandwidth), run them on separate GPUs
  and send the KV cache from the prefill GPU to the decode GPU.
