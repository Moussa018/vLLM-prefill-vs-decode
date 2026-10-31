"""Step 3: continuous batching - send N requests at once and measure throughput.

Idea: decode is memory-bound (the weights are read once per step whatever the
batch size), so the more requests are batched, the more tokens/s we get for
almost the same step cost.

For each batch size N we send N identical-length requests in a single
llm.generate() call. vLLM's scheduler batches them continuously.
We measure:
  - total time for the N requests
  - global throughput  = N * decode_tokens / total_time   (tokens/s, whole GPU)
  - per-request speed  = decode_tokens / total_time        (tokens/s, as seen by 1 user)
  - speedup vs. running the N requests one after the other (N * time_of_batch_1)
"""
import argparse
import random
import time

from vllm import LLM, SamplingParams


def make_prompt_ids(n_tokens: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(1000, 10000) for _ in range(n_tokens)]


def run_batch(llm: LLM, n_requests: int, prompt_len: int, decode_tokens: int) -> float:
    # different seed per request so requests are not identical
    prompts = [{"prompt_token_ids": make_prompt_ids(prompt_len, seed=i)} for i in range(n_requests)]
    sp = SamplingParams(temperature=0.0, max_tokens=decode_tokens, ignore_eos=True)
    t0 = time.perf_counter()
    llm.generate(prompts, sp, use_tqdm=False)
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    llm = LLM(
        model=args.model,
        enable_prefix_caching=False,
        max_model_len=4096,
        max_num_seqs=max(args.batch_sizes),  # allow all N requests to run together
    )

    # warmup
    run_batch(llm, 2, 64, 8)

    print(
        f"{'N':>4} | {'total (s)':>9} | {'global tok/s':>12} | "
        f"{'per-req tok/s':>13} | {'speedup vs seq.':>15}"
    )
    print("-" * 66)

    t_single = None
    for n in args.batch_sizes:
        times = sorted(run_batch(llm, n, args.prompt_len, args.decode_tokens) for _ in range(args.repeats))
        t = times[len(times) // 2]  # median
        if n == 1:
            t_single = t
        global_tps = n * args.decode_tokens / t
        per_req_tps = args.decode_tokens / t
        speedup = f"{n * t_single / t:>14.1f}x" if t_single else f"{'n/a':>15}"
        print(f"{n:>4} | {t:>9.2f} | {global_tps:>12.0f} | {per_req_tps:>13.1f} | {speedup}")


if __name__ == "__main__":
    main()
