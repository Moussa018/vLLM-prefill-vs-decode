"""Step 2: measure PREFILL and DECODE separately.

Simple trick (without touching vLLM internals):
  - t1 = time with max_tokens=1   -> ~ prefill (+ 1 token)
  - tN = time with max_tokens=N   -> prefill + (N-1) decode tokens
  - decode per token = (tN - t1) / (N - 1)

Prefix caching is disabled so that every repetition really redoes the prefill.
"""
import argparse
import random
import statistics
import time

from vllm import LLM, SamplingParams


def make_prompt_ids(n_tokens: int, seed: int = 0) -> list[int]:
    rng = random.Random(seed)
    # ids "normaux" (on évite les tokens spéciaux au début du vocabulaire)
    return [rng.randint(1000, 10000) for _ in range(n_tokens)]


def timed_generate(llm: LLM, prompt_ids: list[int], max_tokens: int) -> float:
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    t0 = time.perf_counter()
    llm.generate([{"prompt_token_ids": prompt_ids}], sp, use_tqdm=False)
    return time.perf_counter() - t0


def median_time(llm, prompt_ids, max_tokens, repeats):
    return statistics.median(timed_generate(llm, prompt_ids, max_tokens) for _ in range(repeats))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompt-lens", type=int, nargs="+", default=[128, 512, 1024, 2048])
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    llm = LLM(model=args.model, enable_prefix_caching=False, max_model_len=4096)

    # warmup (compilation, CUDA graphs, etc.)
    timed_generate(llm, make_prompt_ids(64), 8)

    print(f"{'prompt':>7} | {'prefill (ms)':>12} | {'prefill tok/s':>13} | {'decode ms/tok':>13} | {'decode tok/s':>12}")
    print("-" * 70)
    for n in args.prompt_lens:
        ids = make_prompt_ids(n)
        t1 = median_time(llm, ids, 1, args.repeats)
        tN = median_time(llm, ids, args.decode_tokens, args.repeats)
        decode_per_tok = (tN - t1) / (args.decode_tokens - 1)
        print(
            f"{n:>7} | {t1 * 1000:>12.1f} | {n / t1:>13.0f} | "
            f"{decode_per_tok * 1000:>13.2f} | {1 / decode_per_tok:>12.1f}"
        )


if __name__ == "__main__":
    main()