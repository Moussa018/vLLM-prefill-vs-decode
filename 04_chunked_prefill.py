"""Step 4: chunked prefill - a big prompt must not freeze the users who are already decoding.

Scenario (everything is driven by hand with the engine's step() loop, so we can
timestamp every token):
  1. D "chat users" are already decoding (short prompt, long answer).
  2. After a few steps, ONE request with a huge prompt arrives (the "intruder").
  3. We look at the time between two tokens (ITL) seen by the chat users.

The knob is `max_num_batched_tokens` = the token budget of ONE engine step.
  - big budget (>= prompt length): the whole prefill fits in one step
      -> one very heavy step, every chat user waits for it (ITL spike)
  - small budget (e.g. 256): the prefill is cut in slices, each slice shares the
      step with the ongoing decodes -> many light steps, no big spike
      (but the intruder needs more steps, so its own TTFT gets longer)

Each budget runs in its own subprocess (a fresh GPU state for every config).

    python 04_chunked_prefill.py
    python 04_chunked_prefill.py --budgets 8192 2048 512 128 --big-prompt 6000
"""
import argparse
import json
import random
import statistics
import subprocess
import sys
import time

RESULT_TAG = "RESULT "


def make_prompt_ids(n_tokens: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(1000, 10000) for _ in range(n_tokens)]


def percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


# --------------------------------------------------------------------------- #
# Worker: one engine, one budget, one scenario
# --------------------------------------------------------------------------- #
def run_scenario(args) -> dict:
    from vllm import EngineArgs, LLMEngine, SamplingParams

    engine = LLMEngine.from_engine_args(
        EngineArgs(
            model=args.model,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            max_model_len=args.big_prompt + 512,
            max_num_batched_tokens=args.budget,
            max_num_seqs=args.users + 8,
        )
    )

    chat_sp = SamplingParams(temperature=0.0, max_tokens=args.decode_tokens, ignore_eos=True)
    big_sp = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)

    # warmup (compilation, CUDA graphs...)
    engine.add_request("warm", {"prompt_token_ids": make_prompt_ids(64, 999)}, big_sp)
    while engine.has_unfinished_requests():
        engine.step()

    # 1. the chat users start
    for i in range(args.users):
        engine.add_request(f"chat-{i}", {"prompt_token_ids": make_prompt_ids(64, i)}, chat_sp)

    last_time: dict[str, float] = {}      # request id -> time of its previous token
    n_tokens: dict[str, int] = {}         # request id -> number of tokens seen so far
    itl_before: list[float] = []          # gaps (s) of chat users, before the intruder
    itl_after: list[float] = []           # gaps (s) of chat users, while the intruder is processed
    step_times_after: list[float] = []    # duration of each step while the intruder is processed
    t_inject = None
    big_first_token = None
    big_steps = 0
    steps_done = 0

    while engine.has_unfinished_requests():
        # 2. the intruder arrives once the chat users are in steady decode
        if t_inject is None and steps_done == args.warm_steps:
            t_inject = time.perf_counter()
            engine.add_request("big", {"prompt_token_ids": make_prompt_ids(args.big_prompt, 12345)}, big_sp)

        t_step = time.perf_counter()
        outputs = engine.step()
        now = time.perf_counter()
        steps_done += 1

        intruder_running = t_inject is not None and big_first_token is None
        if intruder_running:
            big_steps += 1
            step_times_after.append(now - t_step)

        for out in outputs:
            rid = out.request_id
            count = len(out.outputs[0].token_ids)
            if count > n_tokens.get(rid, 0):
                n_tokens[rid] = count
                if rid.startswith("chat-") and rid in last_time:
                    gap = now - last_time[rid]
                    # gaps measured during the intruder's prefill go in "after"
                    (itl_after if intruder_running else itl_before if t_inject is None else []).append(gap)
                last_time[rid] = now
                if rid == "big" and big_first_token is None:
                    big_first_token = now - t_inject

    return {
        "budget": args.budget,
        "itl_before_p50_ms": statistics.median(itl_before) * 1000 if itl_before else float("nan"),
        "itl_after_p50_ms": statistics.median(itl_after) * 1000 if itl_after else float("nan"),
        "itl_after_p99_ms": percentile(itl_after, 99) * 1000 if itl_after else float("nan"),
        "itl_after_max_ms": max(itl_after) * 1000 if itl_after else float("nan"),
        "big_ttft_s": big_first_token if big_first_token is not None else float("nan"),
        "big_steps": big_steps,
        "heaviest_step_ms": max(step_times_after) * 1000 if step_times_after else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Driver: one subprocess per budget, then a comparison table
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--budgets", type=int, nargs="+", default=[8192, 2048, 512, 128],
                    help="max_num_batched_tokens values to compare (must be >= --users + a few)")
    ap.add_argument("--users", type=int, default=16, help="chat users already decoding")
    ap.add_argument("--big-prompt", type=int, default=6000, help="tokens in the intruder's prompt")
    ap.add_argument("--decode-tokens", type=int, default=256, help="answer length of the chat users")
    ap.add_argument("--warm-steps", type=int, default=30, help="steps before the intruder arrives")
    ap.add_argument("--budget", type=int, help="(internal) run ONE budget in this process")
    args = ap.parse_args()

    if args.budget is not None:  # worker mode
        print(RESULT_TAG + json.dumps(run_scenario(args)))
        return

    rows = []
    for budget in args.budgets:
        print(f"\n=== running with max_num_batched_tokens={budget} ===", flush=True)
        cmd = [sys.executable, __file__, "--budget", str(budget), "--model", args.model,
               "--users", str(args.users), "--big-prompt", str(args.big_prompt),
               "--decode-tokens", str(args.decode_tokens), "--warm-steps", str(args.warm_steps)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        line = next((l for l in proc.stdout.splitlines() if l.startswith(RESULT_TAG)), None)
        if proc.returncode != 0 or line is None:
            print(f"budget {budget} failed (exit code {proc.returncode})")
            continue
        rows.append(json.loads(line[len(RESULT_TAG):]))

    print(f"\n{args.users} chat users decoding, then 1 intruder with a {args.big_prompt}-token prompt\n")
    print(f"{'budget':>7} | {'ITL before':>10} | {'ITL p50':>8} | {'ITL p99':>8} | {'ITL max':>8} | "
          f"{'heaviest step':>13} | {'intruder steps':>14} | {'intruder TTFT':>13}")
    print(f"{'(tokens)':>7} | {'p50 (ms)':>10} | {'(ms)':>8} | {'(ms)':>8} | {'(ms)':>8} | "
          f"{'(ms)':>13} | {'':>14} | {'(s)':>13}")
    print("-" * 105)
    for r in rows:
        print(f"{r['budget']:>7} | {r['itl_before_p50_ms']:>10.1f} | {r['itl_after_p50_ms']:>8.1f} | "
              f"{r['itl_after_p99_ms']:>8.1f} | {r['itl_after_max_ms']:>8.1f} | "
              f"{r['heaviest_step_ms']:>13.1f} | {r['big_steps']:>14} | {r['big_ttft_s']:>13.2f}")


if __name__ == "__main__":
    main()
