"""Step 5: disaggregated prefill/decode - each phase gets its own GPU.

Same scenario as step 4 (D chat users decoding, then one big-prompt "intruder"), but replayed on two setups:

  colocated      : ONE vLLM server (GPU 0) does prefill AND decode, as in steps 1-4.
  disaggregated  : TWO vLLM servers.
                     prefill server (GPU 0): computes the KV cache of the prompt, generates 1 token
                     decode  server (GPU 1): receives the KV cache (NIXL connector), generates the rest
                   The decode GPU never runs a big prefill -> no interference with the chat users.

Request flow in disaggregated mode (this script plays the role of the "proxy"):
  1. POST /v1/completions to the prefill server with max_tokens=1 and
     kv_transfer_params={"do_remote_decode": true}   -> response contains kv_transfer_params
        (where the KV blocks are: host, port, block ids)
  2. POST /v1/completions to the decode server with those kv_transfer_params, streaming
        -> it pulls the KV cache from the prefill GPU, skips the prefill and decodes.

Requirements: 2 NVIDIA GPUs, `pip install vllm nixl aiohttp`.

    python 05_disaggregated.py
    python 05_disaggregated.py --big-prompt 6000 --users 16
"""
import argparse
import asyncio
import json
import os
import random
import statistics
import subprocess
import sys
import time
import urllib.request

import aiohttp

PREFILL_PORT, DECODE_PORT, COLOC_PORT = 8100, 8200, 8300


# --------------------------------------------------------------------------- #
# Server management
# --------------------------------------------------------------------------- #
def start_server(model, port, gpu, side_channel_port, big_prompt, disagg):
    cmd = [
        "vllm", "serve", model, "--port", str(port),
        "--max-model-len", str(big_prompt + 512),
        "--no-enable-prefix-caching",
        "--gpu-memory-utilization", "0.7",
    ]
    if disagg:
        cmd += ["--kv-transfer-config", json.dumps({"kv_connector": "NixlConnector", "kv_role": "kv_both"})]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), VLLM_NIXL_SIDE_CHANNEL_PORT=str(side_channel_port))
    return subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_ready(port, proc, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server on port {port} died (exit code {proc.returncode})")
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
            return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"server on port {port} not ready after {timeout}s")


def stop_servers(procs):
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()


# --------------------------------------------------------------------------- #
# Client: one request = prompt of token ids, streaming, timestamps of every token
# --------------------------------------------------------------------------- #
def make_prompt_ids(n_tokens, seed):
    rng = random.Random(seed)
    return [rng.randint(1000, 10000) for _ in range(n_tokens)]


async def post_json(session, port, body):
    async with session.post(f"http://localhost:{port}/v1/completions", json=body) as r:
        r.raise_for_status()
        return await r.json()


async def stream_tokens(session, port, body, stamps):
    """Stream a completion; append time.perf_counter() to `stamps` for every chunk (= token)."""
    body = dict(body, stream=True)
    async with session.post(f"http://localhost:{port}/v1/completions", json=body) as r:
        r.raise_for_status()
        async for raw in r.content:
            line = raw.decode().strip()
            if line.startswith("data:") and line != "data: [DONE]":
                stamps.append(time.perf_counter())


async def run_request(session, args, model, prompt_ids, max_tokens, disagg, stamps):
    base = {"model": model, "prompt": prompt_ids, "temperature": 0.0, "ignore_eos": True}
    if not disagg:
        await stream_tokens(session, COLOC_PORT, dict(base, max_tokens=max_tokens), stamps)
        return
    # 1. prefill server computes the KV cache (+ first token)
    pre = await post_json(session, PREFILL_PORT, dict(
        base, max_tokens=1, stream=False,
        kv_transfer_params={"do_remote_decode": True, "do_remote_prefill": False, "remote_engine_id": None,
                            "remote_block_ids": None, "remote_host": None, "remote_port": None}))
    stamps.append(time.perf_counter())  # first token comes from the prefill server
    # 2. decode server pulls the KV cache and continues from there
    await stream_tokens(session, DECODE_PORT,
                        dict(base, max_tokens=max_tokens, kv_transfer_params=pre["kv_transfer_params"]),
                        stamps)


async def scenario(args, model, disagg):
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:
        # warmup (compilation, CUDA graphs, connection setup)
        await run_request(session, args, model, make_prompt_ids(64, 999), 4, disagg, [])

        chat = [[] for _ in range(args.users)]
        tasks = [asyncio.create_task(run_request(session, args, model, make_prompt_ids(64, i),
                                                 args.decode_tokens, disagg, chat[i]))
                 for i in range(args.users)]
        await asyncio.sleep(args.warm_seconds)          # chat users in steady decode

        big = []
        t_inject = time.perf_counter()
        await run_request(session, args, model, make_prompt_ids(args.big_prompt, 12345), 4, disagg, big)
        t_end = time.perf_counter()
        await asyncio.gather(*tasks)

    def gaps(lo, hi):
        out = []
        for st in chat:
            out += [b - a for a, b in zip(st, st[1:]) if lo <= b <= hi]
        return out

    before, during = gaps(0, t_inject), gaps(t_inject, t_end)
    pct = lambda v, p: sorted(v)[min(len(v) - 1, int(p / 100 * len(v)))] * 1000 if v else float("nan")
    return {
        "mode": "disaggregated" if disagg else "colocated",
        "itl_before_p50": statistics.median(before) * 1000 if before else float("nan"),
        "itl_during_p50": statistics.median(during) * 1000 if during else float("nan"),
        "itl_during_p99": pct(during, 99),
        "itl_during_max": max(during) * 1000 if during else float("nan"),
        "intruder_ttft": big[0] - t_inject if big else float("nan"),
        "intruder_total": t_end - t_inject,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--users", type=int, default=16, help="chat users already decoding")
    ap.add_argument("--big-prompt", type=int, default=6000, help="tokens in the intruder's prompt")
    ap.add_argument("--decode-tokens", type=int, default=512, help="answer length of the chat users")
    ap.add_argument("--warm-seconds", type=float, default=2.0, help="seconds before the intruder arrives")
    args = ap.parse_args()

    rows = []
    for disagg in (False, True):
        procs = []
        try:
            if disagg:
                print("\n=== disaggregated: prefill on GPU 0, decode on GPU 1 ===", flush=True)
                procs.append(start_server(args.model, PREFILL_PORT, 0, 5600, args.big_prompt, True))
                procs.append(start_server(args.model, DECODE_PORT, 1, 5601, args.big_prompt, True))
                wait_ready(PREFILL_PORT, procs[0]); wait_ready(DECODE_PORT, procs[1])
            else:
                print("\n=== colocated: prefill + decode on GPU 0 ===", flush=True)
                procs.append(start_server(args.model, COLOC_PORT, 0, 5600, args.big_prompt, False))
                wait_ready(COLOC_PORT, procs[0])
            rows.append(asyncio.run(scenario(args, args.model, disagg)))
        except Exception as e:
            print(f"{'disaggregated' if disagg else 'colocated'} failed: {e}")
        finally:
            stop_servers(procs)

    print(f"\n{args.users} chat users decoding, then 1 intruder with a {args.big_prompt}-token prompt\n")
    print(f"{'mode':>14} | {'ITL before':>10} | {'ITL p50':>8} | {'ITL p99':>8} | {'ITL max':>8} | "
          f"{'intruder TTFT':>13} | {'intruder total':>14}")
    print(f"{'':>14} | {'p50 (ms)':>10} | {'(ms)':>8} | {'(ms)':>8} | {'(ms)':>8} | {'(s)':>13} | {'(s)':>14}")
    print("-" * 100)
    for r in rows:
        print(f"{r['mode']:>14} | {r['itl_before_p50']:>10.1f} | {r['itl_during_p50']:>8.1f} | "
              f"{r['itl_during_p99']:>8.1f} | {r['itl_during_max']:>8.1f} | "
              f"{r['intruder_ttft']:>13.2f} | {r['intruder_total']:>14.2f}")


if __name__ == "__main__":
    main()
