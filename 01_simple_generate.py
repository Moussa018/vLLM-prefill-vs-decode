"""Step 1: the simplest possible generation with vLLM.

Behind this generate() call, vLLM runs two phases:
  - PREFILL: the whole prompt is processed in a single pass (parallel, compute-bound)
  - DECODE:  tokens are generated one at a time (sequential, memory-bandwidth-bound)
"""
from vllm import LLM , SamplingParams

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

llm = LLM(model=MODEL , enable_prefix_caching= False)
params = SamplingParams(temperature=0.0 , max_tokens=64)

 
outputs = llm.generate(["Explain in 2 sentences the prefill in llm."], params)
for out in outputs:
    print("Prompt tokens   :", len(out.prompt_token_ids))
    print("Generated tokens:", len(out.outputs[0].token_ids))
    print("Texte           :", out.outputs[0].text)