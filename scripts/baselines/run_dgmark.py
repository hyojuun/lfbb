"""dgMARK baseline (Hong and No): generation and detection with the official code.

The official modules generation and utils are imported from --official-repo/src and called as its scripts/generate.py calls them,
without a private key (the official default, keyless parity): LLaDA-8B-Instruct, 256 steps for 256 tokens, blocks of 32, no
classifier-free guidance, low-confidence remasking, temperature 0.5 (an argument of the official code, whose default is 0).
Variants:
  none        generate_original                          no watermark
  greedy      generate_watermark_greedy
  top3        generate_watermark_multinomial(top_k=3)
  none_top3   generate_multinomial(top_k=3)              no watermark, the unwatermarked counterpart of top3
  beam3       generate_beam_search(beam_size=3, sampling_strategy='greedy', top_k=3, enable_watermark=True)
The official code draws its noise from the global PyTorch generator, which is seeded for every (prompt, replicate). Detection is the
official match statistic calculate_watermark_stats on the generated ids, up to and including the first end-of-turn token: n positions,
k matches, z = (k - n/2) / sqrt(n/4) and the exact one-sided binomial p-value P(Bin(n, 1/2) >= k). The response text is cut before the
first end-of-sequence or padding token, as for the other methods.

Example:
  python scripts/baselines/run_dgmark.py --official-repo external/dgmark-watermarking --prompts data/waterbench600_llada.jsonl \
      --variant top3 --out runs/dgmark_top3
"""
import argparse
import importlib
import json
import math
import os
import sys
import time

import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol import add_arguments, effective_tokens, generation_seed, load_prompts, open_rows, row, seed_everything  # noqa: E402

VARIANTS = {
    "none": ("generate_original", {}),
    "greedy": ("generate_watermark_greedy", {}),
    "top3": ("generate_watermark_multinomial", {"top_k": 3}),
    "none_top3": ("generate_multinomial", {"top_k": 3}),
    "beam3": ("generate_beam_search", {"beam_size": 3, "sampling_strategy": "greedy", "top_k": 3, "enable_watermark": True}),
}


def import_official(repo):
    """The official WatermarkGenerator and calculate_watermark_stats (top-level modules of the repository's src/)."""
    src = os.path.abspath(os.path.join(repo, "src"))
    sys.path.insert(0, src)
    generation, utils = importlib.import_module("generation"), importlib.import_module("utils")
    for m in (generation, utils):
        if not os.path.abspath(m.__file__).startswith(src + os.sep):
            raise ImportError(f"{m.__name__} was imported from {m.__file__}, not from {src}")
    return generation.WatermarkGenerator, utils.calculate_watermark_stats


def match_detection(matched, n):
    """(n, k, z, p) of the match count: z-score and exact one-sided p-value against Bin(n, 1/2)."""
    if not n:
        return 0, 0, float("nan"), 1.0
    z = (matched - n / 2) / math.sqrt(n / 4)
    p = sum(math.comb(n, i) for i in range(matched, n + 1)) / 2 ** n
    return n, matched, z, p


def main():
    ap = argparse.ArgumentParser()
    add_arguments(ap)
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--arm", help="arm name of the rows (default: dgmark_<variant>)")
    args = ap.parse_args()
    arm = args.arm or f"dgmark_{args.variant}"

    WatermarkGenerator, calculate_watermark_stats = import_official(args.official_repo)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
    gen = WatermarkGenerator(model, tok, "cuda", private_key=None)
    method, extra = VARIANTS[args.variant]
    kw = dict(steps=256, gen_length=256, block_length=32, temperature=args.temperature, cfg_scale=0.0, remasking="low_confidence")

    prompts = load_prompts(args.prompts, args.prompt_ids, args.shard, args.num_shards)
    fout, done = open_rows(args.out, args.shard)
    with fout:
        for pr in prompts:
            for rep in range(args.seed_start, args.seed_start + args.seeds):
                seed = generation_seed(args.study_seed, pr["dataset"], pr["prompt_id"], rep)
                if (pr["prompt_id"], seed) in done:
                    continue
                t0 = time.time()
                seed_everything(seed)
                ids = torch.tensor([pr["prompt_token_ids"]], device="cuda")
                P = ids.shape[1]
                with torch.no_grad():
                    out = getattr(gen, method)(ids, **extra, **kw)
                gen_ids = out[0, P:].tolist()
                matched, _, _, trimmed = calculate_watermark_stats(gen_ids, P, private_key=None)
                eff = effective_tokens(gen_ids, {tok.eos_token_id}, tok.pad_token_id)
                text = tok.decode(eff, skip_special_tokens=True)
                r = row(pr, seed, arm, None, text, len(eff), match_detection(matched, trimmed),
                        {"eff_ids": eff, "gen_ids": gen_ids, "tokens": len(eff), "prompt_len": P})
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"{pr['prompt_id']} replicate {rep} {time.time() - t0:.0f}s", flush=True)
    print("done", args.shard, flush=True)


if __name__ == "__main__":
    main()
