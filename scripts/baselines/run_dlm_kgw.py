"""DLM-KGW baseline (Gloaguen et al.): generation and detection with the official code.

The official package dlm_watermark is imported from --official-repo/src, and the model is decoded through its wrapper as in the
reported runs: LLaDA-8B-Instruct, 256 steps for 256 tokens, blocks of 32, temperature 0.5, low-confidence remasking (or the official
'random' order with --remasking random), no classifier-free guidance. The watermark is the official one in its WaterBench
configuration: gamma 0.25, strength --delta, boost of the top-50 candidates, one-token context with the sum hash, one iteration, no
KL constraint; --delta 0 decodes the same pipeline without the watermark. The official detector (greenlist seed 42) scores the
generated token ids before the first end-of-sequence or padding token: units after the official masking of repeated (context, token)
pairs, green units, binomial z-score at gamma and the official exact binomial p-value.

--family dream decodes Dream-v0-Instruct-7B instead (256 steps, 256 tokens, temperature 0.5, remasking 'origin', alg_temp 0.1);
its prompt file must hold token ids from the Dream chat template.

Example:
  python scripts/baselines/run_dlm_kgw.py --official-repo external/diffusion-lm-watermark --prompts data/waterbench600_llada.jsonl \
      --delta 3 --out runs/dlm_kgw_d3
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

GEN_LENGTH = 256
GAMMA = 0.25
FAMILIES = {
    "llada": {"model": "GSAI-ML/LLaDA-8B-Instruct", "wrapper": ("dlm_watermark.models.llada", "LLaDA"), "remasking": "low_confidence",
              "mask_id": 126336, "model_specific_arguments": {"cfg_scale": 0.0, "mask_id": 126336, "block_length": 32}},
    "dream": {"model": "Dream-org/Dream-v0-Instruct-7B", "wrapper": ("dlm_watermark.models.dream", "Dream"), "remasking": "origin",
              "mask_id": 151666, "model_specific_arguments": {"alg_temp": 0.1, "output_history": False}},
}


def import_official(repo):
    src = os.path.abspath(os.path.join(repo, "src"))
    sys.path.insert(0, src)
    import dlm_watermark
    if not os.path.abspath(dlm_watermark.__file__).startswith(src + os.sep):
        raise ImportError(f"dlm_watermark was imported from {dlm_watermark.__file__}, not from {src}")


def official_detection(det, ids, device):
    """(n, k, z, p) of the official detector on the token ids."""
    if not ids:
        return 0, 0, float("nan"), None
    d = det.detect(input_ids=torch.tensor(ids, dtype=torch.long, device=device))
    green = [float(c) for c, keep in zip(d.get("token_color", []), d.get("mask", [])) if keep]
    n, k = len(green), int(sum(green))
    z = (k - n * GAMMA) / math.sqrt(n * GAMMA * (1.0 - GAMMA)) if n else float("nan")
    return n, k, z, d.get("p_value")


def main():
    ap = argparse.ArgumentParser()
    add_arguments(ap)
    ap.add_argument("--family", choices=sorted(FAMILIES), default="llada")
    ap.add_argument("--model", help="model id or path (default: the family's instruct model)")
    ap.add_argument("--delta", type=float, default=4.0, help="watermark strength; 0 decodes without the watermark")
    ap.add_argument("--remasking", help="official remasking strategy (default: low_confidence for LLaDA, origin for Dream)")
    ap.add_argument("--arm", help="arm name of the rows (default: dlm_kgw_d<delta>, or none for delta 0)")
    ap.add_argument("--steps", type=int, default=256,
                    help="denoising steps for the 256 tokens (256: one token per step, the reported runs; 128, 64, 32: 2, 4, 8 tokens per step)")
    args = ap.parse_args()
    fam = FAMILIES[args.family]
    model_id = args.model or fam["model"]
    arm = args.arm or ("none" if args.delta == 0 else f"dlm_kgw_d{args.delta:g}")

    import_official(args.official_repo)
    from dlm_watermark.configs import ModelConfiguration
    from dlm_watermark.watermarks.diffusion_watermark import OurWatermark
    wrapper_cls = getattr(importlib.import_module(fam["wrapper"][0]), fam["wrapper"][1])

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side="left")
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    cfg = ModelConfiguration(model_name=model_id, tokenizer_name=model_id, torch_dtype="bfloat16", device_map="cuda",
                             trust_remote_code=True, steps=args.steps, gen_length=GEN_LENGTH, temperature=0.5,
                             remasking=args.remasking or fam["remasking"], model_specific_arguments=dict(fam["model_specific_arguments"]))
    wrapper = wrapper_cls(cfg, model)
    det = OurWatermark(delta=args.delta, enforce_kl=False, convolution_kernel=[-1], greenlist_type="bernoulli",
                       greenlist_params={"gamma": GAMMA}, tokenizer=tok, topk=50, n_iter=1, seeding_scheme="sumhash", device="cuda:0")
    det.set_temperature(0.5)
    det.set_mask_token(fam["mask_id"])
    key = int(det.greenlist.seed)

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
                ids = torch.tensor([pr["prompt_token_ids"]], device=model.device)
                with torch.no_grad():
                    out = wrapper.generate_diffusion(ids, attention_mask=torch.ones_like(ids), watermark=det if args.delta > 0 else None)
                raw = out.output_sequences[0].detach().cpu().tolist()[:GEN_LENGTH]
                eff = effective_tokens(raw, {tok.eos_token_id}, tok.pad_token_id)
                text = tok.decode(eff, skip_special_tokens=True)
                r = row(pr, seed, arm, key, text, len(eff), official_detection(det, eff, "cuda:0"), {"eff_ids": eff})
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"{pr['prompt_id']} replicate {rep} {time.time() - t0:.0f}s", flush=True)
    print("done", args.shard, flush=True)


if __name__ == "__main__":
    main()
