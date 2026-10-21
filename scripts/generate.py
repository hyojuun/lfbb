"""Watermarked generation and detection on a prompt set.

Every prompt is decoded once per (seed, arm), with all arms of the config file decoded together (they share their random numbers).
The response is cut at the first end-of-text token, decoded, re-tokenized and passed to the detector. --family dream runs
Dream-v0-Instruct-7B with its official decoding (lfbb/dream.py); with --transfer, the partition learned for LLaDA is applied to
Dream's vocabulary through the token map of scripts/transfer_partition.py, and so is the detector. --arm-prefix renames the arms in the
output rows, so that runs of one config with another partition or key can be reported together (scripts/report.py identifies arms by
name).

Example:
  python scripts/generate.py --arms configs/arms/ours_d3_d4.json --partition checkpoints/bands.pt --predictor checkpoints/predictor.pt \
      --prompts data/waterbench600_llada.jsonl --out runs/ours_d3_d4 --shard 0 --num-shards 1
"""
import argparse
import json
import os
import sys
import time
import zlib

import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lfbb import EOS_IDS, ArmConfig, WatermarkSampler, detect, load_partition, load_special_ids  # noqa: E402
from lfbb.dream import DREAM_EOS_ID, DREAM_FIRST_SPECIAL, DreamWatermarkSampler  # noqa: E402
from lfbb.predictor import load_predictor, load_window_predictor  # noqa: E402


def seeds_of(prompt_id, n, start=0):
    return [8_000_000 + 1000 * s + (zlib.crc32(prompt_id.encode()) % 997) for s in range(start, start + n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True, help="JSON list of ArmConfig fields")
    ap.add_argument("--prompts", required=True, help="JSONL with prompt_id, dataset, prompt_token_ids (chat template applied)")
    ap.add_argument("--prompt-ids", help="JSON list of prompt ids to use, in this order (default: all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--partition", required=True, help="learned band table (.pt) or 'hash'")
    ap.add_argument("--predictor", help="foresight predictor checkpoint (needed by arms with foresight 'predictor')")
    ap.add_argument("--transfer", help="token map of scripts/transfer_partition.py through which the partition is applied")
    ap.add_argument("--special-ids", default=os.path.join(os.path.dirname(__file__), "..", "data", "special_ids_llada.json"))
    ap.add_argument("--family", default="llada", choices=["llada", "dream"])
    ap.add_argument("--model", default=None, help="default: GSAI-ML/LLaDA-8B-Instruct or Dream-org/Dream-v0-Instruct-7B")
    ap.add_argument("--key", type=int, default=None, help="watermark key (default: the key stored with the partition, 42)")
    ap.add_argument("--arm-prefix", default="", help="prefix of the arm names in the output rows (e.g. hash_ with --partition hash)")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--gen-length", type=int, default=256)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--commits-per-step", type=int, default=1, help="commits per forward pass (LLaDA)")
    ap.add_argument("--refresh", type=int, default=0, choices=[0, 1, 2],
                    help="with several commits per step: refresh the beliefs around each commit with the predictor (1: rows with "
                         "predicted foresight, 2: every row)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    arms = [ArmConfig.from_dict(a) for a in json.load(open(args.arms))]
    if not args.predictor and any(a.foresight == "predictor" and a.lam > 0 and a.delta > 0 for a in arms):
        ap.error("arms with foresight 'predictor' need --predictor (without it they would use the current beliefs)")
    dream = args.family == "dream"
    model_id = args.model or ("Dream-org/Dream-v0-Instruct-7B" if dream else "GSAI-ML/LLaDA-8B-Instruct")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    V = len(tok.get_vocab())
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
    special = range(DREAM_FIRST_SPECIAL, V) if dream else load_special_ids(args.special_ids)
    if args.transfer:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from transfer_partition import TransferredBands, detect_transferred
        part = TransferredBands(load_partition(args.partition, key=args.key), torch.load(args.transfer)["map"], special)
        detector = detect_transferred
    else:
        part = load_partition(args.partition, key=args.key, special_ids=special)
        detector = detect
    use_pred = args.predictor and any(a.foresight == "predictor" and a.lam > 0 for a in arms)
    if dream:
        pred = (load_window_predictor(args.predictor, model.get_input_embeddings().weight.detach(), model.lm_head.weight.detach())
                if use_pred else None)
        sampler = DreamWatermarkSampler(model, V, part, pred, T=args.temperature, gen_length=args.gen_length)
        eos = (DREAM_EOS_ID,)
    else:
        pred = load_predictor(args.predictor, model) if use_pred else None
        sampler = WatermarkSampler(model, V, part, pred, T=args.temperature, gen_length=args.gen_length, block_length=args.block_length,
                                   commits_per_step=args.commits_per_step, refresh=args.refresh)
        eos = EOS_IDS

    prompts = [json.loads(l) for l in open(args.prompts)]
    if args.prompt_ids:
        order = json.load(open(args.prompt_ids))
        rank = {p: i for i, p in enumerate(order)}
        prompts = sorted((p for p in prompts if p["prompt_id"] in rank), key=lambda p: rank[p["prompt_id"]])
    prompts = prompts[args.shard::args.num_shards]

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"rows_shard{args.shard}.jsonl")
    done = {json.loads(l)["prompt_id"] for l in open(path)} if os.path.exists(path) else set()
    with open(path, "a") as fout:
        for pr in prompts:
            if pr["prompt_id"] in done:
                continue
            t0 = time.time()
            rows = [(sd, arm) for sd in seeds_of(pr["prompt_id"], args.seeds, args.seed_start) for arm in arms]
            outs = sampler.generate(pr["prompt_token_ids"], rows)
            for (sd, arm), o in zip(rows, outs):
                gen = o["tokens"]
                cut = next((i for i, t in enumerate(gen) if t in eos), len(gen))
                text = tok.decode(gen[:cut], skip_special_tokens=True)
                ids = tok(text, add_special_tokens=False)["input_ids"]
                n, k, z, p = detector(part, ids, arm.context_width)
                fout.write(json.dumps({"prompt_id": pr["prompt_id"], "dataset": pr.get("dataset"), "seed": sd,
                                       "arm": args.arm_prefix + arm.name, "key": part.key, "completion": text, "n_gen": cut,
                                       "n_tokens": len(ids), "detection": {"n": n, "k": k, "z": z, "p": p}, "kl": o["kl"]},
                                      ensure_ascii=False) + "\n")
            fout.flush()
            print(f"{pr['prompt_id']} {time.time() - t0:.0f}s", flush=True)
    print("done", args.shard, sampler.stats, flush=True)


if __name__ == "__main__":
    main()
