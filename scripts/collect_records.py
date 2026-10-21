"""Commit records of unwatermarked decoding: the training data of the band table (scripts/learn_bands.py).

Each training prompt is decoded without a watermark, in LLaDA's random order or in its low-confidence order (or with Dream's official
decoding, --family dream). Every commit gives a record with the committed position and the top-50 candidates with their probabilities
at temperature T; records after the first end-of-text token are dropped by the learner (field "cut").

Example:
  python scripts/collect_records.py --prompts data/train_prompts_llada.jsonl --out records/random --order random --shard 0 --num-shards 4
"""
import argparse
import json
import os
import sys

import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lfbb import EOS_IDS, ArmConfig, WatermarkSampler, load_partition, load_special_ids  # noqa: E402
from lfbb.dream import DREAM_EOS_ID, DREAM_FIRST_SPECIAL, DreamWatermarkSampler  # noqa: E402
from generate import seeds_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="JSONL with prompt_id, dataset, prompt_token_ids")
    ap.add_argument("--ids", help="JSON list of prompt ids to use (default: all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--order", default="random", choices=["random", "low_confidence"])
    ap.add_argument("--family", default="llada", choices=["llada", "dream"])
    ap.add_argument("--model", default=None, help="default: GSAI-ML/LLaDA-8B-Instruct or Dream-org/Dream-v0-Instruct-7B")
    ap.add_argument("--special-ids", default=os.path.join(os.path.dirname(__file__), "..", "data", "special_ids_llada.json"))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    dream = args.family == "dream"
    model_id = args.model or ("Dream-org/Dream-v0-Instruct-7B" if dream else "GSAI-ML/LLaDA-8B-Instruct")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    V = len(tok.get_vocab())
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
    special = range(DREAM_FIRST_SPECIAL, V) if dream else load_special_ids(args.special_ids)
    part = load_partition("hash", special_ids=special)                                   # unused without a watermark
    sampler = (DreamWatermarkSampler if dream else WatermarkSampler)(model, V, part, T=args.temperature)
    arm = ArmConfig("none", delta=0.0, order="random" if args.order == "random" else "current_tilt")
    eos = (DREAM_EOS_ID,) if dream else EOS_IDS

    rec = {}

    def add(r, pos, tok_, cand, cand_p):
        o = rec.setdefault(r, {"pos": [], "tok": [], "cand": [], "cand_p": []})
        o["pos"].append(pos); o["tok"].append(int(tok_))
        o["cand"].append(cand.to(torch.int32).cpu()); o["cand_p"].append(cand_p.half().cpu())

    def on_commit(x, logits, tok, cand, cand_p, P, step, j=None, r=None, positions=None):
        if positions is None:                     # one commit per row and step
            for q in range(x.shape[0]):
                add(q, int(j[q]) - P, tok[q], cand[q], cand_p[q])
        else:                                     # the positions revealed by row r at this step
            for q, m in enumerate(positions.tolist()):
                add(r, m - P, tok[q], cand[q], cand_p[q])
    sampler.on_commit = on_commit

    os.makedirs(os.path.join(args.out, "items"), exist_ok=True)
    prompts = [json.loads(l) for l in open(args.prompts)]
    if args.ids:
        keep = set(json.load(open(args.ids)))
        prompts = [p for p in prompts if p["prompt_id"] in keep]
    prompts = prompts[args.shard::args.num_shards]
    for pr in prompts:
        path = os.path.join(args.out, "items", pr["prompt_id"].replace("/", "_").replace(":", "_") + ".pt")
        if os.path.exists(path):
            continue
        rec.clear()
        seeds = seeds_of(pr["prompt_id"], args.seeds)
        outs = sampler.generate(pr["prompt_token_ids"], [(sd, arm) for sd in seeds])
        rows = []
        for r, (sd, o) in enumerate(zip(seeds, outs)):
            gen = o["tokens"]
            cut = next((i for i, t in enumerate(gen) if t in eos), len(gen))
            rows.append({"seed": sd, "tokens": gen, "cut": cut,
                         "rec": {"pos": torch.tensor(rec[r]["pos"]), "tok": torch.tensor(rec[r]["tok"]),
                                 "cand": torch.stack(rec[r]["cand"]), "cand_p": torch.stack(rec[r]["cand_p"])}})
        torch.save({"prompt_id": pr["prompt_id"], "task": pr.get("dataset", ""), "rows": rows}, path + ".tmp")
        os.replace(path + ".tmp", path)
        print(pr["prompt_id"], flush=True)
    print("done", args.shard, flush=True)


if __name__ == "__main__":
    main()
