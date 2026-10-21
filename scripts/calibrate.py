"""Detection thresholds calibrated over keys on responses decoded without the watermark.

Every response is re-tokenized (no special tokens added, as in generate.py) and detected under --num-keys keys (42 first, then 1, 2,
... skipping 42) with the given partition and context width; the z-scores of all (response, key) pairs are pooled. The thresholds for
false-positive rates of 1% and 0.1% are the max(1, floor(aN))-th largest of the N pooled z-scores for a = 0.01 and 0.001 (q99, q999),
so that the false-positive rate holds on average over keys. A response with fewer than two tokens or without a counted unit has no score.

Output JSON: {"partition", "context_width", "keys", "texts": [[arm, prompt_id, seed], ...], "z": {key: [z per text]},
"pooled": {"n", "q99", "q999", "mean", "sd"}}.

Example:
  python scripts/calibrate.py --partition checkpoints/bands.pt --texts runs/none/rows_shard*.jsonl --out calib/bands.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lfbb import detect, load_partition, load_special_ids  # noqa: E402
from lfbb.dream import DREAM_FIRST_SPECIAL  # noqa: E402


def calibration_keys(n):
    return [42] + [k for k in range(1, n + 1) if k != 42][: n - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", required=True, help="learned band table (.pt), 'hash' or 'hash:B' (B bands)")
    ap.add_argument("--texts", nargs="+", required=True, help="JSONL files of unwatermarked responses (field 'completion')")
    ap.add_argument("--arm", help="use only the rows of this arm (default: every row)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-keys", type=int, default=100)
    ap.add_argument("--context-width", type=int, default=2, help="context width of the detector (as the arms' context_width)")
    ap.add_argument("--family", default="llada", choices=["llada", "dream"])
    ap.add_argument("--model", default=None, help="tokenizer (default: GSAI-ML/LLaDA-8B-Instruct or Dream-org/Dream-v0-Instruct-7B)")
    ap.add_argument("--special-ids", default=os.path.join(os.path.dirname(__file__), "..", "data", "special_ids_llada.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    texts = {}
    for path in args.texts:
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                if args.arm is None or r.get("arm") == args.arm:
                    texts[(r.get("arm"), r["prompt_id"], r.get("seed"))] = r["completion"]
    names = sorted(texts, key=lambda t: tuple(map(str, t)))
    print("texts", len(names), flush=True)
    dream = args.family == "dream"
    tok = AutoTokenizer.from_pretrained(args.model or ("Dream-org/Dream-v0-Instruct-7B" if dream else "GSAI-ML/LLaDA-8B-Instruct"),
                                        trust_remote_code=True)
    ids = {t: tok(texts[t], add_special_tokens=False)["input_ids"] for t in names}
    special = range(DREAM_FIRST_SPECIAL, len(tok.get_vocab())) if dream else load_special_ids(args.special_ids)

    keys = calibration_keys(args.num_keys)
    part = load_partition(args.partition, key=keys[0], special_ids=special, device=args.device)
    z = {}
    for key in keys:
        part.key = int(key)
        zz = []
        for t in names:
            if len(ids[t]) < 2:
                zz.append(float("nan"))
                continue
            n, k, s, p = detect(part, ids[t], args.context_width)
            zz.append(round(s, 5) if n else float("nan"))
        z[key] = zz
    allz = np.array([s for key in keys for s in z[key] if s == s])
    desc = np.sort(allz)[::-1]
    pooled = {"n": int(len(allz)), "q99": round(float(desc[max(1, int(0.01 * len(desc))) - 1]), 5),
              "q999": round(float(desc[max(1, int(0.001 * len(desc))) - 1]), 5),
              "mean": round(float(allz.mean()), 4), "sd": round(float(allz.std()), 4)}
    print("pooled", pooled, flush=True)
    res = {"partition": args.partition, "context_width": args.context_width, "keys": keys, "texts": [list(t) for t in names],
           "z": {str(k): z[k] for k in keys}, "pooled": pooled}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
