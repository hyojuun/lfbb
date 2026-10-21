"""Prompts, seeds, response cut and output rows shared by the baseline drivers.

Each (prompt, replicate) is decoded once, batch size 1, after seeding Python, NumPy and PyTorch with
generation_seed(study_seed, dataset, prompt_id, replicate): the first 8 bytes of a SHA-256 of these fields, as a 63-bit integer.
The response is cut before the first end-of-sequence or padding token of the tokenizer. Rows have the format of scripts/generate.py.
"""
import hashlib
import json
import os
import random

import numpy as np
import torch

STUDY_SEED = "dlm-waterbench600-g256-b32-v1"     # seed string of the baseline runs reported in the paper


def generation_seed(study_seed, dataset, prompt_id, replicate):
    payload = f"{study_seed}\0{dataset}\0{prompt_id}\0{replicate}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def effective_tokens(raw, eos_ids, pad_id):
    """The generated tokens before the first end-of-sequence or padding token."""
    for i, t in enumerate(raw):
        if t in eos_ids or (pad_id is not None and t == pad_id):
            return raw[:i]
    return raw


def add_arguments(ap):
    ap.add_argument("--official-repo", required=True, help="root of a checkout of the baseline's official repository")
    ap.add_argument("--prompts", required=True, help="JSONL with prompt_id, dataset, prompt_token_ids (chat template applied)")
    ap.add_argument("--prompt-ids", help="JSON list of prompt ids to use, in this order (default: all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=1, help="replicates per prompt")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--study-seed", default=STUDY_SEED)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)


def load_prompts(path, ids_path, shard, num_shards):
    prompts = [json.loads(line) for line in open(path) if line.strip()]
    if ids_path:
        rank = {p: i for i, p in enumerate(json.load(open(ids_path)))}
        prompts = sorted((p for p in prompts if p["prompt_id"] in rank), key=lambda p: rank[p["prompt_id"]])
    return prompts[shard::num_shards]


def open_rows(out_dir, shard):
    """The shard's row file in append mode and the (prompt_id, seed) pairs it already holds."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"rows_shard{shard}.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                done.add((r["prompt_id"], r["seed"]))
    return open(path, "a"), done


def row(prompt, seed, arm, key, completion, n_tokens, detection, extra=None):
    """detection = (n, k, z, p) of the baseline's own detector. The tilt's KL divergence is not available (null). extra adds fields,
    e.g. the generated token ids that the baselines' detectors read (scripts/eval_codeqa.py)."""
    n, k, z, p = detection
    r = {"prompt_id": prompt["prompt_id"], "dataset": prompt.get("dataset"), "seed": seed, "arm": arm, "key": key,
         "completion": completion, "n_gen": n_tokens, "n_tokens": n_tokens, "detection": {"n": n, "k": k, "z": z, "p": p}, "kl": None}
    r.update(extra or {})
    return r
