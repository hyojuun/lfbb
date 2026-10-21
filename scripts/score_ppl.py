"""Log-perplexity of the responses under a scoring model (Qwen3-8B), conditioned on the prompt.

A response is scored as the reply to its prompt: the prompt is the user turn of the scorer's chat template (generation prompt appended,
thinking disabled), the response is tokenized without special tokens and appended, and the log-perplexity is the mean negative
log-likelihood of the response tokens in one teacher-forced forward pass (bf16 weights, SDPA attention, no truncation). A response that
is empty after stripping whitespace has no score (null).

Writes RUN_DIR/ppl_<arm>.json for every arm of every run directory: {"model", "revision", "prompt_field", "n", "mean", "median",
"log_ppl": {prompt_id: {seed: log-PPL}}}, where n, mean and median cover the scored responses. Existing files are resumed.

Example:
  python scripts/score_ppl.py --runs runs/ours runs/dlm_kgw --prompts data/waterbench600_llada.jsonl
"""
import argparse
import glob
import json
import os
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def token_list(enc):
    """One sequence of token ids from the output of a tokenizer or chat template."""
    if isinstance(enc, Mapping):
        enc = enc["input_ids"]
    if hasattr(enc, "tolist"):
        enc = enc.tolist()
    if len(enc) == 1 and isinstance(enc[0], list):
        enc = enc[0]
    return [int(t) for t in enc]


class ResponseScorer:
    def __init__(self, model, revision=None, device="cuda:0", max_length=8192):
        self.tok = AutoTokenizer.from_pretrained(model, revision=revision, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model, revision=revision, torch_dtype=torch.bfloat16,
                                                          attn_implementation="sdpa").to(device).eval().requires_grad_(False)
        self.device, self.max_length = device, max_length

    @torch.no_grad()
    def log_ppl(self, prompt, response):
        if not response.strip():
            return None
        prefix = token_list(self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                                         add_generation_prompt=True, enable_thinking=False))
        resp = token_list(self.tok.encode(response, add_special_tokens=False))
        if len(prefix) + len(resp) > self.max_length:
            raise ValueError(f"prompt and response ({len(prefix)} + {len(resp)} tokens) exceed {self.max_length} tokens")
        ids = torch.tensor([prefix + resp], dtype=torch.long, device=self.device)
        logits = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        sel = logits[0, len(prefix) - 1:-1].float()                         # the positions that predict the response tokens
        if not torch.isfinite(sel).all():
            raise FloatingPointError("non-finite logits")
        return float(F.cross_entropy(sel, ids[0, len(prefix):], reduction="mean"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run directories with rows_shard*.jsonl")
    ap.add_argument("--prompts", required=True, help="JSONL with prompt_id and the prompt text")
    ap.add_argument("--prompt-field", default="prompt_text", help="field of the prompt file used as the user turn")
    ap.add_argument("--arms", help="comma-separated arms to score (default: all)")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--revision", default="b968826d9c46dd6066d109eabc6255188de91218")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    prompts = {}
    for line in open(args.prompts):
        if line.strip():
            p = json.loads(line)
            prompts[p["prompt_id"]] = p
    arms = set(args.arms.split(",")) if args.arms else None
    scorer = ResponseScorer(args.model, args.revision, args.device)
    for run in args.runs:
        by_arm = {}
        for f in sorted(glob.glob(os.path.join(run, "rows_shard*.jsonl"))):
            for line in open(f):
                if line.strip():
                    r = json.loads(line)
                    if arms is None or r["arm"] in arms:
                        by_arm.setdefault(r["arm"], {})[(r["prompt_id"], str(r["seed"]))] = r
        for arm, rows in by_arm.items():
            path = os.path.join(run, f"ppl_{arm}.json")
            done = json.load(open(path))["log_ppl"] if os.path.exists(path) else {}
            for (pid, seed), r in rows.items():
                if seed in done.get(pid, {}):
                    continue
                done.setdefault(pid, {})[seed] = scorer.log_ppl(prompts[pid][args.prompt_field], r.get("completion", ""))
            lp = sorted(v for per in done.values() for v in per.values() if v is not None)
            mean = sum(lp) / len(lp) if lp else None
            median = lp[len(lp) // 2] if lp else None
            with open(path, "w") as f:
                json.dump({"model": args.model, "revision": args.revision, "prompt_field": args.prompt_field, "n": len(lp), "mean": mean,
                           "median": median, "log_ppl": done}, f)
            print(os.path.basename(os.path.normpath(run)), arm, "n", len(lp), "mean", None if mean is None else round(mean, 4),
                  "median", None if median is None else round(median, 4), flush=True)


if __name__ == "__main__":
    main()
