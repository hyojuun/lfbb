"""Builds the WaterBench-600 prompt file: the first 200 prompts of three WaterBench tasks (long-form QA, finance QA, AlpacaFarm), with the
task templates of the WaterBench configuration of Gloaguen et al. (2026), LLaDA's chat template and the input truncated to 150 tokens.

The source files are data/WaterBench/{2-1_longform_qa,2-2_finance_qa,5-1_alpacafarm}.jsonl of the official DLM-KGW code
(https://github.com/eth-sri/diffusion-lm-watermark, commit 380a4bef). With --model Dream-org/Dream-v0-Instruct-7B the file uses Dream's
chat template. The training and development prompt files (data/train_prompts_llada.jsonl, data/dev150_prompts_llada.jsonl) are
distributed with the code.

Example:
  python scripts/prepare_waterbench.py --source path/to/diffusion-lm-watermark/data/WaterBench --out data/waterbench600_llada.jsonl
"""
import argparse
import hashlib
import json
import os

from transformers import AutoTokenizer

TASKS = {"longform_qa": "2-1_longform_qa.jsonl", "finance_qa": "2-2_finance_qa.jsonl", "alpacafarm": "5-1_alpacafarm.jsonl"}


def prompt_text(task, row):
    context, user_input = str(row.get("context", "")), str(row.get("input", ""))
    if task == "longform_qa":
        return f"You are a helpful assistant, please answer the following question within 300 words:\n{context}\n{user_input}"
    if task == "finance_qa":
        return ("You are a helpful assistant, please answer the following question with financial knowledge within 300 words:\n"
                f"{context}\n{user_input}")
    if task == "alpacafarm":
        return f"You are a helpful assistant, please answer the following instruction: \n{context}\n{user_input}"
    raise ValueError(task)


def chat_ids(tok, text, **kw):
    ids = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True, **kw)
    if isinstance(ids, dict) or hasattr(ids, "keys"):
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if len(ids) == 1 and isinstance(ids[0], list):
        ids = ids[0]
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="directory with the WaterBench task files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--rows", default="0:200", help="row range start:end of each task file")
    ap.add_argument("--max-length", type=int, default=150)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, padding_side="left")
    a, b = (int(s) for s in args.rows.split(":"))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for task, name in TASKS.items():
            rows = [json.loads(l) for l in open(os.path.join(args.source, name)) if l.strip()]
            for i, row in enumerate(rows[a:b], start=a):
                h = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
                text = prompt_text(task, row)
                full = chat_ids(tok, text)
                ids = chat_ids(tok, text, truncation=True, max_length=args.max_length)
                f.write(json.dumps({"dataset": task, "prompt_id": f"{task}:{row.get('_id') or h}", "source_line": i + 1, "prompt_text": text,
                                    "prompt_token_ids": ids, "original_prompt_length": len(full), "prompt_truncated": len(full) > len(ids),
                                    "scoring_prompt": tok.decode(ids, skip_special_tokens=True)}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
