"""Builds the code and QA training prompts of the components of Table 3 (Appendix I), disjoint from the evaluation prompts of
scripts/prepare_codeqa.py and in the format of its prompt file (chat template of LLaDA-8B-Instruct with the generation prompt).

  mbpp_train      every MBPP problem (task ids 1-974) that is not evaluated, in task-id order; the MBPP prompt of prepare_codeqa.py.
  mmlupro_train   30 MMLU-Pro test questions per category that are not evaluated: the other questions of the category in file order,
                  shuffled by random.Random(1), the first 30; the MMLU-Pro prompt of prepare_codeqa.py.
  mmlupro4_train  300 further MMLU-Pro questions of the STEM categories (biology, chemistry, computer science, engineering, health,
                  math, physics) that are neither evaluated nor in mmlupro_train: the pool in category order, shuffled by
                  random.Random(2), the first 300. Each is reduced to four choices, the correct one and three distractors drawn by
                  random.Random(7 question_id + 3), which then shuffles the four, and written in the GPQA format of prepare_codeqa.py.
  gsm8k_train     the first --gsm8k-n problems of the GSM8K training split: "Question: ...\\nLet's think step by step\\nAnswer:".

The evaluated prompts are those of --eval-ids (codeqa_ids.json written by prepare_codeqa.py), and the MMLU-Pro file is the one of the
evaluation prompts. --lowconf-ids also writes the ids of the first --lowconf-n prompts of mbpp_train, mmlupro_train and mmlupro4_train,
which are decoded in the low-confidence order as well (scripts/collect_records.py --ids).

Example:
  python scripts/prepare_codeqa_train.py --mbpp raw/mbpp.jsonl --gsm8k raw/gsm8k_train.jsonl --eval-ids data/codeqa/codeqa_ids.json \
      --out data/codeqa/codeqa_train_llada.jsonl --lowconf-ids data/codeqa/codeqa_train_lowconf_ids.json
"""
import argparse
import json
import os
import random
import sys

import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_codeqa import (LETTERS, MMLU_PRO_REVISION, gpqa_content, mbpp_prompt, mmlu_pro_path, mmlu_pro_prompt,  # noqa: E402
                            prompt_row, read_jsonl)

STEM = ("biology", "chemistry", "computer science", "engineering", "health", "math", "physics")


def gsm8k_prompt(r):
    return f"Question: {r['question']}\nLet's think step by step\nAnswer:"


def four_choices(r):
    """(options, letter of the correct one): the correct option and three distractors of an MMLU-Pro question, shuffled."""
    opts = [str(o) for o in list(r["options"])]
    gold = opts[LETTERS.index(r["answer"])]
    wrong = [o for k, o in enumerate(opts) if LETTERS[k] != r["answer"]]
    rnd = random.Random(int(r["question_id"]) * 7 + 3)
    choices = [gold] + rnd.sample(wrong, 3)
    perm = list(range(4))
    rnd.shuffle(perm)
    return [choices[p].strip() for p in perm], LETTERS[perm.index(0)]


def build(mbpp, mmlu_pro, gsm8k, eval_ids, gsm8k_n=300, per_category=30, n_four=300):
    """[(dataset, prompt_id, content, ref)] of the training prompts. per_category and n_four may be None (no cap), to build every
    candidate of a set before keeping the ids of --ids."""
    out = []
    for r in mbpp:
        if f"mbpp:{r['task_id']}" not in eval_ids:
            out.append(("mbpp_train", f"mbpp_train:{r['task_id']}", mbpp_prompt(r), {"task_id": int(r["task_id"])}))
    by_category = {}
    for r in mmlu_pro:
        if f"mmlupro:{r['question_id']}" not in eval_ids:
            by_category.setdefault(r["category"], []).append(r)
    used = set()
    for cat in sorted(by_category):
        questions = by_category[cat][:]
        random.Random(1).shuffle(questions)
        for r in questions[:per_category] if per_category is not None else questions:
            content, n = mmlu_pro_prompt(r, cat)
            out.append(("mmlupro_train", f"mmlupro_train:{r['question_id']}", content, {"gold": r["answer"], "n_choices": n, "category": cat}))
            if per_category is not None:
                used.add(r["question_id"])
    pool = [r for cat in sorted(by_category) if cat in STEM for r in by_category[cat] if r["question_id"] not in used]
    random.Random(2).shuffle(pool)
    for r in (pool[:n_four] if n_four is not None else pool):
        if n_four is None and len(r["options"]) < 4:                 # (candidates only) too few choices for the four-choice format
            continue
        opts, gold = four_choices(r)
        out.append(("mmlupro4_train", f"mmlupro4_train:{r['question_id']}", gpqa_content(r["question"], opts),
                    {"gold": gold, "n_choices": 4, "category": r["category"]}))
    for i, r in enumerate(gsm8k[:gsm8k_n]):
        out.append(("gsm8k_train", f"gsm8k_train:{i}", gsm8k_prompt(r), {"index": i}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mbpp", required=True, help="all MBPP problems in task-id order (JSONL), as given to prepare_codeqa.py")
    ap.add_argument("--mmlu-pro", default="TIGER-Lab/MMLU-Pro", help="MMLU-Pro test parquet file, or the dataset id on the Hub")
    ap.add_argument("--mmlu-pro-revision", default=MMLU_PRO_REVISION, help="dataset revision used with a dataset id")
    ap.add_argument("--gsm8k", required=True, help="GSM8K training split (JSONL with the field question)")
    ap.add_argument("--gsm8k-n", type=int, default=300, help="number of GSM8K problems (the first ones)")
    ap.add_argument("--eval-ids", required=True, help="codeqa_ids.json of prepare_codeqa.py (the evaluated prompts)")
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct", help="tokenizer and chat template")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lowconf-ids", help="also write the ids of the prompts decoded in the low-confidence order")
    ap.add_argument("--lowconf-n", type=int, default=100, help="prompts per code or QA set in --lowconf-ids")
    ap.add_argument("--ids", help="keep exactly these prompt ids (data/codeqa_train_ids.json: the 1,791 prompts of the paper)")
    args = ap.parse_args()

    eval_ids = set(json.load(open(args.eval_ids)))
    mmlu_pro = pd.read_parquet(mmlu_pro_path(args.mmlu_pro, args.mmlu_pro_revision)).to_dict("records")
    keep = json.load(open(args.ids)) if args.ids else None
    if keep is None:
        items = build(read_jsonl(args.mbpp), mmlu_pro, read_jsonl(args.gsm8k), eval_ids, args.gsm8k_n)
    else:
        n_gsm = 1 + max([int(k.split(":", 1)[1]) for k in keep if k.startswith("gsm8k_train:")] or [-1])
        cand = {it[1]: it for it in build(read_jsonl(args.mbpp), mmlu_pro, read_jsonl(args.gsm8k), eval_ids, n_gsm, None, None)}
        missing = [k for k in keep if k not in cand]
        if missing:
            raise SystemExit(f"{len(missing)} ids of --ids are not among the candidates, e.g. {missing[:3]}")
        items = [cand[k] for k in keep]
    if any(pid in eval_ids for _, pid, _, _ in items):
        raise SystemExit("a training prompt id is also an evaluation id")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = [prompt_row(tok, *it) for it in items]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if args.lowconf_ids:
        sets = ("mbpp_train", "mmlupro_train", "mmlupro4_train")
        with open(args.lowconf_ids, "w") as f:
            json.dump([pid for ds in sets for pid in [p for d, p, _, _ in items if d == ds][: args.lowconf_n]], f)
    counts = {}
    for r in rows:
        counts[r["dataset"]] = counts.get(r["dataset"], 0) + 1
    print("prompts", counts, "total", len(rows), "max prompt tokens", max(len(r["prompt_token_ids"]) for r in rows), flush=True)


if __name__ == "__main__":
    main()
