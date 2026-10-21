"""Builds the code and multiple-choice QA prompt file in the format of the WaterBench prompt file: dataset, prompt_id, prompt_text,
prompt_token_ids (chat template of LLaDA-8B-Instruct with the generation prompt), scoring_prompt, and ref, the reference that
scripts/eval_codeqa.py scores against.

  humaneval  all 164 problems: "Complete the following python code:" followed by the problem prompt.
  mbpp       200 of the 500 test problems (task ids 11-510): the test problems in file order, shuffled by random.Random(0), the first
             200, sorted by task id; a zero-shot instruction with the test asserts.
  gpqa       GPQA-Diamond, all 198 questions: the four choices shuffled by random.Random(h mod 2^31), h the first 16 hex digits of the
             SHA-256 of the question; zero-shot chain of thought ending in "The answer is (X)".
  mmlupro    MMLU-Pro test split, 10 questions per category (14 categories, 140): the questions of a category in file order, shuffled
             by random.Random(0), the first 10; the chain-of-thought instruction of the MMLU-Pro evaluation, zero-shot.

The selection depends on the row order of the sources: the HumanEval problem file (human-eval-v2-20210705.jsonl), the complete MBPP file
(mbpp.jsonl, task ids 1-974 in order), both as distributed with OpenCompass; gpqa_diamond.csv of the gated dataset Idavidrein/gpqa; and
the MMLU-Pro test file, a local parquet file or the dataset TIGER-Lab/MMLU-Pro at --mmlu-pro-revision. GPQA must not be redistributed:
the script prints no question text, and the output file is for local use only.

Outputs in --out-dir: codeqa_llada.jsonl, codeqa_ids.json (every prompt id, in file order) and <dataset>_ids.json for each dataset.

Example:
  python scripts/prepare_codeqa.py --humaneval raw/human-eval-v2-20210705.jsonl --mbpp raw/mbpp.jsonl --gpqa raw/gpqa_diamond.csv \
      --out-dir data/codeqa
"""
import argparse
import hashlib
import json
import os
import random

import pandas as pd
from transformers import AutoTokenizer

DATASETS = ("humaneval", "mbpp", "gpqa", "mmlupro")
LETTERS = "ABCDEFGHIJ"
MMLU_PRO_FILE = "data/test-00000-of-00001.parquet"
MMLU_PRO_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"      # dataset revision of the prompts used in the paper


def read_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def humaneval_prompt(r):
    return "Complete the following python code:\n" + r["prompt"]


def mbpp_prompt(r):
    tests = "\n".join(r["test_list"])
    return (f"You are an expert Python programmer, and here is your task: {r['text']} "
            f"Your code should pass these tests:\n\n{tests}\n\nWrite only the Python code.")


def gpqa_content(question, opts):
    """The zero-shot chain-of-thought prompt of a question with four choices, in the GPQA format."""
    return (f"What is the correct answer to this question: {question.strip()}\n\nChoices:\n"
            + "\n".join(f"({LETTERS[k]}) {o}" for k, o in enumerate(opts))
            + "\n\nThink step by step, then finish with \"The answer is (X)\" where X is the letter of the correct choice.")


def gpqa_prompt(r):
    """(prompt id, prompt, letter of the correct choice)."""
    qid = hashlib.sha256(r["Question"].encode()).hexdigest()[:16]
    choices = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
    perm = list(range(4))
    random.Random(int(qid, 16) % (2 ** 31)).shuffle(perm)
    opts = [choices[p].strip() for p in perm]
    return f"gpqa:{qid}", gpqa_content(r["Question"], opts), LETTERS[perm.index(0)]


def mmlu_pro_prompt(r, category):
    """(prompt, number of choices)."""
    opts = [str(o) for o in list(r["options"])]
    content = (f"The following is a multiple choice question about {category}. Think step by step and then finish your answer with "
               f"\"the answer is (X)\" where X is the correct letter choice.\n\nQuestion: {r['question'].strip()}\nOptions:\n"
               + "\n".join(f"{LETTERS[k]}. {o}" for k, o in enumerate(opts)))
    return content, len(opts)


def prompt_row(tok, dataset, prompt_id, content, ref):
    """A row of the prompt file: the content as the user turn of the chat template, with the generation prompt."""
    text = tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
    return {"dataset": dataset, "prompt_id": prompt_id, "prompt_text": content,
            "prompt_token_ids": tok(text, add_special_tokens=False)["input_ids"], "scoring_prompt": content, "ref": ref}


def mmlu_pro_path(source, revision):
    """A local parquet file, a local copy of the dataset repository, or a dataset id on the Hugging Face Hub."""
    if os.path.isfile(source):
        return source
    if os.path.isdir(source):
        return os.path.join(source, MMLU_PRO_FILE)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(source, MMLU_PRO_FILE, repo_type="dataset", revision=revision)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--humaneval", required=True, help="HumanEval problems (JSONL)")
    ap.add_argument("--mbpp", required=True, help="all MBPP problems in task-id order (JSONL)")
    ap.add_argument("--gpqa", required=True, help="gpqa_diamond.csv of the gated dataset Idavidrein/gpqa")
    ap.add_argument("--mmlu-pro", default="TIGER-Lab/MMLU-Pro", help="MMLU-Pro test parquet file, or the dataset id on the Hub")
    ap.add_argument("--mmlu-pro-revision", default=MMLU_PRO_REVISION, help="dataset revision used with a dataset id")
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct", help="tokenizer and chat template")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = []

    def add(dataset, prompt_id, content, ref):
        rows.append(prompt_row(tok, dataset, prompt_id, content, ref))

    for r in read_jsonl(args.humaneval):
        add("humaneval", f"humaneval:{r['task_id']}", humaneval_prompt(r), {"task_id": r["task_id"]})

    mbpp = [r for r in read_jsonl(args.mbpp) if 11 <= int(r["task_id"]) <= 510]
    random.Random(0).shuffle(mbpp)
    for r in sorted(mbpp[:200], key=lambda r_: int(r_["task_id"])):
        add("mbpp", f"mbpp:{r['task_id']}", mbpp_prompt(r), {"task_id": int(r["task_id"])})

    for r in pd.read_csv(args.gpqa).to_dict("records"):
        prompt_id, content, gold = gpqa_prompt(r)
        add("gpqa", prompt_id, content, {"gold": gold, "n_choices": 4})

    by_category = {}
    for r in pd.read_parquet(mmlu_pro_path(args.mmlu_pro, args.mmlu_pro_revision)).to_dict("records"):
        by_category.setdefault(r["category"], []).append(r)
    for cat in sorted(by_category):
        questions = by_category[cat][:]
        random.Random(0).shuffle(questions)
        for r in questions[:10]:
            content, n = mmlu_pro_prompt(r, cat)
            add("mmlupro", f"mmlupro:{r['question_id']}", content, {"gold": r["answer"], "n_choices": n, "category": cat})

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "codeqa_llada.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.out_dir, "codeqa_ids.json"), "w") as f:
        json.dump([r["prompt_id"] for r in rows], f)
    for ds in DATASETS:
        with open(os.path.join(args.out_dir, f"{ds}_ids.json"), "w") as f:
            json.dump([r["prompt_id"] for r in rows if r["dataset"] == ds], f)
    counts = {}
    for r in rows:
        counts[r["dataset"]] = counts.get(r["dataset"], 0) + 1
    print("prompts", counts, "max prompt tokens", max(len(r["prompt_token_ids"]) for r in rows), flush=True)


if __name__ == "__main__":
    main()
