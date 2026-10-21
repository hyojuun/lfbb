"""Renders prompt texts with a model's chat template (generation prompt added), for prompt sets other than WaterBench (e.g.
data/other_tasks_prompts.jsonl: 300 C4 RealNewsLike prefixes, 300 Dolly creative-writing instructions and 300 WritingPrompts).

Example:
  python scripts/render_prompts.py --texts data/other_tasks_prompts.jsonl --out data/other_tasks_llada.jsonl
"""
import argparse
import json

from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True, help="JSONL with dataset, prompt_id, prompt_text and an optional system_prompt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    with open(args.out, "w") as f:
        for l in open(args.texts):
            if not l.strip():
                continue
            r = json.loads(l)
            sysp = r.get("system_prompt") or ""
            msgs = ([{"role": "system", "content": sysp}] if sysp else []) + [{"role": "user", "content": r["prompt_text"]}]
            s = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            ids = tok(s, add_special_tokens=False)["input_ids"]
            f.write(json.dumps(dict(r, prompt_token_ids=ids, scoring_prompt=r["prompt_text"]), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
