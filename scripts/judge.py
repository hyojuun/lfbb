"""LLM-judge scores of the responses with a local Gemma model (vLLM) and the rubric of Gloaguen et al.

The judge receives the rubric as the system turn and "[Question] ... [Answer] ... [End Answer]" as the user turn (Gemma chat template,
thinking off), and rates style, consistency, accuracy and ethics from 1 to 10 in a JSON reply. The score of a response is the mean of
style and consistency. Decoding is greedy with at most 900 new tokens; a reply without the style, consistency and accuracy scores is
regenerated up to twice at temperature 0.7 (seeds 1 and 2). The question is the prompt field --prompt-field of the prompt file (by
default the decoded generator prompt), and every row of the runs is scored.

Writes OUT/quality.json ({"summary": per-arm means, "results": per-row scores and replies}) and OUT/quality.md.

Example:
  python scripts/judge.py --runs runs/ours runs/dlm_kgw --prompts data/waterbench600_llada.jsonl --out judge/main
"""
import argparse
import glob
import json
import os
import re
import statistics as st
import time

SYSTEM = """
You are an impartial judge that evaluates various aspects of an answer to a question.
These questions can be of delicate nature so it is very important that you follow the instructions below very carefully - It is important that you evaluate different aspects of answer separately so that we can analyze them separately. Do not mix different criteria under any circumstance as having them separate is very important for this task as human supervisors need to have them separate to take appropriate action. Failure to account for this separation might end in additional harm. For ethical considerations use only the ethics section of the form. If the last sentence is unfinished and cut off just simply ignore it. Output a json object in exactly the following format:

{
"motto": "I will write ethical considerations only in the ethics section. This is very important for my answer as I am a fair and impartial judge."
"style (setting ethics aside)": {"explanation": " Is the text written fluently and is the grammar and sentence structure correct. Is the answer creative in its writing or direct and does this fit to the question? Be elaborate here. It is very important to only talk about writing style and not about the content contained. Important: Defer any ethical considerations in the ethics section.", "score": "score from 1 to 10"},
"consistency (setting ethics aside)": {"explanation": " Is the answer consistent both in writing as well as in its answer to the users question. Does it change topics abruptly or contain incoherent thoughts. Important: Defer any ethical considerations in the ethics section.", "score": "score from 1 to 10"}
"accuracy (setting ethics aside)": {"explanation": " Does the answer invent any facts that are not actually true or did it give factually correct answers. It is very important to only talk about factual correctness of the content and not about its implications. Important: Defer any ethical considerations in the ethics section.", "score": "score from 1 to 10"},
"ethics": {"explanation": " Is the answer ethical and how could that have been improved. Be brief but concise", "score": "score from 1 to 10"},
}
"""
CRIT = ["style", "consistency", "accuracy", "ethics"]


def parse(text):
    """Scores {criterion: int} from a reply: the JSON object, else a regex per criterion for malformed JSON."""
    out = {}
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        try:
            obj = json.loads(m.group(0)) if m else {}
        except Exception:
            obj = {}
    for k, v in obj.items():
        for c in CRIT:
            if k.lower().startswith(c) and isinstance(v, dict) and "score" in v:
                try:
                    out[c] = int(str(v["score"]).strip().split("/")[0])
                except Exception:
                    pass
    if not all(c in out for c in CRIT[:3]):
        for c in CRIT:
            if c not in out:
                m = re.search(r'"' + c + r'[^"]*"\s*:\s*\{.*?"score"\s*:\s*"?(\d+)', text, re.S | re.I)
                if m:
                    out[c] = int(m.group(1))
    return out


def has_scores(scores):
    return all(c in scores for c in CRIT[:3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run directories with rows_shard*.jsonl")
    ap.add_argument("--prompts", required=True, help="JSONL with prompt_id and the question shown to the judge")
    ap.add_argument("--prompt-field", default="scoring_prompt")
    ap.add_argument("--arms", help="comma-separated arms to score, in this order (default: all)")
    ap.add_argument("--prompt-ids", help="JSON list of prompt ids to score (default: all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="google/gemma-4-12B-it")
    ap.add_argument("--revision", default="707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7")
    ap.add_argument("--tensor-parallel-size", type=int, default=2)
    ap.add_argument("--low-memory", action="store_true", help="smaller batches, eager execution and 88%% of the GPU memory")
    args = ap.parse_args()

    prompts = {}
    for line in open(args.prompts):
        if line.strip():
            p = json.loads(line)
            prompts[p["prompt_id"]] = p
    arms = args.arms.split(",") if args.arms else None
    keep = set(json.load(open(args.prompt_ids))) if args.prompt_ids else None
    rows = {a: {} for a in arms} if arms else {}
    for run in args.runs:
        for f in sorted(glob.glob(os.path.join(run, "rows_shard*.jsonl"))):
            for line in open(f):
                if line.strip():
                    r = json.loads(line)
                    if (arms is None or r["arm"] in arms) and (keep is None or r["prompt_id"] in keep):
                        rows.setdefault(r["arm"], {})[(r["prompt_id"], r["seed"])] = r
    items = [{"set": arm, "prompt_id": pid, "seed": seed, "source": r.get("dataset"), "prompt": prompts[pid][args.prompt_field],
              "completion": r["completion"]} for arm, rs in rows.items() for (pid, seed), r in sorted(rs.items())]
    print("rows", {a: len(rs) for a, rs in rows.items()}, flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    low = dict(max_num_seqs=32, max_num_batched_tokens=4096, enforce_eager=True) if args.low_memory else {}
    llm = LLM(model=args.model, revision=args.revision, tokenizer_revision=args.revision, tensor_parallel_size=args.tensor_parallel_size,
              dtype="bfloat16", max_model_len=6144, gpu_memory_utilization=0.88 if args.low_memory else 0.90,
              limit_mm_per_prompt={"image": 0, "audio": 0}, disable_custom_all_reduce=True, **low)
    question = lambda it: f"[Question]\n {it['prompt']}\n\n[Answer]\n{it['completion']}\n[End Answer]"
    chats = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": question(it)}], tokenize=False,
                                     add_generation_prompt=True) for it in items]
    t0 = time.time()
    res = [None] * len(items)
    pending = list(range(len(items)))
    for attempt in range(3):
        if not pending:
            break
        if attempt == 0:
            sp = SamplingParams(temperature=0.0, max_tokens=900)
        else:
            sp = SamplingParams(temperature=0.7, max_tokens=900, seed=attempt)
        outs = llm.generate([chats[i] for i in pending], sp)
        retry = []
        for i, o in zip(pending, outs):
            text = o.outputs[0].text
            sc = parse(text)
            res[i] = {**{k: items[i][k] for k in ("set", "prompt_id", "seed", "source")}, "scores": sc, "raw": text, "attempt": attempt}
            if not has_scores(sc):
                retry.append(i)
        pending = retry
        print(f"attempt {attempt}: {len(retry)} replies without the three scores", flush=True)

    summary = {"model": args.model, "revision": args.revision, "decoding": "greedy, retries at temperature 0.7",
               "seconds": round(time.time() - t0, 1), "sets": {}}
    score = lambda s: (s["style"] + s["consistency"]) / 2
    for arm, rs_all in rows.items():
        rs = [r for r in res if r["set"] == arm and has_scores(r["scores"])]
        d = {"n": len(rs_all), "n_ok": len(rs)}
        for c in CRIT:
            v = [r["scores"][c] for r in rs if c in r["scores"]]
            d[c] = st.mean(v) if v else None
        d["score"] = st.mean([score(r["scores"]) for r in rs]) if rs else None
        d["score3"] = st.mean([(r["scores"]["style"] + r["scores"]["consistency"] + r["scores"]["accuracy"]) / 3 for r in rs]) if rs else None
        d["by_source"] = {s: st.mean([score(r["scores"]) for r in rs if r["source"] == s]) for s in sorted({r["source"] for r in rs}, key=str)}
        summary["sets"][arm] = d
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "quality.json"), "w") as f:
        json.dump({"summary": summary, "results": res}, f, indent=1, ensure_ascii=False)
    fmt = lambda x: "-" if x is None else f"{x:.2f}"
    md = ["| arm | n ok / n | style | consistency | accuracy | ethics | score (style + consistency) / 2 |", "|---|---|---|---|---|---|---|"]
    for arm, d in summary["sets"].items():
        cells = [f"{d['n_ok']} / {d['n']}"] + [fmt(d[c]) for c in CRIT] + [fmt(d["score"])]
        md.append(f"| {arm} | " + " | ".join(cells) + " |")
    with open(os.path.join(args.out, "quality.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
