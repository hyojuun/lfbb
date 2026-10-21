"""Quality and detection on the code and QA prompts of scripts/prepare_codeqa.py, for our method and the baselines.

Quality: pass@1 on HumanEval and MBPP, accuracy on GPQA and MMLU-Pro.
  Code: the longest fenced code block of the response (else the text between [BEGIN] and [DONE], else the rest of an unterminated fence,
  else the whole response), without top-level assert and print lines; for HumanEval the problem prompt is prepended unless the code
  defines the entry point. The program and the tests run in a fresh Python process with a 2 GB address-space limit and a 6 s timeout.
  QA: the last "answer is (X)", else the last "Answer: X", else the last standalone choice letter (the MMLU-Pro answer parser).

Detection: TPR at empirical false-positive rates of 1% and 0.1%. The null scores are the unwatermarked responses of the same prompts
(--none) scored by each method's detector, under --num-keys keys (42, then 1, 2, ... skipping 42) for the keyed detectors, pooled. The
threshold for a rate alpha is the max(1, floor(alpha N))-th largest of the N pooled scores, and a response is detected when its z-score
is strictly above it (a missing z-score is not detected).
  ours     the response is re-tokenized and detected with the partition under the key of its row (lfbb.detector.detect, as in
           scripts/generate.py); null: the re-tokenized unwatermarked responses (fewer than two tokens: no score).
  dlm_kgw  the z-score of the official detector stored in the row; null: the official green list (OnTheFlyGreenlist, Bernoulli with
           gamma 0.25, seeded by the key) counted as the official detector does (context = the previous token, every distinct
           (previous token, token) pair once), binomial z-score of the generated token ids of the unwatermarked responses.
  dgmark   keyless: the official calculate_watermark_stats of the generated token ids followed by the end-of-text token, binomial
           z-score at 1/2 over the counted tokens, for the watermarked and the unwatermarked responses alike.

Inputs are directories of rows_shard*.jsonl files or single JSONL files, with one response per prompt (a later row of a prompt replaces
an earlier one). DIR:ARM keeps the rows of one arm, which is needed when DIR holds several arms.
  --none  unwatermarked responses: prompt_id, response or completion, and eff_ids (the generated token ids before the first end-of-text
          or padding token), which the dlm_kgw and dgmark nulls need. In the paper they come from the decoding of the official DLM-KGW
          code without the watermark, for every method.
  --run NAME=KIND:DIR[:ARM], one per evaluated set:
          ours     rows of scripts/generate.py (prompt_id, arm, key, completion, detection).
          dlm_kgw  prompt_id, response or completion, and z (or detection.z), the z-score of the official detector.
          dgmark   prompt_id, response or completion, and gen_ids (the generated ids after the prompt) with tokens (the number of
                   them before the first end-of-text or padding token), or eff_ids; prompt_len (default: the prompt length of --prompts).

Output JSON: {"n_none", "sets": {"none": {"quality"}, NAME: {"kind", "n", "thr1", "thr01", "per_task", "quality", "z"}}} with
per_task[dataset] = {"n", "tpr1", "tpr01", "quality", "quality_none", "quality_diff"} in %, where quality_none is the quality of the
unwatermarked responses of the same prompts and quality_diff the paired difference, and quality[prompt_id] = [passed, outcome].

Example:
  python scripts/eval_codeqa.py --prompts data/codeqa/codeqa_llada.jsonl --humaneval raw/human-eval-v2-20210705.jsonl \
      --mbpp raw/mbpp.jsonl --none runs/codeqa/none --partition checkpoints/bands.pt --run ours_d3=ours:runs/codeqa/ours:ours_d3 \
      --run dlm_kgw_d4=dlm_kgw:runs/codeqa/dlm_kgw --official-dlm-kgw path/to/diffusion-lm-watermark --out results/codeqa.json
"""
import argparse
import glob
import importlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lfbb import detect, load_partition, load_special_ids  # noqa: E402
from lfbb.constants import EOT_ID  # noqa: E402

DATASETS = ("humaneval", "mbpp", "gpqa", "mmlupro")
KINDS = ("ours", "dlm_kgw", "dgmark")
TIMEOUT = 6.0                   # seconds per program
GREEN_LIST_SIZE = 126349        # hash and vocabulary size of the official DLM-KGW green list for LLaDA
GAMMA = 0.25                    # green fraction of the official DLM-KGW configuration

# Executed by a fresh interpreter: limits the address space to 2 GB, then runs the program read from stdin.
RUNNER = r"""
import resource, sys
try:
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 ** 3, 2 * 1024 ** 3))
except (ValueError, OSError):
    pass
src = sys.stdin.read()
exec(compile(src, "<prog>", "exec"), {"__name__": "__main__"})
"""


# ---------------------------------------------------------------------------------------------------------------------------- quality
def extract_code(text):
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    if m:
        return max(m, key=len)
    m = re.search(r"\[BEGIN\](.*?)(\[DONE\]|$)", text, flags=re.S)
    if m:
        return m.group(1).strip().strip("'")
    m = re.search(r"```(?:python|py)?\s*\n(.*)$", text, flags=re.S)       # unterminated fence
    if m:
        return m.group(1)
    return text


def strip_top_level_asserts(code):
    """Drops top-level assert and print lines (responses often repeat the asserts of the prompt), so that the tests decide."""
    return "\n".join(line for line in code.split("\n") if not (line.startswith("assert ") or line.startswith("print(")))


def run_program(src, timeout=TIMEOUT):
    """Runs src in a fresh Python process. Returns (passed, outcome), outcome in pass, syntax, timeout, assert, name, runtime."""
    try:
        compile(src, "<prog>", "exec")
    except Exception:
        return False, "syntax"
    with tempfile.TemporaryDirectory() as d:
        try:
            p = subprocess.run([sys.executable, "-c", RUNNER], input=src, capture_output=True, text=True, timeout=timeout, cwd=d)
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if p.returncode == 0:
        return True, "pass"
    err = p.stderr.strip().splitlines()[-1] if p.stderr.strip() else ""
    if err.startswith("AssertionError"):
        return False, "assert"
    if err.startswith("NameError"):
        return False, "name"
    return False, "runtime"


def humaneval_check(problem, text):
    code = strip_top_level_asserts(extract_code(text))
    body = code if ("def " + problem["entry_point"]) in code else problem["prompt"] + code
    return run_program(body + "\n\n" + problem["test"] + f"\n\ncheck({problem['entry_point']})\n")


def mbpp_check(problem, text):
    code = strip_top_level_asserts(extract_code(text))
    return run_program((problem.get("test_setup_code") or "") + "\n" + code + "\n\n" + "\n".join(problem["test_list"]) + "\n")


def extract_choice(text, n):
    letters = "ABCDEFGHIJ"[:n]
    for pattern in (rf"answer is \(?([{letters}])\)?", rf"[aA]nswer:\s*\(?([{letters}])\)?", rf"\b([{letters}])\b"):
        m = re.findall(pattern, text)
        if m:
            return m[-1]
    return None


def text_of(r):
    return r["response"] if r.get("response") is not None else r.get("completion", "")


def score_quality(rows, prompts, problems, workers):
    """{prompt_id: (passed, outcome)}; the programs run in a thread pool, each in its own process."""
    res, jobs = {}, []
    for pid, r in rows.items():
        p, text = prompts[pid], text_of(r)
        if p["dataset"] == "humaneval":
            jobs.append((pid, humaneval_check, problems["humaneval"][p["ref"]["task_id"]], text))
        elif p["dataset"] == "mbpp":
            jobs.append((pid, mbpp_check, problems["mbpp"][int(p["ref"]["task_id"])], text))
        else:
            pred = extract_choice(text, p["ref"]["n_choices"])
            res[pid] = (pred is not None and pred == p["ref"]["gold"], "parsed" if pred else "unparsed")
    with ThreadPoolExecutor(workers) as ex:
        for (pid, *_), v in zip(jobs, ex.map(lambda j: j[1](j[2], j[3]), jobs)):
            res[pid] = v
    return res


# -------------------------------------------------------------------------------------------------------------------------- detectors
def same(a, b):
    return (a != a and b != b) or abs(a - b) <= 1e-6


def ours_null(part, tok, none_rows, keys):
    ids = [tok(text_of(r), add_special_tokens=False)["input_ids"] for r in none_rows.values()]
    zs = []
    for key in keys:
        part.key = int(key)
        for t in ids:
            if len(t) < 2:
                zs.append(float("nan"))
                continue
            n, k, z, p = detect(part, t)
            zs.append(z if n else float("nan"))
    return zs


def dlm_kgw_z(green_list, ids):
    """z-score of the official DLM-KGW detector: context = the previous token, every distinct (previous token, token) pair once."""
    if len(ids) < 2:
        return float("nan")
    t = torch.tensor(ids, dtype=torch.long)
    h, tk = t[:-1], t[1:]
    scores = green_list.lookup(h[None], tk[None])[0].tolist()
    seen, s = set(), []
    for a, b, x in zip(h.tolist(), tk.tolist(), scores):
        if (a, b) not in seen:
            s.append(x)
        seen.add((a, b))
    n, k = len(s), sum(s)
    return (k - GAMMA * n) / math.sqrt(n * GAMMA * (1 - GAMMA)) if n else float("nan")


def dlm_kgw_null(green_list_cls, none_rows, keys):
    zs = []
    for key in keys:
        gl = green_list_cls(hash_size=GREEN_LIST_SIZE, vocab_size=GREEN_LIST_SIZE, mode="bernoulli", distrib_params={"gamma": GAMMA},
                            seed=key, device="cpu", dtype=torch.float32)
        zs += [dlm_kgw_z(gl, r["eff_ids"]) for r in none_rows.values()]
    return zs


def dgmark_z(stats, ids, prompt_len):
    m, _, _, n = stats(list(ids) + [EOT_ID], prompt_len, private_key=None)
    return (m - n / 2) / math.sqrt(n / 4) if n else float("nan")


def dgmark_ids(r):
    if "gen_ids" in r:
        return r["gen_ids"][: r.get("tokens", len(r["gen_ids"]))] if r["gen_ids"] else []
    if "eff_ids" in r:
        return r["eff_ids"]
    raise SystemExit(f"dgmark row {r['prompt_id']} has neither gen_ids nor eff_ids")


def stored_z(r):
    z = r.get("z")
    if z is None:
        z = (r.get("detection") or {}).get("z")
    return float("nan") if z is None else z


def threshold(null, alpha):
    """The max(1, floor(alpha N))-th largest of the N scores that are not missing."""
    v = np.sort(np.asarray([x for x in null if x == x]))[::-1]
    return float(v[max(0, int(math.floor(alpha * len(v))) - 1)])


def official(root, module, attr, flag):
    """attr of a module of an official code checkout; its src/ directory goes first on the path."""
    if not root:
        raise SystemExit(f"{flag} is needed")
    src = os.path.abspath(os.path.join(root, "src"))
    sys.path.insert(0, src)
    mod = importlib.import_module(module)
    if not os.path.abspath(mod.__file__).startswith(src + os.sep):
        raise SystemExit(f"{module} was imported from {mod.__file__}, not from {src}")
    return getattr(mod, attr)


# ------------------------------------------------------------------------------------------------------------------------------ rows
def read_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def load_rows(spec):
    """{prompt_id: row} of DIR or DIR:ARM."""
    path, arm = spec, None
    if not os.path.exists(spec) and ":" in spec:
        path, arm = spec.rsplit(":", 1)
    files = sorted(glob.glob(os.path.join(path, "rows_shard*.jsonl"))) if os.path.isdir(path) else [path]
    rows, arms = {}, set()
    for f in files:
        for r in read_jsonl(f):
            if arm is None or r.get("arm") == arm:
                rows[r["prompt_id"]] = r
                arms.add(r.get("arm"))
    if arm is None and len(arms) > 1:
        raise SystemExit(f"{path} holds the arms {sorted(map(str, arms))}: give one as {path}:ARM")
    if not rows:
        raise SystemExit(f"no rows in {spec}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="prompt file of scripts/prepare_codeqa.py")
    ap.add_argument("--humaneval", required=True, help="HumanEval problems (JSONL), as given to prepare_codeqa.py")
    ap.add_argument("--mbpp", required=True, help="MBPP problems (JSONL), as given to prepare_codeqa.py")
    ap.add_argument("--none", required=True, metavar="DIR[:ARM]", help="unwatermarked responses of the same prompts")
    ap.add_argument("--run", action="append", default=[], metavar="NAME=KIND:DIR[:ARM]",
                    help="an evaluated set, KIND one of ours, dlm_kgw, dgmark (repeatable)")
    ap.add_argument("--partition", help="learned band table (.pt) or 'hash', for the ours sets")
    ap.add_argument("--special-ids", default=os.path.join(os.path.dirname(__file__), "..", "data", "special_ids_llada.json"))
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct", help="tokenizer used to re-tokenize the responses")
    ap.add_argument("--official-dlm-kgw", help="root of a checkout of the official DLM-KGW code (commit 380a4bef), for the dlm_kgw sets")
    ap.add_argument("--official-dgmark", help="root of a checkout of the official dgMARK code (commit cb6d79ff), for the dgmark sets")
    ap.add_argument("--num-keys", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8, help="programs executed in parallel")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs = []
    for spec in args.run:
        name, _, rest = spec.partition("=")
        kind, _, path = rest.partition(":")
        if not name or kind not in KINDS or not path:
            ap.error(f"--run {spec}: expected NAME=KIND:DIR[:ARM] with KIND in {KINDS}")
        runs.append((name, kind, load_rows(path)))
    kinds = {kind for _, kind, _ in runs}
    prompts = {r["prompt_id"]: r for r in read_jsonl(args.prompts)}
    problems = {"humaneval": {r["task_id"]: r for r in read_jsonl(args.humaneval)},
                "mbpp": {int(r["task_id"]): r for r in read_jsonl(args.mbpp)}}
    keys = [42] + [k for k in range(1, args.num_keys + 1) if k != 42][: args.num_keys - 1]
    none = load_rows(args.none)
    if kinds & {"dlm_kgw", "dgmark"} and any("eff_ids" not in r for r in none.values()):
        raise SystemExit("the dlm_kgw and dgmark nulls need eff_ids in every --none row")

    qn = score_quality(none, prompts, problems, args.workers)
    res = {"n_none": len(none), "sets": {"none": {"quality": {k: list(v) for k, v in qn.items()}}}}
    nulls = {}
    if "ours" in kinds:
        if not args.partition:
            ap.error("--partition is needed for the ours sets")
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        part = load_partition(args.partition, special_ids=load_special_ids(args.special_ids), device=args.device)
        default_key = part.key
        nulls["ours"] = ours_null(part, tok, none, keys)
    if "dlm_kgw" in kinds:
        green_list_cls = official(args.official_dlm_kgw, "dlm_watermark.watermarks.random_utils", "OnTheFlyGreenlist", "--official-dlm-kgw")
        nulls["dlm_kgw"] = dlm_kgw_null(green_list_cls, none, keys)
    if "dgmark" in kinds:
        stats = official(args.official_dgmark, "utils", "calculate_watermark_stats", "--official-dgmark")
        nulls["dgmark"] = [dgmark_z(stats, r["eff_ids"], len(prompts[pid]["prompt_token_ids"])) for pid, r in none.items()]

    for name, kind, rows in runs:
        q = score_quality(rows, prompts, problems, args.workers)
        if kind == "ours":
            z, differ = {}, 0
            for pid, r in rows.items():
                part.key = int(r.get("key", default_key))
                z[pid] = detect(part, tok(text_of(r), add_special_tokens=False)["input_ids"])[2]
                stored = (r.get("detection") or {}).get("z")
                differ += stored is not None and not same(stored, z[pid])
            if differ:
                print(f"{name}: the stored detection of {differ} rows differs from the re-detection (another partition?)", flush=True)
        elif kind == "dlm_kgw":
            z = {pid: stored_z(r) for pid, r in rows.items()}
        else:
            z = {pid: dgmark_z(stats, dgmark_ids(r), r.get("prompt_len", len(prompts[pid]["prompt_token_ids"]))) for pid, r in rows.items()}
        t1, t01 = threshold(nulls[kind], 0.01), threshold(nulls[kind], 0.001)
        per = {}
        for ds in DATASETS:
            ids = [pid for pid in rows if prompts[pid]["dataset"] == ds]
            if not ids:
                continue
            zz = np.array([z[i] if z[i] == z[i] else -1e9 for i in ids])
            paired = [i for i in ids if i in qn]
            per[ds] = {"n": len(ids), "tpr1": round(100 * float((zz > t1).mean()), 2), "tpr01": round(100 * float((zz > t01).mean()), 2),
                       "quality": round(100 * float(np.mean([bool(q[i][0]) for i in ids])), 2),
                       "quality_none": round(100 * float(np.mean([bool(qn[i][0]) for i in paired])), 2),
                       "quality_diff": round(100 * float(np.mean([bool(q[i][0]) - bool(qn[i][0]) for i in paired])), 2)}
        res["sets"][name] = {"kind": kind, "n": len(rows), "thr1": t1, "thr01": t01, "per_task": per,
                             "quality": {k: list(v) for k, v in q.items()}, "z": z}
        print(name, json.dumps(per), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
