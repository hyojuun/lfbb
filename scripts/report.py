"""Metrics of the runs per arm, and paired differences between arms with bootstrap confidence intervals.

Per arm, over the prompts that every reported arm has:
  z        mean z-score (a missing z-score counts as --missing-z, -9 by default);
  tpr1     TPR at the calibrated threshold for a false-positive rate of 1% (z strictly above the threshold), tpr01 the same at 0.1%;
  ntpr1    nominal TPR at 1%: p-value <= 0.01 (a missing p-value counts as 1), ntpr01 at 0.1% and tpr1e6 at 1e-6 the same;
  log_ppl  mean log-perplexity of the scored responses (RUN_DIR/ppl_<arm>.json written by score_ppl.py);
  kl       mean KL divergence of the tilt, summed over the commits of a response;
  rep      share of responses with a repetition loop: more than 15% of their word 4-grams repeat an earlier one;
  judge    mean judge score, (style + consistency) / 2 (optional, quality.json written by judge.py);
  valid    with --validity TOKENIZER, the share of valid responses under the validity rule of DLM-KGW: at least 192 characters and
           at most half of the token bigrams repeated (1 - distinct / all <= 0.5, tokens of TOKENIZER without special tokens);
           lp_valid the mean log-perplexity of the valid responses (Table 4, with the Dream tokenizer).
TPRs and shares are in %. The values of a prompt are averaged over its seeds. A difference A - B is paired over prompts, with a 95%
interval from 4,000 bootstrap resamples of the prompts (numpy default_rng(0)); for lp_valid, over the prompts valid in both arms.

Thresholds: --thresholds [PREFIX=]T1,T01 or [PREFIX=]CALIB.json (calibrate.py output, its pooled q99 and q999), repeatable. An arm
uses the entry with the longest prefix of its name; an entry without a prefix applies to every arm. Arms without thresholds get no
calibrated TPR. --datasets restricts the report to the rows of some datasets (per task), with the same thresholds.

Example (runs/hash_ours_d3_d4 decoded with generate.py --partition hash --arm-prefix hash_):
  python scripts/report.py --runs runs/ours_d3_d4 runs/hash_ours_d3_d4 --thresholds calib/bands.json --thresholds hash_=calib/hash.json \
      --judge judge/main/quality.json --contrast ours_d3:hash_ours_d3 --out reports/main.json
"""
import argparse
import glob
import json
import os
import warnings

import numpy as np

METRICS = ("z", "tpr1", "tpr01", "ntpr1", "ntpr01", "tpr1e6", "lp", "kl", "rep", "judge", "valid", "lp_valid")


def loop(text, n=4, thr=0.15):
    """1 if more than thr of the word n-grams of the text repeat an earlier one, else 0."""
    w = text.split()
    g = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    if not g:
        return 0.0
    seen, rep = set(), 0
    for x in g:
        rep += x in seen
        seen.add(x)
    return float(rep / len(g) > thr)


def validity_rule(model):
    """valid(text) under the rule of DLM-KGW: at least 192 characters and at most half of the token bigrams repeated."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    def valid(text):
        w = tok(text, add_special_tokens=False)["input_ids"]
        g = list(zip(w, w[1:]))
        return len(text) >= 192 and (1 - len(set(g)) / len(g) if g else 0.0) <= 0.5
    return valid


def num(x, missing=float("nan")):
    return missing if x is None or x != x else float(x)


def load_runs(dirs, datasets=None):
    """rows[arm][(prompt_id, seed)] and log-PPL ppl[arm][(prompt_id, seed)] of the run directories (later rows replace earlier ones),
    optionally only the rows of some datasets."""
    rows, ppl = {}, {}
    for d in dirs:
        arms = set()
        for f in sorted(glob.glob(os.path.join(d, "rows_shard*.jsonl"))):
            for line in open(f):
                if line.strip():
                    r = json.loads(line)
                    if datasets and r.get("dataset") not in datasets:
                        continue
                    rows.setdefault(r["arm"], {})[(r["prompt_id"], str(r["seed"]))] = r
                    arms.add(r["arm"])
        for arm in arms:
            pf = os.path.join(d, f"ppl_{arm}.json")
            if os.path.exists(pf):
                for pid, per in json.load(open(pf))["log_ppl"].items():
                    for seed, v in per.items():
                        ppl.setdefault(arm, {})[(pid, seed)] = v
    return rows, ppl


def load_judge(paths):
    out = {}
    for path in paths:
        for r in json.load(open(path))["results"]:
            s = r["scores"]
            v = (s["style"] + s["consistency"]) / 2 if all(c in s for c in ("style", "consistency", "accuracy")) else None
            out.setdefault(r["set"], {})[(r["prompt_id"], str(r["seed"]))] = v
    return out


def parse_thresholds(specs):
    table = []
    for spec in specs:
        prefix, val = spec.split("=", 1) if "=" in spec else ("", spec)
        if val.endswith(".json"):
            pooled = json.load(open(val))["pooled"]
            table.append((prefix, (float(pooled["q99"]), float(pooled["q999"]))))
        else:
            t1, t01 = val.split(",")
            table.append((prefix, (float(t1), float(t01))))
    return table


def thresholds_of(arm, table):
    match = [(len(p), t) for p, t in table if arm.startswith(p)]
    return max(match, key=lambda m: m[0])[1] if match else None


def per_prompt(values, ids):
    """Mean over the seeds of each prompt (NaN-aware) -> array over ids."""
    by = {}
    for (pid, _), v in values.items():
        by.setdefault(pid, []).append(v)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.array([np.nanmean(np.array(by[i], dtype=float)) for i in ids], dtype=float)


def summary(x, digits, scale=1.0):
    ok = np.isfinite(x)
    return round(float(x[ok].mean() * scale), digits) if ok.any() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run directories with rows_shard*.jsonl (and ppl_<arm>.json)")
    ap.add_argument("--arms", help="comma-separated arms to report, in this order (default: all, in order of appearance)")
    ap.add_argument("--datasets", help="comma-separated datasets (field 'dataset' of the rows) to report (default: all)")
    ap.add_argument("--thresholds", action="append", default=[], help="[PREFIX=]T1,T01 or [PREFIX=]CALIB.json (repeatable)")
    ap.add_argument("--judge", nargs="*", default=[], help="quality.json files written by judge.py")
    ap.add_argument("--contrast", action="append", default=[], help="A:B for A - B (repeatable; default: every arm - the first arm)")
    ap.add_argument("--interaction", action="append", default=[], help="A:B:C:D for (A - B) - (C - D) (repeatable)")
    ap.add_argument("--missing-z", type=float, default=-9.0,
                    help="z-score of a response without a counted unit (the paper: -9 on LLaDA, 0 on Dream)")
    ap.add_argument("--validity", metavar="TOKENIZER", help="validity rule of DLM-KGW with this tokenizer (Valid %%, log-PPL of valid)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows, ppl = load_runs(args.runs, set(args.datasets.split(",")) if args.datasets else None)
    judge = load_judge(args.judge)
    arms = args.arms.split(",") if args.arms else list(rows)
    missing = [a for a in arms if a not in rows]
    if not arms or missing:
        raise SystemExit(f"no rows for the arms {missing}" if missing else "no rows found")
    table = parse_thresholds(args.thresholds)
    valid = validity_rule(args.validity) if args.validity else None
    ids = sorted(set.intersection(*[{pid for pid, _ in rows[a]} for a in arms]))
    if not ids:
        raise SystemExit("no prompt is shared by all arms")
    res = {"n_prompts": len(ids), "n_rows": {a: len(rows[a]) for a in arms}, "thresholds": {}, "arms": {}, "contrasts": {}}
    V = {}
    for arm in arms:
        R = rows[arm]
        thr = thresholds_of(arm, table)
        res["thresholds"][arm] = list(thr) if thr else None
        z = {k: num(r["detection"]["z"], args.missing_z) for k, r in R.items()}
        p = {k: num(r["detection"]["p"], 1.0) for k, r in R.items()}
        rep = per_prompt({k: loop(r.get("completion", "")) for k, r in R.items()}, ids)
        ok = {k: valid(r.get("completion", "")) for k, r in R.items()} if valid else None
        v = {"z": per_prompt(z, ids),
             "tpr1": per_prompt({k: (s > thr[0]) * 100.0 for k, s in z.items()}, ids) if thr else None,
             "tpr01": per_prompt({k: (s > thr[1]) * 100.0 for k, s in z.items()}, ids) if thr else None,
             "ntpr1": per_prompt({k: (s <= 0.01) * 100.0 for k, s in p.items()}, ids),
             "ntpr01": per_prompt({k: (s <= 0.001) * 100.0 for k, s in p.items()}, ids),
             "tpr1e6": per_prompt({k: (s <= 1e-6) * 100.0 for k, s in p.items()}, ids),
             "lp": per_prompt({k: num(ppl.get(arm, {}).get(k)) for k in R}, ids),
             "kl": per_prompt({k: num(r.get("kl")) for k, r in R.items()}, ids),
             "rep": rep * 100,
             "judge": per_prompt({k: num(judge.get(arm, {}).get(k)) for k in R}, ids) if arm in judge else None,
             "valid": per_prompt({k: x * 100.0 for k, x in ok.items()}, ids) if valid else None,
             "lp_valid": per_prompt({k: num(ppl.get(arm, {}).get(k)) if ok[k] else float("nan") for k in R}, ids) if valid else None}
        V[arm] = v
        res["arms"][arm] = {"z": summary(v["z"], 3), "tpr1": summary(v["tpr1"], 2) if thr else None,
                            "tpr01": summary(v["tpr01"], 2) if thr else None, "ntpr1": summary(v["ntpr1"], 2),
                            "ntpr01": summary(v["ntpr01"], 2), "tpr1e6": summary(v["tpr1e6"], 2),
                            "log_ppl": summary(v["lp"], 4), "n_ppl": int(np.isfinite(v["lp"]).sum()), "kl": summary(v["kl"], 2),
                            "rep": summary(rep, 1, scale=100.0),
                            "judge": summary(v["judge"], 3) if v["judge"] is not None else None,
                            "n_judge": int(np.isfinite(v["judge"]).sum()) if v["judge"] is not None else 0}
        if valid:
            shared = set(ids)
            res["arms"][arm].update({"valid": summary(v["valid"], 2), "log_ppl_valid": summary(v["lp_valid"], 4),
                                     "n_valid": int(sum(x for (pid, _), x in ok.items() if pid in shared))})

    B = np.random.default_rng(0).integers(0, len(ids), (4000, len(ids)))

    def contrast(a, b, c=None, d=None):
        out = {}
        for k in METRICS:
            parts = [V[x][k] for x in ((a, b) if c is None else (a, b, c, d))]
            if any(x is None for x in parts):
                continue
            x = parts[0] - parts[1] if c is None else (parts[0] - parts[1]) - (parts[2] - parts[3])
            x = np.where(np.isfinite(x), x, np.nan)
            if not np.isfinite(x).any():
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                m, bs = float(np.nanmean(x)), np.nanmean(x[B], 1)
            out[k] = [round(m, 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
        return out

    pairs = [c.split(":") for c in args.contrast] or [(a, arms[0]) for a in arms[1:]]
    for a, b in pairs:
        res["contrasts"][f"{a} - {b}"] = contrast(a, b)
    for spec in args.interaction:
        a, b, c, d = spec.split(":")
        res["contrasts"][f"({a} - {b}) - ({c} - {d})"] = contrast(a, b, c, d)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)

    fmt = lambda x, w, d: f"{'-':>{w}}" if x is None else f"{x:>{w}.{d}f}"
    width = max(len(a) for a in arms)
    print(f"{res['n_prompts']} prompts")
    print(f"{'arm':<{width}}  {'z':>6} {'TPR@1%':>7} {'TPR@.1%':>7} {'nTPR@1%':>7} {'nTPR@.1%':>8} {'TPR@1e-6':>8} {'log-PPL':>8} "
          f"{'KL':>7} {'Rep.%':>6} {'judge':>6}")
    for a, m in res["arms"].items():
        print(f"{a:<{width}}  {fmt(m['z'], 6, 3)} {fmt(m['tpr1'], 7, 1)} {fmt(m['tpr01'], 7, 1)} {fmt(m['ntpr1'], 7, 1)} "
              f"{fmt(m['ntpr01'], 8, 1)} {fmt(m['tpr1e6'], 8, 1)} {fmt(m['log_ppl'], 8, 3)} {fmt(m['kl'], 7, 2)} {fmt(m['rep'], 6, 1)} "
              f"{fmt(m['judge'], 6, 2)}")
    if valid:
        print(f"{'arm':<{width}}  {'Valid%':>6} {'log-PPL (valid)':>15}")
        for a, m in res["arms"].items():
            print(f"{a:<{width}}  {fmt(m['valid'], 6, 1)} {fmt(m['log_ppl_valid'], 15, 3)}")
    for name, c in res["contrasts"].items():
        print(name)
        for k, (m, lo, hi) in c.items():
            print(f"  {k:<7} {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
