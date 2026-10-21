"""Selects the watermark configuration on the development prompts (data/dev150_prompts_llada.jsonl), never on evaluation prompts.

Candidate arms are named <family>_dp<delta'> (configs/arms/dev_candidates.json, configs/arms/dev_candidates_b8.json); each family changes
one setting of the reference family `base` (lambda 0.5) or of `lam1` (lambda 1) and is decoded at several strengths with the same seeds.
Rule, fixed before any decoding:
  primary     mean z-score gain over the reference at equal KL: 5 equally spaced KL values inside both families' KL ranges (linear
              interpolation over delta'), 95% percentile interval from 4,000 bootstrap resamples of the prompts (numpy default_rng(0));
  guardrails  at the same 5 KL values: judge difference >= -0.10 ((style + consistency) / 2), log-perplexity difference <= +0.02 and
              repetition-loop share difference <= +3 points;
  adoption    an alternative replaces the reference when its interval lies above 0 and every guardrail holds; among the adopted
              alternatives of one factor, the largest gain wins;
  order       (1) the foresight weight lambda (0.25, 1, 2 against 0.5); (2) at the chosen lambda, the commit order (current tilt), the
              predictor gate (0, 0.1), the plain score and neutral special tokens; (3) eight bands against four.
Response length and the share of responses under 20 tokens are reported, not used. Foresight from the predictor against the current
beliefs is reported, not selected. When two or more settings change, the combination is compared once with the reference on the
confirmation prompts (data/devB150_prompts_llada.jsonl) before it is frozen.
Usage:
  python scripts/select_dev.py --runs runs/dev_candidates runs/dev_candidates_b8 --judge judge/dev/quality.json --out dev_selection.json
(log-perplexities are read from the ppl_<arm>.json files that score_ppl.py writes into each run directory)"""
import argparse
import glob
import json
import os
import re

import numpy as np


def loop(text, n=4, thr=0.15):
    w = text.split()
    g = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    if not g:
        return 0
    seen, rep = set(), 0
    for x in g:
        rep += x in seen
        seen.add(x)
    return int(rep / len(g) > thr)


def load(run_dirs, judge):
    M = {}
    for d in run_dirs:
        ppl = {}
        for f in glob.glob(os.path.join(d, "ppl_*.json")):
            arm = os.path.basename(f)[4:-5]
            ppl[arm] = {pid: list(v.values())[0] for pid, v in json.load(open(f))["log_ppl"].items()}
        for f in glob.glob(os.path.join(d, "rows_shard*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                m = re.match(r"(.+)_dp([0-9.]+)$", r["arm"])
                if not m:
                    continue
                fam, dp = m.group(1), float(m.group(2))
                z = r["detection"]["z"]
                z = -9.0 if z is None or z != z else z
                M.setdefault(fam, {}).setdefault(dp, {})[r["prompt_id"]] = dict(
                    z=z, kl=r["kl"], lp=ppl.get(r["arm"], {}).get(r["prompt_id"]), rep=loop(r["completion"]),
                    judge=judge.get((r["arm"], r["prompt_id"])), len=r["n_gen"], short=int(r["n_gen"] < 20))
    return M


def arrays(M, fam, items, key):
    return np.array([[np.nan if M[fam][dp][i][key] is None else M[fam][dp][i][key] for i in items] for dp in sorted(M[fam])], dtype=float)


def at_kl(M, fam, items, idx, key, grid):
    kl = np.nanmean(arrays(M, fam, items, "kl")[:, idx], 1)
    v = np.nanmean(arrays(M, fam, items, key)[:, idx], 1)
    o = np.argsort(kl)
    return np.interp(grid, kl[o], v[o])


def compare(M, a, b, items, B):
    """a - b at equal KL (primary z with its interval; guardrails as point estimates)."""
    ka, kb = np.nanmean(arrays(M, a, items, "kl"), 1), np.nanmean(arrays(M, b, items, "kl"), 1)
    lo, hi = max(ka.min(), kb.min()), min(ka.max(), kb.max())
    if lo >= hi:
        return {"error": f"no common KL range {ka.min():.1f}-{ka.max():.1f} vs {kb.min():.1f}-{kb.max():.1f}"}
    grid, every = np.linspace(lo, hi, 5), np.arange(len(items))
    out = {"kl_range": [round(float(lo), 1), round(float(hi), 1)]}
    d = at_kl(M, a, items, every, "z", grid) - at_kl(M, b, items, every, "z", grid)
    boot = np.array([(at_kl(M, a, items, s, "z", grid) - at_kl(M, b, items, s, "z", grid)).mean() for s in B])
    out["dz"] = round(float(d.mean()), 3)
    out["dz_ci"] = [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)]
    for key in ("lp", "rep", "judge", "len", "short"):
        v = (at_kl(M, a, items, every, key, grid) - at_kl(M, b, items, every, key, grid)).mean()
        out["d" + key] = None if np.isnan(v) else round(float(v) * (100 if key in ("rep", "short") else 1), 3)
    return out


def adopt(c):
    if "error" in c:
        return False
    ok = c["dz_ci"][0] > 0
    if c.get("djudge") is not None:
        ok &= c["djudge"] >= -0.10
    if c.get("dlp") is not None:
        ok &= c["dlp"] <= 0.02
    if c.get("drep") is not None:
        ok &= c["drep"] <= 3.0
    return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run directories of the candidate arms (rows_shard*.jsonl, ppl_<arm>.json)")
    ap.add_argument("--judge", nargs="*", default=[], help="quality.json files written by judge.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    judge = {}
    for path in args.judge:
        for r in json.load(open(path))["results"]:
            sc = r.get("scores") or {}
            if "style" in sc and "consistency" in sc:
                judge[(r["set"], r["prompt_id"])] = (sc["style"] + sc["consistency"]) / 2
    M = load(args.runs, judge)
    items = sorted(set.intersection(*[set(p) for fam in M for p in M[fam].values()]))
    B = np.random.default_rng(0).integers(0, len(items), size=(4000, len(items)))
    rep = {"n_prompts": len(items), "comparisons": {}}
    C = rep["comparisons"]
    for alt in ("lam025", "lam1", "lam2"):
        C[f"{alt} - base"] = compare(M, alt, "base", items, B)
    gains = {"base": 0.0}
    gains.update({a: C[f"{a} - base"]["dz"] for a in ("lam025", "lam1", "lam2") if adopt(C[f"{a} - base"])})
    lam_fam = max(gains, key=gains.get)
    ref, tag = ("base", "") if lam_fam in ("base", "lam025") else ("lam1", "l1")
    chosen = {"lambda": {"base": 0.5, "lam025": 0.25, "lam1": 1.0, "lam2": 2.0}[lam_fam]}
    for factor, alts in (("order", ["ctilt"]), ("gate", ["gate0", "gate01"]), ("score", ["plain"]), ("special_tokens", ["spneu"])):
        best, gain = None, 0.0
        for a in alts:
            if a + tag not in M:
                continue
            c = C[f"{a}{tag} - {ref}"] = compare(M, a + tag, ref, items, B)
            if adopt(c) and c["dz"] > gain:
                best, gain = a, c["dz"]
        chosen[factor] = best or "default"
    b8 = "b8lam1" if ref == "lam1" else "b8base"
    if b8 in M:
        C[f"{b8} - {ref}"] = compare(M, b8, ref, items, B)
        chosen["bands"] = 8 if adopt(C[f"{b8} - {ref}"]) else 4
    for a, b in (("base", "cur"), ("lam1", "curlam1")):
        if a in M and b in M:
            C[f"{a} - {b}"] = compare(M, a, b, items, B)
    rep["chosen"] = chosen
    json.dump(rep, open(args.out, "w"), indent=1)
    print(json.dumps(chosen, indent=1))
    for k, c in C.items():
        print(f"{k:22s} {c}")


if __name__ == "__main__":
    main()
