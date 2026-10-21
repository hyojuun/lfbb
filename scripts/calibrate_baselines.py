"""Detection thresholds of the baselines' detectors from LLaDA responses decoded without a watermark (Table 1 and the other tables).

  dlm_kgw  keyed: the generated token ids of the responses (eff_ids of scripts/baselines/run_dlm_kgw.py --delta 0) scored as the
           official detector scores them (scripts/eval_codeqa.py dlm_kgw_z: the official green list OnTheFlyGreenlist, Bernoulli with
           gamma 0.25 and seeded by the key, one-token context, every distinct (previous token, token) pair once) under --num-keys keys
           (42, then 1, 2, ... skipping 42), pooled, so that the false-positive rate holds on average over keys;
  dgmark   keyless: the z-score of the official match statistic stored in the rows of scripts/baselines/run_dgmark.py, on dgMARK's own
           responses without the watermark (--variant none for the greedy version, none_top3 for top-3 sampling).
The texts are the responses of the same model, prompts and order as the evaluated runs. The threshold for a false-positive rate alpha
is the max(1, floor(alpha N))-th largest of the N scores that are not missing (scripts/eval_codeqa.py threshold): for 600 responses
under 100 keys the 600th and the 60th largest, for 600 keyless scores the 6th largest and the maximum. A response is detected when its
z-score is strictly above the threshold, and a missing z-score is not detected (as in scripts/report.py).

Output JSON, read by scripts/report.py --thresholds PREFIX=OUT: {"detector", "keys", "texts": [[arm, prompt_id, seed], ...],
"z": {key: [z per text]} (key "keyless" for dgmark), "pooled": {"n", "q99", "q999", "mean", "sd"}}, q99 and q999 being the thresholds
at 1% and 0.1%; with --runs also "tpr": {arm: {"n", "tpr1", "tpr01"}}, the TPRs (%) of the rows of these runs at the thresholds.

Example:
  python scripts/calibrate_baselines.py --detector dlm_kgw --texts runs/dlm_kgw_d0/rows_shard*.jsonl \
      --official-repo external/diffusion-lm-watermark --runs runs/dlm_kgw_d2 runs/dlm_kgw_d3 --out calib/dlm_kgw.json
  python scripts/calibrate_baselines.py --detector dgmark --texts runs/dgmark_none/rows_shard*.jsonl --runs runs/dgmark_greedy \
      --out calib/dgmark_greedy.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from calibrate import calibration_keys  # noqa: E402
from eval_codeqa import dlm_kgw_null, official, stored_z, threshold  # noqa: E402
from report import load_runs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True, choices=["dlm_kgw", "dgmark"])
    ap.add_argument("--texts", nargs="+", required=True,
                    help="JSONL files of unwatermarked responses (dlm_kgw: field eff_ids; dgmark: rows of run_dgmark.py)")
    ap.add_argument("--arm", help="use only the rows of this arm (default: every row)")
    ap.add_argument("--official-repo", help="root of a checkout of the official DLM-KGW code (commit 380a4bef), for dlm_kgw")
    ap.add_argument("--num-keys", type=int, default=100)
    ap.add_argument("--runs", nargs="*", default=[], help="run directories whose rows get TPRs at the thresholds (optional)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    texts = {}
    for path in args.texts:
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                if args.arm is None or r.get("arm") == args.arm:
                    texts[(r.get("arm"), r["prompt_id"], r.get("seed"))] = r
    names = sorted(texts, key=lambda t: tuple(map(str, t)))
    rows = {t: texts[t] for t in names}
    print("texts", len(names), flush=True)

    if args.detector == "dlm_kgw":
        if any("eff_ids" not in r for r in rows.values()):
            raise SystemExit("every row needs eff_ids (the generated token ids), as written by run_dlm_kgw.py")
        green_list_cls = official(args.official_repo, "dlm_watermark.watermarks.random_utils", "OnTheFlyGreenlist", "--official-repo")
        keys = calibration_keys(args.num_keys)
        z = {}
        for key in keys:
            z[str(key)] = [round(s, 5) if s == s else float("nan") for s in dlm_kgw_null(green_list_cls, rows, [key])]
        stored = [(a, stored_z(r)) for a, r in zip(z["42"], rows.values()) if r.get("key") == 42] if "42" in z else []
        diff = [abs(a - b) for a, b in stored if a == a and b == b]
        if diff:
            print(f"key 42 against the stored z-scores of the official detector: max |difference| {max(diff):.2e} ({len(diff)} rows)",
                  flush=True)
    else:
        keys = []
        z = {"keyless": [round(s, 5) if s == s else float("nan") for s in map(stored_z, rows.values())]}

    allz = np.array([s for zz in z.values() for s in zz if s == s])
    if not len(allz):
        raise SystemExit("no score")
    t1, t01 = threshold(allz, 0.01), threshold(allz, 0.001)
    pooled = {"n": int(len(allz)), "q99": round(t1, 5), "q999": round(t01, 5), "mean": round(float(allz.mean()), 4),
              "sd": round(float(allz.std()), 4)}
    print("pooled", pooled, flush=True)
    res = {"detector": args.detector, "keys": keys, "texts": [list(t) for t in names], "z": z, "pooled": pooled}
    if args.runs:
        res["tpr"] = {}
        for arm, R in load_runs(args.runs)[0].items():
            s = np.array([float(r["detection"]["z"]) if (r.get("detection") or {}).get("z") is not None else float("nan")
                          for r in R.values()])
            s = np.where(np.isfinite(s), s, -np.inf)
            res["tpr"][arm] = {"n": len(s), "tpr1": round(100 * float((s > t1).mean()), 2), "tpr01": round(100 * float((s > t01).mean()), 2)}
            print(arm, res["tpr"][arm], flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
