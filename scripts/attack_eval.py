"""Robustness of detection to token edits and paraphrasing, with thresholds from attacked unwatermarked responses.

Positives are watermarked responses (--pos), negatives unwatermarked responses of the same method (--neg). Both are re-tokenized with
the LLaDA tokenizer (no special tokens added) and attacked in the same way:
  none          the response itself;
  deletion      round(rate * n) distinct positions of the n tokens are removed;
  insertion     round(rate * n) tokens are inserted one at a time, each at a uniform position of the current sequence;
  substitution  round(rate * n) distinct positions are replaced, a draw equal to the replaced token being redrawn once;
  paraphrase    the paraphrase written by scripts/attack_paraphrase.py (--paraphrases; responses without one are left out),
at rates 0.1, 0.2, 0.3 and 0.4 (the paper reports 0.2, and 0.1 and 0.2 for the context widths). Inserted and substituted tokens are
drawn from the unigram distribution of the negatives' tokens (special tokens excluded). The edits of a response use numpy
default_rng(crc32 of "NAME|prompt_id|ID|attack|rate"), NAME being the pool name (--pos-name / --neg-name, default: the directory of the
pool's first file) and ID the row's seed, or for the baselines the replicate index of its generation seed (scripts/baselines/protocol.py).

Detectors (a higher score is more evidence of the watermark):
  ours           z-score of lfbb.detector.detect with the partition and its key (default);
  dlm_kgw        z-score of the official DLM-KGW detector, greenlist seed 42 (--official-dlm-kgw);
  dgmark_global  match ratio of the official dgMARK detector (src/detection.py) on the re-tokenized text, given the prompt length (its
                 official z-score is an increasing function of the ratio), and
  dgmark_robust  z-score of the official robust detector (scripts/robust_detection.py, window 8) on the text and the raw prompt, without
                 its default minimum of 200 tokens (--official-dgmark with --prompts). Attacked token ids are decoded to text first;
                 dgMARK is keyless.
A response without a score (fewer than two tokens, no counted unit, no window) ranks below every scored response.

TPR@a is the share of positives scoring strictly above the ceil((1 - a) N)-th smallest of the N attacked negatives' scores. For the
keyed detectors the negatives are also detected under --num-keys keys (42, 1, 2, ...; 20 in the paper), and tpr_pooled@a uses the
pooled (negative, key) scores. Every result also holds the AUROC, the rates at 5%, nominal rates from the p-values, and 95% percentile
intervals from 1,000 bootstrap resamples of positives and negatives with the threshold re-estimated on each: numpy default_rng(12345),
shared in order by all results, for the AUROC and TPR@1%, and default_rng(777) per result for the pooled TPR@1% (negatives resampled
with all their keys). The intervals depend on the order of the responses (files sorted by path, rows in file order). Results are
given for all responses and for those with at least 200 tokens before the attack (len200).

Example:
  python scripts/attack_eval.py --partition checkpoints/bands.pt --pos runs/attack_ours/rows_shard*.jsonl --pos-arm ours_d4 \
      --neg runs/attack_none/rows_shard*.jsonl --paraphrases attack/paraphrase_ours.jsonl attack/paraphrase_none.jsonl \
      --out attack/ours.json
"""
import argparse
import importlib
import json
import math
import os
import sys
import zlib

import numpy as np
import torch
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, "baselines"))
from lfbb import detect, load_partition, load_special_ids  # noqa: E402
from lfbb.constants import EOT_ID, MASK_ID  # noqa: E402
from protocol import STUDY_SEED, generation_seed  # noqa: E402
from run_dlm_kgw import GAMMA, official_detection  # noqa: E402
from run_dlm_kgw import import_official as import_dlm_kgw  # noqa: E402

EDITS = ("deletion", "insertion", "substitution")
RATES = (0.1, 0.2, 0.3, 0.4)
LONG = 200                                         # len200 stratum: at least LONG tokens before the attack
FLOOR = {"ours": -10.0, "dlm_kgw": -10.0, "dgmark_global": -1e9, "dgmark_robust": -1e9}     # value of a missing score


def read_rows(paths, arm=None):
    """Rows of the JSONL files (sorted by path, rows in file order), optionally only those of one arm."""
    rows = []
    for path in sorted(set(paths)):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                if arm is None or r.get("arm") == arm:
                    rows.append(r)
    return rows


def replicate_of(row, study_seed, limit=1000):
    """Replicate index r of a baseline row: its seed is generation_seed(study_seed, dataset, prompt_id, r)."""
    for rep in range(limit):
        if generation_seed(study_seed, row["dataset"], row["prompt_id"], rep) == int(row["seed"]):
            return rep
    raise SystemExit(f"the seed of a row of {row['prompt_id']} is not a generation seed of the study seed {study_seed!r} (--study-seed)")


def load_pool(role, paths, arm, name, tok, study_seed=None):
    """The responses of one pool, re-tokenized. With study_seed (baseline rows) the edit streams use the replicate index."""
    name = name or os.path.basename(os.path.dirname(os.path.abspath(sorted(paths)[0])))
    texts = []
    for r in read_rows(paths, arm):
        ids = tok(r["completion"], add_special_tokens=False)["input_ids"]
        texts.append({"role": role, "name": name, "arm": r.get("arm"), "prompt_id": r["prompt_id"], "seed": r["seed"],
                      "stream_id": r["seed"] if study_seed is None else replicate_of(r, study_seed), "completion": r["completion"],
                      "ids": ids, "n0": len(ids)})
    if not texts:
        raise SystemExit(f"no rows for the {role}")
    if len({(t["prompt_id"], t["seed"]) for t in texts}) < len(texts):
        raise SystemExit(f"the {role} repeat a (prompt_id, seed) pair: select one arm (--pos-arm / --neg-arm)")
    return texts


def response_key(r):
    """A response and its paraphrase row share arm, prompt, seed and text."""
    return r.get("arm"), r["prompt_id"], r["seed"], r["completion"]


def unigram_cdf(texts, tok):
    """Cumulative unigram distribution over the vocabulary of the tokens of the texts, special tokens excluded."""
    V = len(tok.get_vocab())
    special = set(tok.all_special_ids)
    cnt = np.zeros(V)
    for t in texts:
        np.add.at(cnt, [i for i in t["ids"] if i not in special and i < V], 1)
    return np.cumsum(cnt / cnt.sum())


def draw(rng, k, cdf):
    """k token ids from the unigram distribution."""
    return np.searchsorted(cdf, rng.random(k)).clip(max=len(cdf) - 1).tolist()


def edit(ids, kind, rate, rng, cdf):
    """The token ids after a deletion, insertion or substitution attack at the rate, using the response's stream rng."""
    n = len(ids)
    k = int(round(rate * n))
    if k == 0 or n == 0:
        return list(ids)
    if kind == "deletion":
        drop = set(rng.choice(n, size=min(k, n), replace=False).tolist())
        return [x for i, x in enumerate(ids) if i not in drop]
    if kind == "insertion":
        out = list(ids)
        for tkn in draw(rng, k, cdf):
            out.insert(int(rng.integers(0, len(out) + 1)), int(tkn))
        return out
    if kind == "substitution":
        out = list(ids)
        pos = rng.choice(n, size=min(k, n), replace=False)
        for p, tkn in zip(pos.tolist(), draw(rng, len(pos), cdf)):
            out[p] = int(tkn) if int(tkn) != out[p] else int(draw(rng, 1, cdf)[0])
        return out
    raise ValueError(kind)


def edit_stream(t, kind, rate):
    """The random stream of a response for one attack and rate."""
    return np.random.default_rng(zlib.crc32(f"{t['name']}|{t['prompt_id']}|{t['stream_id']}|{kind}|{rate}".encode()))


def finite(x):
    return x if math.isfinite(x) else None


def calibration_keys(n):
    """42 first, then 1, 2, ... skipping 42, as in scripts/calibrate.py."""
    return ([42] + [k for k in range(1, n + 1) if k != 42])[:n]


# Scorers: score(ids, text, response) gives the scores of an attacked response (text: its paraphrase, None for the other attacks);
# key_scores(ids, keys) gives its z-scores under other keys (keyed detectors). None marks a missing score.
class Ours:
    """lfbb.detector.detect; other keys by setting the partition's key."""
    detectors, keyed = ("ours",), True

    def __init__(self, args, tok):
        self.part = load_partition(args.partition, key=args.key, special_ids=load_special_ids(args.special_ids), device=args.device,
                                   nb=args.bands)
        self.key, self.width = self.part.key, args.context_width

    def _z_p(self, ids):
        n, k, z, p = detect(self.part, ids, self.width) if len(ids) >= 2 else (0, 0, float("nan"), 1.0)
        return (z if n and z == z else None), p

    def score(self, ids, text, t):
        z, p = self._z_p(ids)
        return {"n_tokens": len(ids), "ours": z, "ours_p": p}

    def key_scores(self, ids, keys):
        zs = []
        for key in keys:
            self.part.key = int(key)
            zs.append(self._z_p(ids)[0])
        self.part.key = self.key
        return zs


class DlmKgw:
    """The official DLM-KGW detector, configured as in scripts/baselines/run_dlm_kgw.py; other keys by setting its greenlist seed."""
    detectors, keyed = ("dlm_kgw",), True

    def __init__(self, args, tok):
        import_dlm_kgw(args.official_dlm_kgw)
        from dlm_watermark.watermarks.diffusion_watermark import OurWatermark
        self.det = OurWatermark(delta=4.0, enforce_kl=False, convolution_kernel=[-1], greenlist_type="bernoulli",
                                greenlist_params={"gamma": GAMMA}, tokenizer=tok, topk=50, n_iter=1, seeding_scheme="sumhash",
                                device=args.device)                          # the configuration of the attacked responses
        self.det.set_temperature(0.5)
        self.det.set_mask_token(MASK_ID)
        self.key, self.device = int(self.det.greenlist.seed), args.device

    def _z_p(self, ids):
        n, k, z, p = official_detection(self.det, ids, self.device) if len(ids) >= 2 else (0, 0, float("nan"), None)
        return (z if z is not None and z == z else None), (1.0 if p is None else float(p))

    def score(self, ids, text, t):
        z, p = self._z_p(ids)
        return {"n_tokens": len(ids), "dlm_kgw": z, "dlm_kgw_p": p}

    def key_scores(self, ids, keys):
        zs = []
        for key in keys:
            self.det.greenlist.seed = int(key)
            zs.append(self._z_p(ids)[0])
        self.det.greenlist.seed = self.key
        return zs


def import_dgmark(repo):
    """WatermarkDetector (src/detection.py) and RobustWatermarkDetector (scripts/robust_detection.py) of the official dgMARK code."""
    root = os.path.abspath(repo)
    for sub in ("src", "scripts"):
        sys.path.insert(0, os.path.join(root, sub))
    detection, robust = importlib.import_module("detection"), importlib.import_module("robust_detection")
    for m in (detection, robust):
        if not os.path.abspath(m.__file__).startswith(root + os.sep):
            raise ImportError(f"{m.__name__} was imported from {m.__file__}, not from {root}")
    return detection.WatermarkDetector, robust.RobustWatermarkDetector


class DgMark:
    """The official dgMARK detectors on the text, without a private key (as in generation)."""
    detectors, keyed = ("dgmark_global", "dgmark_robust"), False

    def __init__(self, args, tok):
        WatermarkDetector, RobustWatermarkDetector = import_dgmark(args.official_dgmark)
        self.tok = tok
        self.glob = WatermarkDetector(eot_token_id=EOT_ID, private_key=None)
        self.robust = RobustWatermarkDetector(model_name=args.model, window_size=8, private_key=None)
        self.prompts = {r["prompt_id"]: r for r in read_rows([args.prompts])}

    def score(self, ids, text, t):
        text = self.tok.decode(ids) if text is None else text
        pr = self.prompts[t["prompt_id"]]
        ids = self.tok.encode(text, add_special_tokens=False)
        _, _, ratio = self.glob.calculate_match_ratio(ids, len(pr["prompt_token_ids"]))
        windows = self.robust.calculate_window_ratios(pr["prompt_text"], text, min_tokens=0)
        return {"n_tokens": len(ids), "dgmark_global": finite(float(ratio)) if ids else None,
                "dgmark_robust": finite(float(self.robust.detect_z(windows)[1])) if windows else None}


def auroc(pos, neg):
    a = np.concatenate([pos, neg])
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True)
    rk = (np.cumsum(c) - (c - 1) / 2.0)[inv]
    return float((rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def threshold(neg, a):
    return np.sort(neg)[int(math.ceil((1 - a) * len(neg))) - 1]


def percentiles(x):
    return [round(100 * float(np.percentile(x, q)), 2) for q in (2.5, 97.5)]


def evaluate(texts, scored, detectors, keys):
    """Results per (attack, rate), detector and stratum, computed in this order (they share the bootstrap generator)."""
    rng_b = np.random.default_rng(12345)
    res = {}
    for (kind, rate), v in scored.items():
        for det in detectors:
            for stratum in ("all", "len200"):
                keep = [(i, s) for i, s in v if stratum == "all" or texts[i]["n0"] >= LONG]
                P = [s for i, s in keep if texts[i]["role"] == "positives"]
                N = [s for i, s in keep if texts[i]["role"] == "negatives"]
                if not P or not N:
                    continue
                pos = np.array([FLOOR[det] if s[det] is None else s[det] for s in P])
                neg = np.array([FLOOR[det] if s[det] is None else s[det] for s in N])
                rec = {"n_pos": len(pos), "n_neg": len(neg), "auroc": round(100 * auroc(pos, neg), 2),
                       "fpr_floor": round(100 / len(neg), 2), "unscored_pos": sum(s[det] is None for s in P),
                       "unscored_neg": sum(s[det] is None for s in N)}
                for a in (0.05, 0.01, 0.001):
                    thr = threshold(neg, a)
                    rec[f"tpr@{a}"] = round(100 * float((pos > thr).mean()), 2)
                    rec[f"fpr_real@{a}"] = round(100 * float((neg > thr).mean()), 2)
                    if f"{det}_p" in P[0]:
                        rec[f"tpr_nominal@{a}"] = round(100 * float(np.mean([s[f"{det}_p"] < a for s in P])), 2)
                        rec[f"fpr_nominal@{a}"] = round(100 * float(np.mean([s[f"{det}_p"] < a for s in N])), 2)
                au, tp = [], []
                for _ in range(1000):
                    pb = pos[rng_b.integers(0, len(pos), len(pos))]
                    nb = neg[rng_b.integers(0, len(neg), len(neg))]
                    au.append(auroc(pb, nb))
                    tp.append(float((pb > threshold(nb, 0.01)).mean()))
                rec["auroc_ci"] = percentiles(au)
                rec["tpr@0.01_ci"] = percentiles(tp)
                if keys:
                    nk = np.array([[FLOOR[det] if z is None else z for z in s["zk"]] for s in N], dtype=float)
                    for a in (0.01, 0.001):
                        thr = threshold(nk.ravel(), a)
                        rec[f"tpr_pooled@{a}"] = round(100 * float((pos > thr).mean()), 2)
                        rec[f"thr_pooled@{a}"] = round(float(thr), 4)
                    rng_k = np.random.default_rng(777)
                    tk = []
                    for _ in range(1000):
                        pb = pos[rng_k.integers(0, len(pos), len(pos))]
                        nb = nk[rng_k.integers(0, len(nk), len(nk))].ravel()
                        tk.append(float((pb > threshold(nb, 0.01)).mean()))
                    rec["tpr_pooled@0.01_ci"] = percentiles(tk)
                    rec["n_keys"] = len(keys)
                res[f"{det}|{kind}|{rate}|{stratum}"] = rec
    return res


def print_table(res):
    pooled = bool(res["keys"])
    print("TPR (%) at the thresholds of the attacked negatives" + (f" pooled over {len(res['keys'])} keys" if pooled else ""))
    print(f"{'detector':<14} {'attack':<12} {'rate':>4} {'stratum':<7} {'pos':>4} {'neg':>4} {'AUROC':>6} {'TPR@1%':>7} {'95% CI':>16} "
          f"{'TPR@0.1%':>8}")
    for name, r in res["results"].items():
        det, kind, rate, stratum = name.split("|")
        t1, ci, t01 = ((r["tpr_pooled@0.01"], r["tpr_pooled@0.01_ci"], r["tpr_pooled@0.001"]) if pooled else
                       (r["tpr@0.01"], r["tpr@0.01_ci"], r["tpr@0.001"]))
        print(f"{det:<14} {kind:<12} {float(rate):>4.1f} {stratum:<7} {r['n_pos']:>4} {r['n_neg']:>4} {r['auroc']:>6.2f} {t1:>7.2f} "
              f"[{ci[0]:6.2f}, {ci[1]:6.2f}] {t01:>8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", nargs="+", required=True, help="JSONL rows of the watermarked responses (e.g. rows_shard*.jsonl)")
    ap.add_argument("--neg", nargs="+", required=True, help="JSONL rows of the unwatermarked responses of the same method")
    ap.add_argument("--pos-arm", help="use only the positive rows of this arm")
    ap.add_argument("--neg-arm", help="use only the negative rows of this arm")
    ap.add_argument("--pos-name", help="pool name in the edit streams (default: the directory of the first --pos file)")
    ap.add_argument("--neg-name", help="pool name in the edit streams (default: the directory of the first --neg file)")
    ap.add_argument("--paraphrases", nargs="*", default=[], help="rows written by attack_paraphrase.py (positives and negatives)")
    ap.add_argument("--out", required=True, help="output JSON")
    ap.add_argument("--scores", help="optional JSONL with the scores of every (response, attack)")
    ap.add_argument("--num-keys", type=int, default=20, help="keys of the pooled negatives of the keyed detectors (0: none)")
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct", help="tokenizer of the responses (and of dgMARK's robust detector)")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--partition", help="our detector: learned band table (.pt) or 'hash'")
    ap.add_argument("--key", type=int, default=None, help="our detector: key (default: the key stored with the partition, 42)")
    ap.add_argument("--bands", type=int, default=4, help="number of bands of the 'hash' partition")
    ap.add_argument("--context-width", type=int, default=2)
    ap.add_argument("--special-ids", default=os.path.join(HERE, "..", "data", "special_ids_llada.json"))
    ap.add_argument("--official-dlm-kgw", help="checkout of the official DLM-KGW repository: evaluate its detector")
    ap.add_argument("--official-dgmark", help="checkout of the official dgMARK repository: evaluate its detectors")
    ap.add_argument("--prompts", help="prompt file with prompt_id, prompt_text and prompt_token_ids (dgMARK)")
    ap.add_argument("--study-seed", default=STUDY_SEED, help="study seed of the baseline rows (identifies their replicates)")
    args = ap.parse_args()
    if args.official_dlm_kgw and args.official_dgmark:
        ap.error("evaluate one baseline at a time")
    method = "dlm_kgw" if args.official_dlm_kgw else "dgmark" if args.official_dgmark else "ours"
    if method == "ours" and not args.partition:
        ap.error("--partition is needed for our detector")
    if method == "dgmark" and not args.prompts:
        ap.error("--prompts is needed for dgMARK")
    for path in (args.out, args.scores):
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    study = None if method == "ours" else args.study_seed
    texts = (load_pool("positives", args.pos, args.pos_arm, args.pos_name, tok, study) +
             load_pool("negatives", args.neg, args.neg_arm, args.neg_name, tok, study))
    pools = {}
    for role in ("positives", "negatives"):
        g = [t for t in texts if t["role"] == role]
        pools[role] = {"name": g[0]["name"], "arms": sorted({str(t["arm"]) for t in g}), "n": len(g),
                       "n_ge200": sum(t["n0"] >= LONG for t in g)}
        print(role, pools[role], flush=True)
    cdf = unigram_cdf([t for t in texts if t["role"] == "negatives"], tok)
    scorer = {"ours": Ours, "dlm_kgw": DlmKgw, "dgmark": DgMark}[method](args, tok)
    keys = calibration_keys(args.num_keys) if scorer.keyed else []

    para = {response_key(r): r["attacked"] for r in read_rows(args.paraphrases)}
    missing = {role: sum(response_key(t) not in para for t in texts if t["role"] == role) for role in pools} if para else None
    if para:
        print("responses without a paraphrase", missing, flush=True)
    attacks = [("none", 0.0)] + [(kind, rate) for kind in EDITS for rate in RATES] + ([("paraphrase", 0.0)] if para else [])

    scored = {}
    fs = open(args.scores, "w") if args.scores else None
    for kind, rate in attacks:
        for i, t in enumerate(texts):
            text = None
            if kind == "paraphrase":
                text = para.get(response_key(t))
                if text is None:
                    continue
                ids = tok(text, add_special_tokens=False)["input_ids"]
            elif kind == "none":
                ids = t["ids"]
            else:
                ids = edit(t["ids"], kind, rate, edit_stream(t, kind, rate), cdf)
            s = scorer.score(ids, text, t)
            if keys and t["role"] == "negatives":
                s["zk"] = scorer.key_scores(ids, keys)
            scored.setdefault((kind, rate), []).append((i, s))
            if fs:
                fs.write(json.dumps({"pool": t["role"], "arm": t["arm"], "prompt_id": t["prompt_id"], "seed": t["seed"], "n0": t["n0"],
                                     "attack": kind, "rate": rate} | s) + "\n")
        print(kind, rate, "scored", flush=True)
    if fs:
        fs.close()

    res = {"method": method, "keys": keys, "rates": list(RATES), "pools": pools, "missing_paraphrase": missing,
           "results": evaluate(texts, scored, scorer.detectors, keys)}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print_table(res)


if __name__ == "__main__":
    main()
