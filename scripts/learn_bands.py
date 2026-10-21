"""Learning the band table color(v), a partition of the vocabulary into B bands, by coordinate ascent.

The objective is the expected green probability after the tilt, over the commit records of decoding traces:

    J(color) = mean_r (1/B) sum_b q(M_b(r)),    q(M) = M e^beta / (1 + M (e^beta - 1)),

where M_b(r) is the probability mass of the record's 50 candidates (renormalized over the 50) that falls in band b, and the mean over
b is the expectation over the keyed choice of the green band. Only flexible records, whose top candidate has probability below
--select-below, enter the objective. The search starts from a fixed hash coloring and visits the tokens in decreasing order of their
total candidate mass; each token moves to the band with the largest gain in J, and the passes repeat until no token moves or --passes
is reached. Tokens that never occur as candidates keep their hash color, and --passes 0 writes the hash coloring itself. The green band of
a context is still chosen by the key, so the null distribution of the detector does not depend on the table.

Records: DIR/items/*.pt, each {"prompt_id", "task", "rows": [{"cut", "rec": {"pos", "cand", "cand_p"}}]}; the commits at positions
pos < cut (inside the response) are used. Split: the files of the first directory, sorted, are shuffled with random.seed(seed) and a
fraction --val-frac of them is held out for validation; further directories add training records, except the files that have the name
of a held-out file.

Validation metrics, on the held-out records whose top candidate has probability below --eval-below: g2, g3, g4 = mean_b q(M_b) - M_b
at beta = 2, 3, 4, and same12 = the share of records whose two most likely candidates share a band; suffix _l for the table, _r for a
random partition drawn per record.

Example:
  python scripts/learn_bands.py --data records/random,records/lowconf --out checkpoints/bands.pt --beta 4 --passes 12
"""
import argparse
import glob
import json
import math
import os
import random
import time

import numpy as np
import torch


def load_records(files, task_prefix):
    """Candidates (N, 50), their probabilities (N, 50) and the index of the file of every commit record inside the response."""
    cand, prob, item = [], [], []
    for fi, f in enumerate(files):
        blob = torch.load(f, map_location="cpu")
        if task_prefix and not blob["task"].startswith(task_prefix):
            continue
        for row in blob["rows"]:
            rec, cut = row["rec"], int(row["cut"])
            keep = rec["pos"].long() < cut
            if not bool(keep.any()):
                continue
            cand.append(rec["cand"][keep].long().numpy())
            prob.append(rec["cand_p"][keep].float().numpy())
            item.append(np.full(int(keep.sum()), fi))
    if not cand:
        return np.zeros((0, 50), np.int64), np.zeros((0, 50), np.float32), np.zeros(0, np.int64)
    return np.concatenate(cand), np.concatenate(prob), np.concatenate(item)


def tilted(M, beta):
    """Probability of the green band after the tilt exp(beta), for a green band of mass M."""
    e = math.exp(beta)
    return M * e / (1 + M * (e - 1))


def hash_color(v, nb):
    """Fixed hash coloring of token ids (the starting point and the color of tokens never seen)."""
    x = (v.astype(np.int64) * 2654435761 + 12345) % 2147483647
    x = (x * 48271) % 2147483647
    return ((x >> 7) % nb).astype(np.int64)


def band_metrics(color_of, cand, prob, thr, nb):
    y = prob.max(1) < thr
    cand, prob = cand[y], prob[y]
    out = {"n": int(y.sum())}
    rng = np.random.default_rng(123)
    for name, band in (("l", color_of(cand)), ("r", rng.integers(0, nb, cand.shape))):
        M = np.zeros((len(cand), nb))
        for b in range(nb):
            M[:, b] = (prob * (band == b)).sum(1)
        for d in (2.0, 3.0, 4.0):
            out[f"g{int(d)}_{name}"] = round(float((tilted(M, d) - M).mean(1).mean()), 4)
        top2 = np.argsort(-prob, 1)[:, :2]
        same = np.take_along_axis(band, top2[:, :1], 1) == np.take_along_axis(band, top2[:, 1:], 1)
        out[f"same12_{name}"] = round(float(same.mean()), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="DIR[,DIR...] of record files DIR/items/*.pt; the first one gives the validation split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=4.0, help="tilt strength of the objective")
    ap.add_argument("--select-below", type=float, default=0.98, help="train on the records whose top candidate probability is below this")
    ap.add_argument("--eval-below", type=float, default=0.98, help="the same selection for the validation metrics")
    ap.add_argument("--passes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bands", type=int, default=4)
    ap.add_argument("--vocab-size", type=int, default=126464)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--task-prefix", default="", help="use only the record files whose task starts with this (default: all)")
    args = ap.parse_args()
    nb = args.bands
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ---- records and validation split
    t0 = time.time()
    dirs = args.data.split(",")
    files = sorted(glob.glob(os.path.join(dirs[0], "items", "*.pt")))
    order = list(range(len(files)))
    random.shuffle(order)
    val_items = set(order[: max(1, int(args.val_frac * len(order)))])
    val_names = {os.path.basename(files[i]) for i in val_items}
    cand, prob, item = load_records(files, args.task_prefix)
    is_val = np.isin(item, list(val_items))
    tr_c, tr_p = [cand[~is_val]], [prob[~is_val]]
    for d in dirs[1:]:
        extra = [f for f in sorted(glob.glob(os.path.join(d, "items", "*.pt"))) if os.path.basename(f) not in val_names]
        c, p, _ = load_records(extra, args.task_prefix)
        tr_c.append(c)
        tr_p.append(p)
    tr_c, tr_p = np.concatenate(tr_c), np.concatenate(tr_p)
    va_c, va_p = cand[is_val], prob[is_val]
    tr_p = tr_p / np.clip(tr_p.sum(1, keepdims=True), 1e-9, None)
    va_p = va_p / np.clip(va_p.sum(1, keepdims=True), 1e-9, None)
    print(f"records: train {len(tr_c)}, validation {len(va_c)} ({time.time() - t0:.0f}s)", flush=True)

    # ---- coordinate ascent on the flexible training records
    y = tr_p.max(1) < args.select_below
    C, Pm = tr_c[y], tr_p[y].astype(np.float64)
    uniq, cc = np.unique(C, return_inverse=True)
    cc = cc.reshape(C.shape)
    n_tok = len(uniq)
    color = hash_color(uniq, nb)
    M = np.zeros((len(C), nb))                                                 # band masses of every record
    for b in range(nb):
        M[:, b] = (Pm * (color[cc] == b)).sum(1)
    flat = cc.ravel()
    by_token = np.argsort(flat, kind="stable")                                 # occurrences grouped by token
    starts = np.searchsorted(flat[by_token], np.arange(n_tok + 1))
    rec_of = by_token // C.shape[1]
    p_of = Pm.ravel()[by_token]
    mass = np.bincount(flat, weights=Pm.ravel(), minlength=n_tok)
    tok_order = np.argsort(-mass)
    beta = args.beta
    objective = lambda MM: tilted(MM, beta).mean(1)
    print(f"tokens {n_tok}, flexible records {len(C)}, initial objective {objective(M).mean():.5f}", flush=True)
    hist = []
    for ps in range(args.passes):
        moved, t1 = 0, time.time()
        for u in tok_order:
            s, e = starts[u], starts[u + 1]
            if e == s:
                continue
            r, p = rec_of[s:e], p_of[s:e]
            a = color[u]
            Mr = M[r]
            base = objective(Mr)
            best, best_gain = a, 0.0
            for b in range(nb):
                if b == a:
                    continue
                M2 = Mr.copy()
                M2[:, a] -= p
                M2[:, b] += p
                g = float((objective(M2) - base).sum())
                if g > best_gain + 1e-12:
                    best, best_gain = b, g
            if best != a:
                M[r, a] -= p
                M[r, best] += p
                color[u] = best
                moved += 1
        cur = float(objective(M).mean())
        full = np.full(args.vocab_size, -1, dtype=np.int64)
        full[uniq] = color
        mv = band_metrics(lambda v: np.where(full[v] >= 0, full[v], hash_color(v, nb)), va_c, va_p, args.eval_below, nb)
        hist.append({"pass": ps, "moved": moved, "train_obj": round(cur, 5)} | mv)
        print(f"pass {ps}: moved {moved}, objective {cur:.5f} ({time.time() - t1:.0f}s), validation {json.dumps(mv)}", flush=True)
        if moved == 0:
            break

    full = np.full(args.vocab_size, -1, dtype=np.int64)
    full[uniq] = color
    colors = np.where(full >= 0, full, hash_color(np.arange(args.vocab_size), nb))
    final = band_metrics(lambda v: colors[v], va_c, va_p, args.eval_below, nb)
    train_cfg = {k: v for k, v in vars(args).items() if k not in ("data", "out")}
    torch.save({"color": torch.from_numpy(colors), "cfg": {"nb": nb, "key": 42}, "train_cfg": train_cfg, "val": final, "hist": hist},
               args.out)
    print("validation", json.dumps(final), flush=True)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
