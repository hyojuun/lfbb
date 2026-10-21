"""Fidelity of foresight to its exact value on held-out decision states (records of scripts/collect_states.py).

For each decision state, the foresight value of each of the 8 candidates, sum_{t=1,2} (g_t(v) - gamma), is computed from the exact
post-placement beliefs, from the current beliefs, from the predictor, and from an oracle that knows only the exact post-placement top-1
token of each masked neighbor (committed neighbors are exact; contexts before the response are padding). Reported: the share of the
exact values' variance across candidates (centered per state) that each source explains (R^2), their correlation, the rank correlation
and how often the top-ranked candidate is the exact best, and the agreement of the predicted top-1 neighbor token with the exact one.

Example:
  python scripts/analyze_predictor.py --states states/ --predictor checkpoints/predictor.pt --partition checkpoints/bands.pt --out fidelity.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lfbb import MASK_ID, load_partition, load_special_ids  # noqa: E402
from lfbb.predictor import TYPES, load_predictor  # noqa: E402
from train_predictor import prompt_hash  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--predictor", required=True)
    ap.add_argument("--partition", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--special-ids", default=os.path.join(os.path.dirname(__file__), "..", "data", "special_ids_llada.json"))
    ap.add_argument("--val-mod", type=int, default=20, help="use prompts whose hash is 0 modulo this (1 = all prompts)")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--vocab-size", type=int, default=126349)
    ap.add_argument("--bf16", action="store_true", help="run the predictor in bf16 as during decoding (default: fp32, as in training)")
    args = ap.parse_args()

    dev, T, V = "cuda", args.temperature, args.vocab_size
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to(dev).eval()
    net = load_predictor(args.predictor, model, fp32=not args.bf16)
    part = load_partition(args.partition, special_ids=load_special_ids(args.special_ids))
    g = part.gamma
    SRC = ("exact", "current", "predictor", "top1")
    cen = {s: [] for s in SRC}
    spear = {s: [] for s in SRC}
    amax = {s: [] for s in SRC}
    t1 = {"current": [], "predictor": []}
    nd = 0
    with torch.no_grad():
        for f in sorted(glob.glob(os.path.join(args.states, "dump_*.pt"))):
            dd = torch.load(f)
            if prompt_hash(dd["prompt_ids"]) % args.val_mod != 0:
                continue
            P = dd["P"]
            for rec in dd["records"]:
                if "winxk" not in rec:
                    continue
                Wn = rec["win"].shape[0]
                c = Wn // 2
                vs = rec["v"].long()
                kq = len(vs)
                tok, ok = rec["win_tok"].long(), rec["win_ok"]
                pos = torch.arange(Wn) - c + rec["jj"] + P
                typ = torch.full((Wn,), TYPES["committed"], dtype=torch.long)
                typ[tok == MASK_ID] = TYPES["masked"]
                typ[pos < P] = TYPES["prompt"]
                typ[~ok] = TYPES["outside"]
                typ[c] = TYPES["slot"]
                zz = net.hidden(rec["win"].to(dev)[None].expand(kq, -1, -1), typ.to(dev)[None].expand(kq, -1), vs.to(dev), rec["lpv"].to(dev),
                                rec["winxk"].to(dev)[None].expand(kq, -1, -1), rep=kq)
                Bel = {}
                for d in (-1, 1, 2):
                    nb = rec["nb"].get(d)
                    if nb is None:
                        continue
                    lp = torch.log_softmax(net.logits(zz[:, c + d], V) / T, -1)
                    tv, ti = lp.topk(8, -1)
                    post = [(nb["post_ids"][q].long(), nb["post_p"][q].float()) for q in range(kq)]
                    Bel[d] = {"exact": post, "current": [(nb["pre_ids"].long(), nb["pre_p"].float())] * kq,
                              "predictor": [(ti[q].cpu(), tv[q].exp().cpu()) for q in range(kq)],
                              "top1": [(post[q][0][:1], torch.ones(1)) for q in range(kq)]}
                    ex1 = torch.stack([post[q][0][0] for q in range(kq)])
                    t1["current"].append(float((ex1 == nb["pre_ids"][0].long()).float().mean()))
                    t1["predictor"].append(float((ex1 == ti[:, 0].cpu()).float().mean()))
                if not Bel:
                    continue

                def slot(d, src, q, k, other):
                    i = c + d
                    if d in Bel:
                        ids, p = Bel[d][src][q]
                        p = p[:k].float()
                        p = p / p.sum()
                        return list(zip(ids[:k].tolist(), p.tolist()))
                    if not bool(ok[i]):
                        return None
                    if int(pos[i]) < P:
                        return [(-1, 1.0)] if other else None
                    return [(int(tok[i]), 1.0)]

                vals = {}
                for src in SRC:
                    R = np.zeros(kq)
                    flat, cw, own, trow, twt = [], [], [], [], []
                    for q in range(kq):
                        v = int(vs[q])
                        for t in (1, 2):
                            oth, tgt = slot(-1 if t == 1 else 1, src, q, 4, True), slot(t, src, q, 8, False)
                            if oth is None or tgt is None:
                                continue
                            ti_ = [w for w, _ in tgt]
                            tw_ = [p for _, p in tgt]
                            ti_ = ti_ + [ti_[0]] * (8 - len(ti_))
                            tw_ = tw_ + [0.0] * (8 - len(tw_))
                            for u, pu in oth:
                                flat.append([u, v] if t == 1 else [v, u]); cw.append(pu); own.append(q); trow.append(ti_); twt.append(tw_)
                            R[q] -= g
                    if flat:
                        mem = part.member_lists(flat, torch.tensor(trow, device=dev))
                        val = (mem * torch.tensor(twt, device=dev)).sum(-1).cpu().numpy() * np.array(cw)
                        for i_, x_ in zip(own, val):
                            R[i_] += x_
                    vals[src] = R
                ex = vals["exact"]
                if np.std(ex) < 1e-9:
                    continue
                nd += 1
                for src in SRC:
                    a = vals[src]
                    cen[src].extend((a - a.mean()).tolist())
                    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(ex))
                    spear[src].append(float(np.corrcoef(ra, rb)[0, 1]) if np.std(a) > 0 else 0.0)
                    amax[src].append(float(np.argmax(a) == np.argmax(ex)))
    x = np.array(cen["exact"])
    summ = {"decisions": nd, "exact_centered_sd": round(float(x.std()), 4)}
    for src in SRC[1:]:
        e = np.array(cen[src])
        summ[f"R2_{src}"] = round(float(1 - ((e - x) ** 2).sum() / (x ** 2).sum()), 4)
        summ[f"corr_{src}"] = round(float(np.corrcoef(e, x)[0, 1]), 4)
        summ[f"spearman_{src}"] = round(float(np.mean(spear[src])), 4)
        summ[f"argmax_agree_{src}"] = round(float(np.mean(amax[src])), 4)
    for k, v in t1.items():
        if v:
            summ[f"top1_{k}"] = round(float(np.mean(v)), 4)
    print(json.dumps(summ, indent=1))
    json.dump(summ, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
