"""Trains the foresight predictor by distillation of the exact post-placement beliefs (records of scripts/collect_states.py).

The loss is the cross-entropy to the exact top-32 post-placement beliefs and their remaining mass, at temperature T, summed over the
masked positions of the window (weight 1 for j-1, j+1 and j+2, which foresight reads, and --aux-weight for the others). Decision-state
records (8 candidates) and committed-candidate records (1 candidate) are streamed in shuffled chunks of files; batches hold the same
number of (record, candidate) pairs for both kinds. Prompts whose hash is 0 modulo --val-mod are held out; the checkpoint with the lowest
held-out loss is kept, and every epoch is also saved.

For Dream (--family dream), the predictor is the window transformer of lfbb/predictor.py (WindowPredictor), trained in memory on the
decision states with a linear warm-up over 3% of the steps and a cosine schedule.

--init CKPT starts from a trained predictor (its architecture options replace --K, --ktrain and --hid), e.g. to train the predictor
further on the states of the code and QA training prompts (Table 3).

Example:
  python scripts/train_predictor.py --data states/ --out checkpoints/predictor.pt --epochs 6
  python scripts/train_predictor.py --family dream --data states_dream/ --out checkpoints/predictor_dream.pt --epochs 8 --batch 48 --patience 3
  python scripts/train_predictor.py --init checkpoints/predictor_epoch3.pt --data states_codeqa/ --out checkpoints/predictor_codeqa.pt
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lfbb import MASK_ID  # noqa: E402
from lfbb.dream import DREAM_MASK_ID  # noqa: E402
from lfbb.predictor import TYPES, ForesightPredictor, WindowPredictor, generator_parts  # noqa: E402

COLS = ("win", "winxk", "typ", "v", "lpv", "nbm", "qi", "qp", "val")


def coarse_ce(logp, ti, tp):
    """Cross-entropy of predicted log-probabilities (N, V) against a target top-32 (ids ti, probabilities tp) plus its remaining mass."""
    lq = logp.gather(1, ti)
    q_rest = (1 - lq.exp().sum(-1)).clamp_min(1e-9)
    t_rest = (1 - tp.sum(-1)).clamp_min(0)
    return -(tp * lq).sum(-1) - t_rest * q_rest.log()


def prompt_hash(ids):
    return zlib.crc32(json.dumps(list(map(int, ids))).encode())


class Data:
    def __init__(self, offsets, val_mod, block_inputs=True, mask_id=MASK_ID):
        self.offsets, self.val_mod, self.block_inputs, self.mask_id = offsets, val_mod, block_inputs, mask_id

    def add_records(self, dd, by_kq, is_val):
        P = dd["P"]
        for rec in dd["records"]:
            if "win" not in rec or (self.block_inputs and "winxk" not in rec):
                continue
            W_ = rec["win"].shape[0]
            c = W_ // 2
            tok, ok = rec["win_tok"].long(), rec["win_ok"]
            pos = torch.arange(W_) - c + rec["jj"] + P
            typ = torch.full((W_,), TYPES["committed"], dtype=torch.long)
            typ[tok == self.mask_id] = TYPES["masked"]
            typ[pos < P] = TYPES["prompt"]
            typ[~ok] = TYPES["outside"]
            typ[c] = TYPES["slot"]
            kq = rec["v"].shape[0]
            nbm = torch.zeros(len(self.offsets), dtype=torch.bool)
            qi = torch.zeros(len(self.offsets), kq, 32, dtype=torch.int32)
            qp = torch.zeros(len(self.offsets), kq, 32, dtype=torch.float16)
            for k, d in enumerate(self.offsets):
                nb = rec["nb"].get(d)
                if nb is None:
                    continue
                nbm[k] = True
                qi[k] = nb["post_ids"]
                qp[k] = nb["post_p"]
            if not bool(nbm.any()):
                continue
            cols = by_kq.setdefault(kq, {k_: [] for k_ in COLS})
            for k_, v_ in (("win", rec["win"]), ("typ", typ), ("v", rec["v"].long()), ("lpv", rec["lpv"].float().reshape(-1)),
                           ("nbm", nbm), ("qi", qi), ("qp", qp)):
                cols[k_].append(v_)
            if self.block_inputs:
                cols["winxk"].append(rec["winxk"])
            cols["val"].append(is_val)

    def load(self, files, which="all"):
        """Record files -> {candidates per record: columns}; which = "all" | "train" | "val" (split by prompt hash)."""
        by_kq = {}
        for f in files:
            try:
                dd = torch.load(f)
            except Exception as e:
                print("skip unreadable", f, type(e).__name__, flush=True)
                continue
            is_val = prompt_hash(dd["prompt_ids"]) % self.val_mod == 0
            if (which == "train" and is_val) or (which == "val" and not is_val):
                continue
            self.add_records(dd, by_kq, is_val)
        out = {}
        for kq, cols in by_kq.items():
            D = {k: torch.stack(v) for k, v in cols.items() if k not in ("val", "win", "winxk")}
            D["win"], D["winxk"] = cols["win"], cols["winxk"]
            D["val"] = torch.tensor(cols["val"], dtype=torch.bool)
            out[kq] = D
        return out


def batch_loss(net, D, idx, offsets, off_w, T, V, dev, train=True, with_cur=False):
    win = torch.stack([D["win"][int(i)] for i in idx]).to(dev, non_blocking=True)
    B, Wn, _ = win.shape
    kq = D["v"].shape[1]
    winq = win.repeat_interleave(kq, 0)
    typq = D["typ"][idx].to(dev).repeat_interleave(kq, 0)
    v = D["v"][idx].to(dev).reshape(-1)
    lpv = D["lpv"][idx].to(dev).reshape(-1)
    winxkq = torch.stack([D["winxk"][int(i)] for i in idx]).to(dev).repeat_interleave(kq, 0) if D["winxk"] else None
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=train):
        zz = net.hidden(winq, typq, v, lpv, winxkq, rep=kq)
    nbm = D["nbm"][idx]
    qi_b, qp_b = D["qi"][idx], D["qp"][idx]
    rows, cur, tis, tps, ws, offk = [], [], [], [], [], []
    c = Wn // 2
    for k, d in enumerate(offsets):
        sel = nbm[:, k].nonzero().squeeze(-1)
        if not len(sel):
            continue
        rq = (sel[:, None].to(dev) * kq + torch.arange(kq, device=dev)[None]).reshape(-1)
        rows.append(zz[rq, c + d])
        if with_cur:
            cur.append(winq[rq, c + d].float())
        tis.append(qi_b[sel, k].reshape(-1, 32).to(dev).long())
        tps.append(qp_b[sel, k].reshape(-1, 32).to(dev).float())
        ws.append(torch.full((len(rq),), float(off_w[k]), device=dev))
        offk.append(torch.full((len(rq),), k, device=dev))
    ti, tp, w, ok_ = torch.cat(tis), torch.cat(tps), torch.cat(ws), torch.cat(offk)
    logp = lambda z: torch.log_softmax(net.logits(z, V) / T, -1)
    lp = logp(torch.cat(rows))
    ce = coarse_ce(lp, ti, tp)
    top1 = lp.detach().argmax(-1) == ti[:, 0]
    loss = (w * ce).sum() / w.sum()
    if not with_cur:
        return loss, ce.detach(), ok_, top1
    lp0 = logp(torch.cat(cur))
    return loss, ce.detach(), ok_, top1, coarse_ce(lp0, ti, tp), lp0.argmax(-1) == ti[:, 0]


def evaluate(net, D, idx, offsets, off_w, T, V, dev, bs=48):
    """Held-out cross-entropy and top-1 agreement with the exact post-placement top-1, for the predictor and the current beliefs."""
    net.eval()
    acc = {k: torch.zeros(len(offsets)) for k in ("ce_pred", "ce_current", "top1_pred", "top1_current", "n")}
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            _, ce, ok_, top1, ce0, top10 = batch_loss(net, D, idx[i: i + bs], offsets, off_w, T, V, dev, train=False, with_cur=True)
            for k in range(len(offsets)):
                m = ok_ == k
                acc["n"][k] += float(m.sum())
                for nm, a in (("ce_pred", ce), ("ce_current", ce0), ("top1_pred", top1.float()), ("top1_current", top10.float())):
                    acc[nm][k] += float(a[m].sum())
    n = acc["n"]
    res = {"n": {d: int(n[k]) for k, d in enumerate(offsets)}}
    main = [offsets.index(d) for d in (-1, 1, 2)]
    for nm in ("ce_pred", "ce_current", "top1_pred", "top1_current"):
        res[nm] = {d: round(float(acc[nm][k] / max(n[k], 1)), 4) for k, d in enumerate(offsets)}
        res[nm + "_main"] = round(float(acc[nm][main].sum() / n[main].sum()), 4)
    return res


def build(model_path, K, ktrain, hid, dev):
    """The predictor on top of the generator's top K blocks (copies; the top ktrain are trained in fp32, the others frozen in bf16)."""
    m = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    blocks, lnf, emb, W = generator_parts(m)
    top = blocks[len(blocks) - K:]
    top = [b.to(dev, torch.bfloat16) for b in top[: K - ktrain]] + [b.float().to(dev) for b in top[K - ktrain:]]
    net = ForesightPredictor(emb.to(dev), W.to(dev), top, lnf.float(), K=K, ktrain=ktrain, hid=hid).to(dev)
    del m
    return net


def load_init(net, ck):
    """Loads the state of a checkpoint {"state", "cfg"} as lfbb.predictor.load_predictor does (not strict)."""
    missing, unexpected = net.load_state_dict(ck["state"], strict=False)
    print(f"initialized from the checkpoint: {len(missing)} missing and {len(unexpected)} unexpected keys", flush=True)


def train_in_memory(net, data, files, args, offsets, off_w, T, V, dev):
    """All records in memory, one learning-rate schedule over the planned steps (the Dream predictor)."""
    D = data.load(files)[8]
    tr, va = torch.nonzero(~D["val"]).squeeze(-1), torch.nonzero(D["val"]).squeeze(-1)
    print(f"records train {len(tr)} val {len(va)}", flush=True)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    steps = args.epochs * math.ceil(len(tr) / args.batch)
    warm = max(1, int(0.03 * steps))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, i / max(1, steps)))))
    ev = evaluate(net, D, va, offsets, off_w, T, V, dev)
    print("val before", json.dumps(ev), flush=True)
    best, best_state, bad = ev, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0
    for ep in range(args.epochs):
        net.train()
        perm = tr[torch.randperm(len(tr))]
        tl, tn, t1 = 0.0, 0, time.time()
        for i in range(0, len(perm), args.batch):
            loss, _, _, _ = batch_loss(net, D, perm[i: i + args.batch], offsets, off_w, T, V, dev)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sch.step()
            tl += float(loss.detach())
            tn += 1
        ev = evaluate(net, D, va, offsets, off_w, T, V, dev)
        print(f"epoch {ep} train CE {tl / max(tn, 1):.4f} ({time.time() - t1:.0f}s) val {json.dumps(ev)}", flush=True)
        if ev["ce_pred_main"] < best["ce_pred_main"]:
            best, best_state, bad = ev, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0
            torch.save({"state": best_state, "cfg": net.cfg, "val": best, "epoch": ep}, args.out)
        else:
            bad += 1
            if bad >= args.patience:
                break
    return best, best_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="record directories, comma-separated")
    ap.add_argument("--out", required=True)
    ap.add_argument("--family", default="llada", choices=["llada", "dream"])
    ap.add_argument("--model", default=None, help="default: GSAI-ML/LLaDA-8B-Instruct or Dream-org/Dream-v0-Instruct-7B")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--ktrain", type=int, default=8)
    ap.add_argument("--hid", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4, help="adapter learning rate")
    ap.add_argument("--block-lr", type=float, default=2e-5, help="learning rate of the transformer blocks")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=12, help="decision states per batch (8 candidates each)")
    ap.add_argument("--chunk", type=int, default=48, help="record files per streamed chunk")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--aux-weight", type=float, default=0.3)
    ap.add_argument("--val-mod", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--vocab-size", type=int, default=None, help="default: 126349 (LLaDA) or 151667 (Dream)")
    ap.add_argument("--init", help="predictor checkpoint to start from (its K, ktrain and hid replace the options)")
    args = ap.parse_args()
    init = torch.load(args.init, map_location="cpu") if args.init else None

    dream = args.family == "dream"
    dev, T = "cuda", args.temperature
    V = args.vocab_size or (151667 if dream else 126349)
    offsets = tuple(d for d in range(-args.window, args.window + 1) if d != 0)
    off_w = torch.tensor([1.0 if d in (-1, 1, 2) else args.aux_weight for d in offsets])
    dirs = args.data.split(",")
    data = Data(offsets, args.val_mod, block_inputs=not dream, mask_id=DREAM_MASK_ID if dream else MASK_ID)
    files = sorted(f for d in dirs for f in glob.glob(os.path.join(d, "*.pt")))
    if dream:
        m = AutoModel.from_pretrained(args.model or "Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True, torch_dtype=torch.bfloat16)
        cfg = {k: v for k, v in init["cfg"].items() if k != "arch"} if init else {}
        net = WindowPredictor(m.get_input_embeddings().weight.detach().to(dev), m.lm_head.weight.detach().to(dev), **cfg).to(dev)
        del m
        if init:
            net.load_state_dict(init["state"])
        best, best_state = train_in_memory(net, data, files, args, offsets, off_w, T, V, dev)
        torch.save({"state": best_state, "cfg": net.cfg, "val": best}, args.out)
        print("best val", json.dumps(best), flush=True)
        return
    dumps = [f for f in files if os.path.basename(f).startswith("dump_")]
    t0 = time.time()
    Dval = data.load(dumps, which="val")[8]
    va = torch.arange(len(Dval["win"]))
    print(f"files {len(files)} (decision states {len(dumps)}), val records {len(va)} ({time.time() - t0:.0f}s)", flush=True)

    if init:
        args.K, args.ktrain, args.hid = (int(init["cfg"][k]) for k in ("K", "ktrain", "hid"))
    net = build(args.model or "GSAI-ML/LLaDA-8B-Instruct", args.K, args.ktrain, args.hid, dev)
    if init:
        load_init(net, init)
    groups = [{"params": [p for n, p in net.named_parameters() if p.requires_grad and n.startswith("blocks.")], "lr": args.block_lr},
              {"params": [p for n, p in net.named_parameters() if p.requires_grad and not n.startswith("blocks.")], "lr": args.lr}]
    for g in groups:
        g["base"] = g["lr"]
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    ev = evaluate(net, Dval, va, offsets, off_w, T, V, dev)
    print("val before", json.dumps(ev), flush=True)
    best, best_state, bad, step = ev, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0, 0
    rng = random.Random(0)
    pool = ThreadPoolExecutor(1)
    for ep in range(args.epochs):
        net.train()
        fl = files[:]
        rng.shuffle(fl)
        chunks = [fl[i: i + args.chunk] for i in range(0, len(fl), args.chunk)]
        fut = pool.submit(data.load, chunks[0], "train")
        tl, tn, t1 = 0.0, 0, time.time()
        for ci in range(len(chunks)):
            Ds = fut.result()
            if ci + 1 < len(chunks):
                fut = pool.submit(data.load, chunks[ci + 1], "train")
            bat = []
            for kq, D in Ds.items():
                n, bsz = len(D["win"]), max(1, args.batch * 8 // kq)
                perm = torch.randperm(n)
                bat += [(kq, perm[i: i + bsz]) for i in range(0, n, bsz)]
            rng.shuffle(bat)
            for kq, idx in bat:
                cos = 0.5 * (1 + math.cos(math.pi * (ep + ci / len(chunks)) / args.epochs))
                for g in opt.param_groups:
                    g["lr"] = g["base"] * min(1.0, (step + 1) / args.warmup) * cos
                loss, _, _, _ = batch_loss(net, Ds[kq], idx, offsets, off_w, T, V, dev)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                step += 1
                tl += float(loss.detach())
                tn += 1
        ev = evaluate(net, Dval, va, offsets, off_w, T, V, dev)
        print(f"epoch {ep} train CE {tl / max(tn, 1):.4f} steps {tn} ({time.time() - t1:.0f}s) val {json.dumps(ev)}", flush=True)
        state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        torch.save({"state": state, "cfg": net.cfg, "val": ev, "epoch": ep}, os.path.splitext(args.out)[0] + f"_epoch{ep}.pt")
        if ev["ce_pred_main"] < best["ce_pred_main"]:
            best, best_state, bad = ev, state, 0
            torch.save({"state": best_state, "cfg": net.cfg, "val": best, "epoch": ep}, args.out)
        else:
            bad += 1
            if bad >= args.patience:
                break
    torch.save({"state": best_state, "cfg": net.cfg, "val": best}, args.out)
    print("best val", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
