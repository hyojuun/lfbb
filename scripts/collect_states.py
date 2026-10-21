"""Training data of the foresight predictor: decision states with the exact post-placement beliefs.

Training prompts are decoded with the watermark and exact foresight, each at one of several strengths. At a decision state (the second
most likely tilted candidate has probability at least --gate), each of the top-8 candidates is placed and the generator is re-run. A
record holds, for the window of 2W+1 positions around the committed position j, the generator's final states and the inputs of its top
K blocks, the canvas tokens, the candidates and their log-probabilities, and the top-32 beliefs before and after placement of every
masked neighbor. With one commit per step, the next forward pass is the post-placement pass of the committed token, so every
--chosen-stride-th commit of a row is also recorded with its committed token alone at no extra cost.
Output: OUT/dump_<shard>_<n>.pt (decision states, 8 candidates) and OUT/ch_<shard>_<n>.pt (committed candidates).
With --family dream, positions are revealed several at a time, so only decision states are recorded (final states only).

Example:
  python scripts/collect_states.py --prompts data/train_prompts_llada.jsonl --partition checkpoints/bands.pt --out states/ --shard 0
"""
import argparse
import json
import os
import sys

import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lfbb import ArmConfig, MASK_ID, WatermarkSampler, load_partition, load_special_ids  # noqa: E402
from lfbb.dream import DREAM_FIRST_SPECIAL, DREAM_MASK_ID, DreamWatermarkSampler  # noqa: E402
from generate import seeds_of  # noqa: E402


class StateRecorder:
    def __init__(self, V, T, W=6, chosen_stride=2, mask_id=MASK_ID):
        self.V, self.T, self.W, self.stride, self.mask_id = V, T, W, chosen_stride, mask_id
        self.offsets = tuple(d for d in range(-W, W + 1) if d != 0)
        self.reset()

    def reset(self):
        self.records, self.chosen, self.pending, self.count = [], [], {}, {}

    def _window(self, x, jj):
        Lx = x.shape[1]
        pos = torch.arange(jj - self.W, jj + self.W + 1, device=x.device)
        ok = (pos >= 0) & (pos < Lx)
        return pos.clamp(0, Lx - 1), ok

    def _top32(self, logits):
        p = torch.softmax(logits[..., : self.V].float() / self.T, -1)
        return p.topk(32, -1)

    def on_exact(self, x, r, jj, vs, P, logits_r, post_logits, states):
        """A decision state: the 8 candidates vs at jj and the generator re-run with each of them placed (post_logits)."""
        H, XK = states
        pc, ok = self._window(x, jj)
        lpj = torch.log_softmax(logits_r[jj, : self.V].float() / self.T, -1)
        rec = {"r": r, "jj": jj - P, "v": vs.to(torch.int32).cpu(), "lpv": lpj[vs].half().cpu(), "win": H[r][pc].half().cpu(),
               "win_tok": torch.where(ok, x[r, pc], torch.full_like(pc, -1)).to(torch.int32).cpu(), "win_ok": ok.cpu(), "nb": {}}
        if XK is not None:
            rec["winxk"] = XK[r][pc].half().cpu()
        for d in self.offsets:
            p = jj + d
            if p < P or p >= x.shape[1] or int(x[r, p]) != self.mask_id:
                continue
            p32, i32 = self._top32(logits_r[p])
            q32, j32 = self._top32(post_logits[:, p])
            rec["nb"][d] = {"pre_ids": i32.to(torch.int32).cpu(), "pre_p": p32.half().cpu(), "post_ids": j32.to(torch.int32).cpu(),
                            "post_p": q32.half().cpu()}
        self.records.append(rec)

    def on_commit(self, x, logits, j, tok, cand, cand_p, P, step, states):
        """Completes the pending committed-candidate records with this forward pass (the post-placement pass of the previous commit),
        then registers every stride-th commit of this step as pending."""
        H, XK = states
        for r, pd in list(self.pending.items()):
            jj = pd.pop("_jj")
            v = int(x[r, jj])
            ids, lps = pd.pop("_cid"), pd.pop("_clp")
            hit = (ids == v).nonzero()
            if v == self.mask_id or not len(hit):
                continue
            nb = {}
            for d, pre in pd.pop("_pre").items():
                p = jj + d
                if int(x[r, p]) != self.mask_id:
                    continue
                q32, j32 = self._top32(logits[r, p])
                nb[d] = dict(pre, post_ids=j32[None].to(torch.int32).cpu(), post_p=q32[None].half().cpu())
            if nb:
                pd.update(v=torch.tensor([v], dtype=torch.int32), lpv=lps[int(hit[0])].reshape(1), nb=nb, chosen=True)
                self.chosen.append(pd)
        self.pending = {}
        for r in range(x.shape[0]):
            self.count[r] = self.count.get(r, -1) + 1
            if self.count[r] % self.stride:
                continue
            jj = int(j[r])
            pc, ok = self._window(x, jj)
            pre = {}
            for d in self.offsets:
                p = jj + d
                if P <= p < x.shape[1] and int(x[r, p]) == self.mask_id:
                    p32, i32 = self._top32(logits[r, p])
                    pre[d] = {"pre_ids": i32.to(torch.int32).cpu(), "pre_p": p32.half().cpu()}
            if not pre:
                continue
            lpj = torch.log_softmax(logits[r, jj, : self.V].float() / self.T, -1)
            self.pending[r] = {"r": r, "jj": jj - P, "win": H[r][pc].half().cpu(), "winxk": XK[r][pc].half().cpu(),
                               "win_tok": torch.where(ok, x[r, pc], torch.full_like(pc, -1)).to(torch.int32).cpu(), "win_ok": ok.cpu(),
                               "_jj": jj, "_cid": cand[r].cpu(), "_clp": lpj[cand[r]].half().cpu(), "_pre": pre}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--ids", help="JSON list of prompt ids to use (default: all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--partition", required=True,
                    help="band table (.pt) or 'hash'; it changes which states are visited, not the recorded inputs and targets")
    ap.add_argument("--family", default="llada", choices=["llada", "dream"])
    ap.add_argument("--model", default=None, help="default: GSAI-ML/LLaDA-8B-Instruct or Dream-org/Dream-v0-Instruct-7B")
    ap.add_argument("--special-ids", default=os.path.join(os.path.dirname(__file__), "..", "data", "special_ids_llada.json"))
    ap.add_argument("--deltas", default="1.0,1.5,2.0,2.5", help="strengths delta', one per prompt in turn")
    ap.add_argument("--all-deltas", action="store_true", help="decode every prompt at all strengths (one batch)")
    ap.add_argument("--unit-aware", action="store_true", help="unit-aware score while collecting (configs/reproduce/components.json collects with it)")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--gate", type=float, default=0.1)
    ap.add_argument("--K", type=int, default=8, help="number of top blocks whose inputs are recorded")
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--chosen-stride", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    dream = args.family == "dream"
    model_id = args.model or ("Dream-org/Dream-v0-Instruct-7B" if dream else "GSAI-ML/LLaDA-8B-Instruct")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    V = len(tok.get_vocab())
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
    special = range(DREAM_FIRST_SPECIAL, V) if dream else load_special_ids(args.special_ids)
    part = load_partition(args.partition, special_ids=special)
    if dream:
        sampler = DreamWatermarkSampler(model, V, part, T=args.temperature, capture=True)
        rec = StateRecorder(V, args.temperature, W=args.window, chosen_stride=0, mask_id=DREAM_MASK_ID)
    else:
        sampler = WatermarkSampler(model, V, part, T=args.temperature, capture_K=args.K)
        rec = StateRecorder(V, args.temperature, W=args.window, chosen_stride=args.chosen_stride)
        if args.chosen_stride > 0:
            sampler.on_commit = lambda **kw: rec.on_commit(**kw, states=sampler._step_states)
    sampler.on_exact = rec.on_exact
    deltas = [float(d) for d in args.deltas.split(",")]

    os.makedirs(args.out, exist_ok=True)
    prompts = [json.loads(l) for l in open(args.prompts)]
    if args.ids:
        keep = set(json.load(open(args.ids)))
        prompts = [p for p in prompts if p["prompt_id"] in keep]
    for n, pr in list(enumerate(prompts))[args.shard::args.num_shards]:
        dump = os.path.join(args.out, f"dump_{args.shard}_{n}.pt")
        if os.path.exists(dump):              # resume: the dump file of a prompt is written last
            continue
        ds = deltas if args.all_deltas else [deltas[n % len(deltas)]]
        arms = [ArmConfig(f"exact_{d}", delta=d, lam=args.lam, foresight="exact", order="current_tilt", gate=args.gate,
                          unit_aware=args.unit_aware) for d in ds]
        rec.reset()
        sd = seeds_of(pr["prompt_id"], 1)[0]
        out = sampler.generate(pr["prompt_token_ids"], [(sd, a) for a in arms])
        meta = {"P": len(pr["prompt_token_ids"]), "prompt_ids": list(pr["prompt_token_ids"]), "arms": [a.name for a in arms],
                "tokens": [o["tokens"] for o in out]}
        if rec.chosen:
            torch.save(dict(meta, records=rec.chosen, chosen=True), os.path.join(args.out, f"ch_{args.shard}_{n}.pt"))
        torch.save(dict(meta, records=rec.records), dump + ".tmp")
        os.replace(dump + ".tmp", dump)
        print(pr["prompt_id"], len(rec.records), len(rec.chosen), flush=True)
    print("done", args.shard, flush=True)


if __name__ == "__main__":
    main()
