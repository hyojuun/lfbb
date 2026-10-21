"""Watermarked decoding of LLaDA with the tilt q(v) ~ p(v) exp(beta s(v)) (Proposition 1 of the paper).

Decoding follows LLaDA's semi-autoregressive low-confidence rule: blocks of 32 positions, one commit per step, and at each step every
masked position of the block draws a candidate by Gumbel-max at temperature T; the position whose candidate has the highest confidence
is committed. The watermark adds delta' * s(v) to the logits of the top-k candidates before the temperature is applied (beta = delta'/T):

    s(v) = (g_0(v) - gamma) + lambda * sum_{t=1,2} (g_t(v) - gamma)

g_0(v) is the probability that v is green, an expectation over the beliefs about the context positions j-2, j-1 that are still masked.
g_t(v) is the probability that the neighbor j+t is green once v is placed (foresight): its context contains v, and its token follows
the belief about j+t. Foresight comes from the current beliefs, from the foresight predictor (predicted post-placement beliefs), or from
re-running the generator with v placed (exact). The predicted or exact beliefs are used where the choice is open, i.e. where the second
most likely tilted candidate has probability at least `gate`.

Unit-aware score: the detector counts each distinct (context, token) unit once, so a candidate whose unit already occurs in the
committed text gets no own term, and a neighbor whose unit would repeat one adds nothing (its membership is gamma).
Special tokens get the boost of a red token, -delta' * gamma, or with special = "neutral" the boost that keeps their probability
unchanged by the tilt.

Consistent order: the tilt decides the position as well as the token. The position is chosen from the tilted confidences; when the
commit tilt of a position differs from the tilt used to rank it (predicted or exact foresight), the winning position gets its commit
tilt and the ranking is repeated, up to `order_iters` times, so the committed token is the one whose confidence won the position.

All random numbers of a row come from generators seeded by the row's seed and the step, so rows with equal seeds share their noise.
"""
from dataclasses import dataclass

import torch

from .constants import MASK_ID
from .predictor import TYPES


@dataclass
class ArmConfig:
    """One watermark configuration. delta is the logit bias delta' added before the temperature T; the paper reports 2 delta'
    (at T = 0.5, beta = delta' / T equals that value)."""
    name: str
    delta: float = 0.0
    lam: float = 0.5                  # weight of foresight; 0 disables it
    foresight: str = "predictor"      # "current" | "predictor" | "exact"
    unit_aware: bool = True
    order: str = "consistent"         # "consistent" | "current_tilt" | "random"
    gate: float = 0.02                # predicted / exact foresight only where the second tilted candidate has probability >= gate
    order_iters: int = 4
    context_width: int = 2            # 1 or >= 3 for the context-width ablation (own term only, lam = 0)
    special: str = "red"              # special tokens: "red" (boost -delta' * gamma) or "neutral" (probability kept)

    def __post_init__(self):
        if self.context_width != 2 and self.lam > 0:
            raise ValueError("context widths other than 2 are implemented for the own term only (lam = 0)")

    @property
    def watermarked(self):
        return self.delta > 0

    @staticmethod
    def from_dict(d):
        return ArmConfig(**d)


def make_generator(device, *parts):
    g = torch.Generator(device=device)
    s = 0
    for p in parts:
        s = (s * 1000003 + int(p)) % (2 ** 62)
    g.manual_seed(s)
    return g


def gumbel_noise(device, seed, step, n, V):
    u = torch.rand(n, V, generator=make_generator(device, seed, step, 2), device=device).clamp_(1e-10, 1 - 1e-10)
    return -torch.log(-torch.log(u))


def top_beliefs(logits, T, k):
    """Top-k ids and probabilities of softmax(logits / T), for rows of logits (N, V)."""
    p = torch.log_softmax(logits.float() / T, -1).exp()
    p, ids = p.topk(k, -1)
    return ids, p


class WatermarkSampler:
    def __init__(self, model, vocab_size, partition, predictor=None, T=0.5, gen_length=256, block_length=32, k=50, k_foresight=8,
                 device="cuda", capture_K=0, mask_id=MASK_ID, commits_per_step=1, refresh=0):
        self.model, self.V, self.part, self.predictor = model, vocab_size, partition, predictor
        self.mask_id = mask_id
        # several commits per forward pass: the later commits of a step reuse its logits; refresh = 1 (rows with predicted foresight)
        # or 2 (every row) replaces the logits of the block's masked positions around each commit by the predictor's post-placement ones
        self.commits_per_step, self.refresh = int(commits_per_step), int(refresh)
        self.T, self.gen_length, self.block_length, self.k, self.kq = T, gen_length, block_length, k, k_foresight
        self.device = device
        self.special = set(partition.special.tolist())
        self.stats = {"commits": 0, "open": 0}
        self._states = self._block_in = None
        self._step_states = (None, None)
        self.on_commit = None             # optional callbacks used to record training data (scripts/collect_*.py)
        self.on_exact = None
        self._register_hooks(predictor, capture_K)

    def _register_hooks(self, predictor, capture_K):
        K = int(predictor.cfg["K"]) if predictor is not None else int(capture_K)
        if K > 0:                                                       # capture the states the predictor reads during the forward pass
            from .predictor import generator_parts
            blocks, _, _, _ = generator_parts(self.model)
            ff_out = next(m for n, m in self.model.named_modules() if n.endswith("transformer.ff_out"))
            ff_out.register_forward_hook(lambda m, inp, out: setattr(self, "_states", inp[0].detach()))
            blocks[len(blocks) - K].register_forward_pre_hook(lambda m, args: setattr(self, "_block_in", args[0].detach()))

    def _forward(self, x):
        return self.model(x).logits

    # ------------------------------------------------------------------ decoding
    @torch.no_grad()
    def generate(self, prompt_ids, rows):
        """rows: list of (seed, ArmConfig) decoded together for one prompt. Returns per row {"tokens", "kl"} (the response tokens of the
        canvas and the summed KL divergence of the tilted from the untilted sampling distribution)."""
        dev, B, P = self.device, len(rows), len(prompt_ids)
        Lx = P + self.gen_length
        x = torch.full((B, Lx), self.mask_id, dtype=torch.long, device=dev)
        x[:, :P] = torch.tensor(prompt_ids, device=dev)
        kl_sum = [0.0] * B
        wm_rows = [r for r in range(B) if rows[r][1].watermarked]
        step = 0
        logits = None
        for b in range(self.gen_length // self.block_length):
            lo, hi = P + b * self.block_length, P + (b + 1) * self.block_length
            for i in range(self.block_length):
                if self.commits_per_step <= 1 or logits is None or i % self.commits_per_step == 0:
                    logits = self._forward(x)
                    self._step_states = (self._states, self._block_in)        # this forward's states (exact foresight runs more forwards)
                j = torch.empty(B, dtype=torch.long, device=dev)
                gumbel = torch.empty(B, self.V, device=dev)
                for r in range(B):
                    if rows[r][1].order == "random":                         # LLaDA's random remasking order
                        u = torch.rand(self.block_length, generator=make_generator(dev, rows[r][0], step, 0), device=dev)
                        u = torch.where(x[r, lo:hi] == self.mask_id, u, torch.full_like(u, -1e9))
                        j[r] = lo + int(u.argmax())
                        uu = torch.rand(self.V, generator=make_generator(dev, rows[r][0], step, 1), device=dev).clamp_(1e-10, 1 - 1e-10)
                        gumbel[r] = -torch.log(-torch.log(uu))
                        continue
                    pos_m = torch.arange(lo, hi, device=dev)[x[r, lo:hi] == self.mask_id]
                    lg_m = logits[r, pos_m, : self.V].float()
                    gb_m = gumbel_noise(dev, rows[r][0], step, len(pos_m), self.V)
                    cand_m, boost_m = self._position_tilt(x, logits, r, pos_m, rows, P, step)
                    tl_m = lg_m.scatter_add(1, cand_m, boost_m.to(lg_m.dtype))
                    x0_m = (tl_m + self.T * gb_m).argmax(-1)
                    conf_m = torch.softmax(tl_m.double(), -1).gather(1, x0_m[:, None])[:, 0]
                    q = int(conf_m.argmax())
                    j[r] = pos_m[q]
                    gumbel[r] = gb_m[q]
                ar = torch.arange(B, device=dev)
                lj = logits[ar, j, : self.V].float()
                top = torch.softmax(lj / self.T, dim=-1).topk(self.k, dim=-1)
                cand = top.indices
                boost = torch.zeros(B, self.k, device=dev)
                if wm_rows:
                    boost[wm_rows] = self._commit_boost(x, logits, j, cand, wm_rows, rows, P)
                tok = (lj + self.T * gumbel).scatter_add_(1, cand, boost).argmax(-1)
                logp0 = torch.log_softmax(lj / self.T, -1)
                logps = torch.log_softmax(lj.scatter_add(1, cand, boost) / self.T, -1)
                kl = (logps.exp() * (logps - logp0)).sum(-1).clamp_min(0.0)
                if self.on_commit is not None:
                    self.on_commit(x=x, logits=logits, j=j, tok=tok, cand=cand, cand_p=top.values, P=P, step=step)
                x[ar, j] = tok
                if self.commits_per_step > 1 and self.refresh and (i + 1) % self.commits_per_step != 0 and self.predictor is not None:
                    self._refresh_beliefs(x, logits, j, tok, rows, P, lo, hi)
                for r in range(B):
                    kl_sum[r] += float(kl[r])
                step += 1
        return [{"tokens": x[r, P:].tolist(), "kl": kl_sum[r]} for r in range(B)]

    # ------------------------------------------------------------------ helpers
    @torch.no_grad()
    def _refresh_beliefs(self, x, logits, j, tok, rows, P, lo, hi):
        net = self.predictor
        H, XK = self._step_states
        if H is None or XK is None:
            return
        c, dev, Lx = net.c, x.device, x.shape[1]
        for r in range(x.shape[0]):
            arm = rows[r][1]
            if not (arm.foresight == "predictor" or self.refresh >= 2):
                continue
            jj, v = int(j[r]), int(tok[r])
            pos = jj + torch.arange(-c, c + 1, device=dev)
            ok = (pos >= 0) & (pos < Lx)
            pc = pos.clamp(0, Lx - 1)
            typ = torch.full_like(pc, TYPES["committed"])
            typ[x[r, pc] == self.mask_id] = TYPES["masked"]; typ[pos < P] = TYPES["prompt"]; typ[~ok] = TYPES["outside"]; typ[c] = TYPES["slot"]
            vs = torch.tensor([v], device=dev)
            lpv = torch.log_softmax(logits[r, jj, : self.V].float() / self.T, -1)[vs]
            zz = net.hidden(H[r][pc][None], typ[None], vs, lpv, XK[r][pc][None], rep=1)
            new = (zz[0].to(net.W.dtype) @ net.W.t())[:, : self.V]
            idx = torch.nonzero(ok & (pos >= lo) & (pos < hi) & (x[r, pc] == self.mask_id)).flatten()
            if len(idx):
                logits[r, pos[idx], : self.V] = new[idx].to(logits.dtype)

    def _wide_contexts(self, x, logits, r, slots, P):
        """Contexts of width h >= 3 folded into two slots, [fold(x_{j-h}..x_{j-2}), x_{j-1}], with the top-2 beliefs of masked slots."""
        from .partition import fold
        opts = []
        for pa in slots:
            if pa < P:
                opts.append([(-1, 1.0)])
            elif int(x[r, pa]) != self.mask_id:
                opts.append([(int(x[r, pa]), 1.0)])
            else:
                p, ids = torch.softmax(logits[r, pa, : self.V].float() / self.T, -1).topk(2)
                p = p / p.sum()
                opts.append(list(zip(ids.tolist(), p.tolist())))
        cmb = [([], 1.0)]
        for o in opts:
            cmb = [(c + [u], w * pu) for c, w in cmb for u, pu in o]
        return [([fold(c[:-1]), c[-1]], w) for c, w in cmb]

    def _own_contexts(self, x, logits, r, jj, P, arm):
        """Contexts of position jj with their probabilities, for the own term."""
        h = arm.context_width
        if h == 2:
            return self._context_mixture(x, logits, r, [jj - 2, jj - 1], P)
        if h == 1:
            return [([-1] + c, w) for c, w in self._context_mixture(x, logits, r, [jj - 1], P)]
        return self._wide_contexts(x, logits, r, list(range(jj - h, jj)), P)

    def _units(self, xr, P):
        """Distinct units (x_{i-2}, x_{i-1}, x_i) of the committed response (contexts before the response are padding, -1)."""
        out = set()
        for i in range(P, len(xr)):
            t = xr[i]
            if t == self.mask_id or t in self.special:
                continue
            a = xr[i - 2] if i - 2 >= P else -1
            b = xr[i - 1] if i - 1 >= P else -1
            if a == self.mask_id or b == self.mask_id:
                continue
            out.add((a, b, t))
        return out

    def _repeated_to_gamma(self, xr, P, contexts, targets, mem):
        """Memberships of (context, target) units that already occur in the committed response -> gamma."""
        units = self._units(xr, P)
        if not units:
            return mem
        tl = targets.tolist()
        rep = torch.tensor([[(c[0], c[1], w) in units for w in row] for c, row in zip(contexts, tl)], device=mem.device)
        if bool(rep.any()):
            return torch.where(rep, torch.full_like(mem, self.part.gamma), mem)
        return mem

    def _context_options(self, x, logits, r, pa, P, topk=4):
        """Possible tokens of a context position with their probabilities: padding before the response, the token if committed, else
        the top-k of its current belief (renormalized)."""
        if pa < P:
            return [(-1, 1.0)]
        if int(x[r, pa]) != self.mask_id:
            return [(int(x[r, pa]), 1.0)]
        ids, p = top_beliefs(logits[r, pa, : self.V][None], self.T, topk)
        p = p[0] / p[0].sum()
        return list(zip(ids[0].tolist(), p.tolist()))

    def _context_mixture(self, x, logits, r, slots, P):
        combos = [([], 1.0)]
        for opts in (self._context_options(x, logits, r, pa, P) for pa in slots):
            combos = [(c + [u], w * pu) for c, w in combos for u, pu in opts]
        return combos

    def _expected_member(self, combos_per_query, targets, dedup=None):
        """sum_c w_c member(c, targets[q]) for each query q, whose contexts are combos_per_query[q] = [(context, weight)]."""
        dev = targets.device
        flat, wts, owner = [], [], []
        for qi, combos in enumerate(combos_per_query):
            for c, w in combos:
                flat.append(c); wts.append(w); owner.append(qi)
        own = torch.tensor(owner, device=dev)
        tr = targets[own]
        mem = self.part.member_lists(flat, tr)
        if dedup is not None:
            mem = self._repeated_to_gamma(dedup[0], dedup[1], flat, tr, mem)
        out = torch.zeros(len(combos_per_query), tr.shape[1], device=dev)
        out.index_add_(0, own, mem * torch.tensor(wts, device=dev)[:, None])
        return out

    def _specials(self, boost, cand, arm, lg):
        """Boosts of the special tokens among the candidates cand (n, k), whose rows have the logits lg (n, V).
        special = "red": the boost of a red token, -delta' * gamma. special = "neutral": the boost b that leaves their probability after
        the tilt equal to the untilted one, given the boosts b_u of the other candidates: b = T log(A / (1 - p(S))), where p = softmax(lg / T),
        S is the set of special candidates and A = 1 - p(S) + sum_u p_u (exp(b_u / T) - 1) over the other candidates u."""
        sp = torch.isin(cand, self.part.special)
        if arm.special == "red":
            red = torch.tensor([float(arm.delta) * self.part.gamma], dtype=torch.float32, device=cand.device).reshape(-1, 1)
            return torch.where(sp, (-red).to(boost.dtype).expand_as(boost), boost)
        if arm.special != "neutral":
            raise ValueError(f"unknown special-token treatment {arm.special!r}")
        if not bool(sp.any()):
            return boost
        p = torch.softmax(lg.double() / self.T, -1).gather(1, cand)
        rest = (1.0 - (p * sp).sum(-1)).clamp_min(1e-12)
        other = (p * (torch.exp(boost.double() / self.T) - 1.0) * (~sp)).sum(-1)
        b = self.T * torch.log((rest + other).clamp_min(1e-300) / rest)
        return torch.where(sp, b[:, None].to(boost.dtype).expand_as(boost), boost)

    def _remove_repeated_own(self, x, r, positions, cand, boost, arm, P):
        """Unit-aware own term: a candidate whose unit (x_{j-2}, x_{j-1}, v) already occurs gets no own term."""
        if not arm.unit_aware:
            return boost
        xr = x[r].tolist()
        units = self._units(xr, P)
        if not units:
            return boost
        for q, j in enumerate(positions):
            a = xr[j - 2] if j - 2 >= P else -1
            b = xr[j - 1] if j - 1 >= P else -1
            if a == self.mask_id or b == self.mask_id:
                continue
            cl = cand[q].tolist()
            rep = [k for k, v in enumerate(cl) if v not in self.special and (a, b, v) in units]
            if not rep:
                continue
            mem = self._expected_member([[([a, b], 1.0)]], cand[q][None])[0]
            idx = torch.tensor(rep, device=boost.device)
            boost[q, idx] = boost[q, idx] - (arm.delta * (mem[idx] - self.part.gamma)).to(boost.dtype)
        return boost

    # ------------------------------------------------------------------ the tilt used to rank the masked positions of the block
    def _position_tilt(self, x, logits, r, pos_m, rows, P, step):
        """Candidates (n, k) and boosts (n, k) of every masked position pos_m of the block of row r, as if it were committed now, with
        foresight from the current beliefs; with the consistent order, the winning positions get their commit tilt."""
        arm = rows[r][1]
        lg = logits[r, pos_m, : self.V].float()
        cand = torch.softmax(lg / self.T, dim=-1).topk(self.k, dim=-1).indices
        if not arm.watermarked:
            return cand, torch.zeros(len(pos_m), self.k, device=x.device)
        boost = self._block_tilt(x, logits, r, pos_m, cand, arm, P)
        boost = self._remove_repeated_own(x, r, pos_m.tolist(), cand, boost.clone(), arm, P)
        boost = self._specials(boost, cand, arm, lg)
        open_foresight = arm.lam > 0 and (arm.foresight == "exact" or (arm.foresight == "predictor" and self.predictor is not None))
        if arm.order != "consistent" or arm.order_iters <= 0 or not open_foresight or len(pos_m) == 0:
            return cand, boost
        gb = gumbel_noise(x.device, rows[r][0], step, len(pos_m), self.V)       # the same noise as the caller's
        B = x.shape[0]
        done = set()
        for _ in range(arm.order_iters):
            tl = lg.scatter_add(1, cand, boost.to(lg.dtype))
            x0 = (tl + self.T * gb).argmax(-1)
            conf = torch.softmax(tl.double(), -1).gather(1, x0[:, None])[:, 0]
            q = int(conf.argmax())
            if q in done:
                break
            jt = torch.zeros(B, dtype=torch.long, device=x.device); jt[r] = pos_m[q]
            ct = torch.zeros(B, cand.shape[1], dtype=cand.dtype, device=x.device); ct[r] = cand[q]
            boost[q] = self._commit_boost(x, logits, jt, ct, [r], rows, P)[0].to(boost.dtype)
            done.add(q)
        return cand, boost

    def _block_tilt(self, x, logits, r, pos_m, cand, arm, P):
        """Own term and current-belief foresight for all masked positions of the block at once."""
        dev = x.device
        n = len(pos_m)
        delta, lam, g = float(arm.delta), float(arm.lam), self.part.gamma
        Lx = x.shape[1]
        xr = x[r].tolist()
        pm = pos_m.tolist()
        need = sorted({pa for m in pm for pa in range(m - 2, m + 3) if P <= pa < Lx and xr[pa] == self.mask_id})
        pr = torch.log_softmax(logits[r, torch.tensor(need, device=dev), : self.V].float() / self.T, -1).exp()
        p4, i4 = pr.topk(4, -1); p4 = p4 / p4.sum(-1, keepdim=True)
        p8, i8 = pr.topk(8, -1); p8 = p8 / p8.sum(-1, keepdim=True)
        row = {pa: q for q, pa in enumerate(need)}
        i4l, p4l, i8l, p8l = i4.tolist(), p4.tolist(), i8.tolist(), p8.tolist()

        def opts(pa):
            if pa < P:
                return [(-1, 1.0)]
            if xr[pa] != self.mask_id:
                return [(xr[pa], 1.0)]
            return list(zip(i4l[row[pa]], p4l[row[pa]]))

        def combos_of(slots):
            cmb = [([], 1.0)]
            for o in (opts(pa) for pa in slots):
                cmb = [(c + [u], w * pu) for c, w in cmb for u, pu in o]
            return cmb

        # own term: expectation over the beliefs about the context positions j-2, j-1
        flat, wts, own = [], [], []
        h = arm.context_width
        for q, m in enumerate(pm):
            if h == 2:
                cmb = combos_of([m - 2, m - 1])
            elif h == 1:
                cmb = [([-1] + c, w) for c, w in combos_of([m - 1])]
            else:
                cmb = self._wide_contexts(x, logits, r, list(range(m - h, m)), P)
            for c, w in cmb:
                flat.append(c); wts.append(w); own.append(q)
        own_t = torch.tensor(own, device=dev)
        mem = self.part.member_lists(flat, cand[own_t])
        em = torch.zeros(n, cand.shape[1], device=dev)
        em.index_add_(0, own_t, mem * torch.tensor(wts, device=dev)[:, None])
        boost = delta * (em - g)
        if lam <= 0:
            return boost
        # foresight from the current beliefs: neighbors j+1 (context (x_{j-1}, v)) and j+2 (context (v, x_{j+1})), top-kq candidates
        kq = min(self.kq, cand.shape[1])
        cl = cand[:, :kq].tolist()
        groups, flat, wts, own, tgts, tps = [], [], [], [], [], []
        for q, m in enumerate(pm):
            for t in (1, 2):
                pt = m + t
                if pt >= Lx:
                    continue
                other = m - 1 if t == 1 else m + 1
                oc = combos_of([other])
                gi = len(groups); groups.append(q)
                for vi, v in enumerate(cl[q]):
                    for u, w in oc:
                        flat.append([u[0], v] if t == 1 else [v, u[0]]); wts.append(w); own.append(gi * kq + vi)
                if xr[pt] != self.mask_id:
                    tgts.append([xr[pt]] + [0] * 7); tps.append([1.0] + [0.0] * 7)
                else:
                    tgts.append(i8l[row[pt]]); tps.append(p8l[row[pt]])
        Q = torch.zeros(n, cand.shape[1], device=dev)
        if not groups:
            return boost
        G = len(groups)
        tg = torch.tensor(tgts, device=dev).repeat_interleave(kq, 0)
        tp = torch.tensor(tps, device=dev).repeat_interleave(kq, 0)
        own_t = torch.tensor(own, device=dev)
        mem = self.part.member_lists(flat, tg[own_t])
        if arm.unit_aware:
            mem = self._repeated_to_gamma(xr, P, flat, tg[own_t], mem)
        agg = torch.zeros(G * kq, 8, device=dev)
        agg.index_add_(0, own_t, mem * torch.tensor(wts, device=dev)[:, None])
        green = (tp * agg).sum(-1).view(G, kq)
        for gi, q in enumerate(groups):
            Q[q, :kq] += green[gi] - g
        return boost + delta * lam * Q

    # ------------------------------------------------------------------ the tilt applied at the commit
    def _commit_boost(self, x, logits, j, cand, rr, rows, P):
        """Boosts (len(rr), k) of the candidates cand[r] of the positions j[r] committed by the rows rr."""
        dev = x.device
        boost = torch.zeros(len(rr), cand.shape[1], device=dev)
        for q, r in enumerate(rr):
            arm, jj = rows[r][1], int(j[r])
            em = self._expected_member([self._own_contexts(x, logits, r, jj, P, arm)], cand[r][None])[0]
            boost[q] = arm.delta * (em - self.part.gamma)
        la = [(q, r) for q, r in enumerate(rr) if rows[r][1].lam > 0]
        for q, r in la:
            arm = rows[r][1]
            boost[q] = boost[q] + arm.delta * arm.lam * self._current_foresight(x, logits, int(j[r]), cand[r], r, arm, P)
        boost = torch.stack([self._specials(boost[q:q + 1], cand[r][None], rows[r][1], logits[r, int(j[r]), : self.V].float()[None])[0]
                             for q, r in enumerate(rr)])
        states, block_in = self._step_states
        for q, r in enumerate(rr):
            arm = rows[r][1]
            use_pred = arm.lam > 0 and arm.foresight == "predictor" and self.predictor is not None and states is not None
            if not (use_pred or (arm.lam > 0 and arm.foresight == "exact")):
                continue
            self.stats["commits"] += 1
            jj, c = int(j[r]), cand[r]
            lj = logits[r, jj, : self.V].float()
            pq = torch.softmax((lj[c] + boost[q]) / self.T, -1)
            if float(pq.topk(2).values[1]) < arm.gate:
                continue
            self.stats["open"] += 1
            em = self._expected_member([self._own_contexts(x, logits, r, jj, P, arm)], c[None])[0]
            kq = min(self.kq, c.shape[0])
            if use_pred:
                fs = self._predicted_foresight(x, r, jj, c[:kq], P, arm, states[r], logits[r], None if block_in is None else block_in[r])
            else:
                fs = self._exact_foresight(x, r, jj, c[:kq], P, arm, logits[r])
            new = arm.delta * (em - self.part.gamma)
            new[:kq] = new[:kq] + arm.delta * arm.lam * fs
            boost[q] = new
        boost = torch.stack([self._specials(boost[q:q + 1], cand[r][None], rows[r][1], logits[r, int(j[r]), : self.V].float()[None])[0]
                             for q, r in enumerate(rr)])
        boost = boost.clone()
        for q, r in enumerate(rr):
            boost[q:q + 1] = self._remove_repeated_own(x, r, [int(j[r])], cand[r][None], boost[q:q + 1], rows[r][1], P)
            if rows[r][1].special == "neutral":       # the neutral boost depends on the final boosts of the other candidates
                boost[q:q + 1] = self._specials(boost[q:q + 1], cand[r][None], rows[r][1], logits[r, int(j[r]), : self.V].float()[None])
        return boost

    def _current_foresight(self, x, logits, jj, cand_r, r, arm, P):
        """sum_t (g_t(v) - gamma) for the top-kq candidates, with the neighbors' current beliefs. (k,)"""
        dev = x.device
        kq = min(self.kq, cand_r.shape[0])
        out = torch.zeros(cand_r.shape[0], device=dev)
        dd = (x[r].tolist(), P) if arm.unit_aware else None
        for t in (1, 2):
            pt = jj + t
            if pt >= x.shape[1]:
                continue
            other = jj - 1 if t == 1 else jj + 1
            oc = self._context_mixture(x, logits, r, [other], P)
            combos = [[(([u[0], v] if t == 1 else [v, u[0]]), w) for u, w in oc] for v in cand_r[:kq].tolist()]
            if int(x[r, pt]) != self.mask_id:
                tgt = torch.full((kq, 1), int(x[r, pt]), device=dev); tp = torch.ones(kq, 1, device=dev)
            else:
                ids, p = top_beliefs(logits[r, pt, : self.V][None], self.T, 8)
                tgt = ids.expand(kq, -1); tp = (p / p.sum()).expand(kq, -1)
            mem = self._expected_member(combos, tgt, dedup=dd)
            out[:kq] += (tp * mem).sum(-1) - self.part.gamma
        return out

    def _foresight_from_beliefs(self, x, r, jj, vs, P, arm, post):
        """sum_t (g_t(v) - gamma) for candidates vs, given post(p) = post-placement logits (kq, V) of position p for each candidate."""
        dev = x.device
        kq = len(vs)
        Lx = x.shape[1]
        flat, wts, own, tg, tp = [], [], [], [], []
        groups = 0
        for t in (1, 2):
            pt = jj + t
            if pt >= Lx:
                continue
            other = jj - 1 if t == 1 else jj + 1
            if int(x[r, pt]) != self.mask_id:
                t_ids = torch.full((kq, 8), int(x[r, pt]), device=dev); t_p = torch.zeros(kq, 8, device=dev); t_p[:, 0] = 1.0
            else:
                ids, p = top_beliefs(post(pt), self.T, 8)
                t_ids, t_p = ids, p / p.sum(-1, keepdim=True)
            if other < P:
                o_opts = [[(-1, 1.0)]] * kq
            elif int(x[r, other]) != self.mask_id:
                o_opts = [[(int(x[r, other]), 1.0)]] * kq
            else:
                ids, p = top_beliefs(post(other), self.T, 4)
                p = p / p.sum(-1, keepdim=True)
                o_opts = [list(zip(ids[q].tolist(), p[q].tolist())) for q in range(kq)]
            for q in range(kq):
                v = int(vs[q])
                for u, w in o_opts[q]:
                    flat.append([u, v] if t == 1 else [v, u]); wts.append(w); own.append(groups * kq + q)
            tg.append(t_ids); tp.append(t_p)
            groups += 1
        if not groups:
            return torch.zeros(kq, device=dev)
        tg = torch.cat(tg, 0); tp = torch.cat(tp, 0)
        own_t = torch.tensor(own, device=dev)
        mem = self.part.member_lists(flat, tg[own_t])
        if arm.unit_aware:
            mem = self._repeated_to_gamma(x[r].tolist(), P, flat, tg[own_t], mem)
        agg = torch.zeros(groups * kq, 8, device=dev)
        agg.index_add_(0, own_t, mem * torch.tensor(wts, device=dev)[:, None])
        green = (tp * agg).sum(-1).view(groups, kq)
        return (green - self.part.gamma).sum(0)

    def _predicted_foresight(self, x, r, jj, vs, P, arm, states_r, logits_r, block_in_r):
        net = self.predictor
        dev = x.device
        kq = len(vs)
        Lx = x.shape[1]
        c = net.c
        pos = jj + torch.arange(-c, c + 1, device=dev)
        ok = (pos >= 0) & (pos < Lx)
        pc = pos.clamp(0, Lx - 1)
        typ = torch.full_like(pc, TYPES["committed"])
        typ[x[r, pc] == self.mask_id] = TYPES["masked"]; typ[pos < P] = TYPES["prompt"]; typ[~ok] = TYPES["outside"]; typ[c] = TYPES["slot"]
        lpv = torch.log_softmax(logits_r[jj, : self.V].float() / self.T, -1)[vs]
        blk = None if block_in_r is None else block_in_r[pc][None].expand(kq, -1, -1)
        zz = net.hidden(states_r[pc][None].expand(kq, -1, -1), typ[None].expand(kq, -1), vs, lpv, blk, rep=kq)
        cache = {}

        def post(p):
            if p not in cache:
                cache[p] = net.logits(zz[:, c + p - jj], self.V)
            return cache[p]
        return self._foresight_from_beliefs(x, r, jj, vs, P, arm, post)

    def _exact_foresight(self, x, r, jj, vs, P, arm, logits_r):
        """Re-runs the generator once per candidate (a batch of kq canvases with v placed at jj)."""
        dev = x.device
        kq = len(vs)
        xb = x[r].unsqueeze(0).repeat(kq, 1)
        xb[torch.arange(kq, device=dev), jj] = vs
        lg = self._forward(xb)
        if self.on_exact is not None:
            self.on_exact(x=x, r=r, jj=jj, vs=vs, P=P, logits_r=logits_r, post_logits=lg, states=self._step_states)
        return self._foresight_from_beliefs(x, r, jj, vs, P, arm, lambda p: lg[:, p, : self.V])
