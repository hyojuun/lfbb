"""The watermark on Dream-v0-Instruct-7B with its official decoding.

Dream decodes one canvas of gen_length masked positions in `steps` steps ("origin" schedule): at step i every still-masked position is
revealed independently with probability 1 - t_{i+1} / t_i, t = linspace(1, eps, steps + 1), and every remaining position at the last
step; the token of a revealed position is sampled at temperature T. Dream's logits are shifted by one position (the output at index
i - 1 is the belief about position i), and so are the final states the predictor reads.
The tilt of a revealed position is the one of WatermarkSampler. With predicted or exact foresight, every revealed position gets its
commit tilt (foresight replaced at decision states).
"""
import torch

from .sampler import WatermarkSampler, make_generator

DREAM_MASK_ID = 151666
DREAM_EOS_ID = 151643            # responses end at the first <|endoftext|>
DREAM_FIRST_SPECIAL = 151643     # ids from here on are special tokens


class DreamWatermarkSampler(WatermarkSampler):
    def __init__(self, model, vocab_size, partition, predictor=None, T=0.5, gen_length=256, steps=256, eps=1e-3, k=50, device="cuda",
                 capture=False):
        self._capture_states = predictor is not None or capture
        super().__init__(model, vocab_size, partition, predictor, T=T, gen_length=gen_length, block_length=gen_length, k=k,
                         device=device, mask_id=DREAM_MASK_ID)
        self.steps, self.eps = steps, eps

    def _register_hooks(self, predictor, capture_K):
        if self._capture_states:
            self.model.lm_head.register_forward_hook(self._capture)

    def _capture(self, m, inp, out):
        h = inp[0].detach()
        self._states = torch.cat([h[:, :1], h[:, :-1]], 1)

    def _forward(self, x):
        lg = self.model(x, "full", None, use_cache=False).logits
        return torch.cat([lg[:, :1], lg[:, :-1]], 1)

    @torch.no_grad()
    def generate(self, prompt_ids, rows):
        dev, B, P = self.device, len(rows), len(prompt_ids)
        x = torch.full((B, P + self.gen_length), self.mask_id, dtype=torch.long, device=dev)
        x[:, :P] = torch.tensor(prompt_ids, device=dev)
        kl_sum = [0.0] * B
        ts = torch.linspace(1, self.eps, self.steps + 1, device=dev)
        for i in range(self.steps):
            masked = x == self.mask_id
            if not bool(masked.any()):
                break
            logits = self._forward(x)
            self._step_states = (self._states, None)
            p_reveal = float(1 - ts[i + 1] / ts[i]) if i < self.steps - 1 else 2.0
            reveal = []
            for r in range(B):
                pos = masked[r].nonzero().squeeze(-1)
                if len(pos) == 0:
                    reveal.append(None)
                    continue
                u = torch.rand(len(pos), generator=make_generator(dev, rows[r][0], i, 3), device=dev)
                rev = pos[u < p_reveal]
                reveal.append(rev if len(rev) else None)
            for r in range(B):
                rev = reveal[r]
                if rev is None:
                    continue
                n = len(rev)
                lg = logits[r, rev, : self.V].float()
                uu = torch.rand(n, self.V, generator=make_generator(dev, rows[r][0], i, 4), device=dev).clamp_(1e-10, 1 - 1e-10)
                samp = lg + self.T * (-torch.log(-torch.log(uu)))
                arm = rows[r][1]
                if arm.watermarked:
                    cand, boost = self._revealed_tilt(x, logits, r, rev, rows, P)
                    tok = samp.scatter_add(1, cand, boost.to(samp.dtype)).argmax(-1)
                    lp0 = torch.log_softmax(lg / self.T, -1)
                    lps = torch.log_softmax(lg.scatter_add(1, cand, boost.to(lg.dtype)) / self.T, -1)
                    kl_sum[r] += float((lps.exp() * (lps - lp0)).sum(-1).clamp_min(0).sum())
                else:
                    tok = samp.argmax(-1)
                if self.on_commit is not None:
                    top = torch.softmax(lg / self.T, dim=-1).topk(self.k, dim=-1)
                    self.on_commit(x=x, logits=logits, r=r, positions=rev, tok=tok, cand=top.indices, cand_p=top.values, P=P, step=i)
                reveal[r] = (rev, tok)
            for r in range(B):
                if reveal[r] is not None:
                    x[r, reveal[r][0]] = reveal[r][1]
        return [{"tokens": x[r, P:].tolist(), "kl": kl_sum[r]} for r in range(B)]

    def _revealed_tilt(self, x, logits, r, rev, rows, P):
        arm = rows[r][1]
        cand = torch.softmax(logits[r, rev, : self.V].float() / self.T, dim=-1).topk(self.k, dim=-1).indices
        boost = self._block_tilt(x, logits, r, rev, cand, arm, P)
        boost = self._remove_repeated_own(x, r, rev.tolist(), cand, boost.clone(), arm, P)
        boost = self._specials(boost, cand, arm, logits[r, rev, : self.V].float())
        if arm.lam > 0 and arm.foresight in ("predictor", "exact"):
            B = x.shape[0]
            boost = boost.clone()
            for q, m in enumerate(rev.tolist()):
                jt = torch.zeros(B, dtype=torch.long, device=x.device); jt[r] = m
                ct = torch.zeros(B, cand.shape[1], dtype=torch.long, device=x.device); ct[r] = cand[q]
                boost[q] = self._commit_boost(x, logits, jt, ct, [r], rows, P)[0].to(boost.dtype)
        return cand, boost
