"""The foresight predictor: an estimate of the generator's post-placement beliefs about the neighbors of a candidate.

Given the inputs h_{j-6..j+6} of the generator's top K transformer blocks in the current forward pass and its final states r, an
adapter rewrites the input at the committed position j for candidate v, h'_j = h_j + a(LN(h_j), E[v], log p(v)), and copies of the K
blocks are run on the 13-position window with and without the rewrite. The predicted final states of a neighbor j+d are
r_{j+d} + LN_f(B(h'))_{j+d} - LN_f(B(h))_{j+d}, read out with the generator's output projection W at temperature T. With a zero adapter
the prediction equals the current belief.
"""
import copy
import re

import torch
import torch.nn as nn

TYPES = {"masked": 0, "committed": 1, "prompt": 2, "outside": 3, "slot": 4}
WINDOW = 13


class ForesightPredictor(nn.Module):
    def __init__(self, emb, W, blocks, lnf, K=8, ktrain=8, win=WINDOW, hid=1024):
        super().__init__()
        half = lambda t: t if t.dtype in (torch.float16, torch.bfloat16) else t.half()
        self.register_buffer("emb", half(emb), persistent=False)        # generator input embeddings (frozen)
        self.register_buffer("W", half(W), persistent=False)            # generator output projection (frozen)
        D = W.shape[1]
        assert len(blocks) == K and 1 <= ktrain <= K
        self._frozen = list(blocks[: K - ktrain])                        # shared with the generator, not part of the state dict
        for b in self._frozen:
            for p in b.parameters():
                p.requires_grad_(False)
        self.blocks = nn.ModuleList(blocks[K - ktrain:])
        self.lnf = lnf
        for p in self.lnf.parameters():
            p.requires_grad_(False)
        self.ln_x = nn.LayerNorm(D)
        self.adapt = nn.Sequential(nn.Linear(2 * D + 1, hid), nn.GELU(), nn.Linear(hid, D))
        nn.init.zeros_(self.adapt[-1].weight)
        nn.init.zeros_(self.adapt[-1].bias)
        self.win, self.c = win, win // 2
        self.cfg = {"arch": "blockk", "K": K, "ktrain": ktrain, "win": win, "hid": hid}

    def to_infer(self):
        """bf16 blocks and final norm, as in the generator (the adapter stays in fp32)."""
        for b in self.blocks:
            b.to(torch.bfloat16)
        self.lnf.to(torch.bfloat16)
        return self

    def run(self, x, bias):
        # .forward instead of __call__: the frozen blocks are the generator's own modules, whose capture hooks must not fire here
        for b in self._frozen:
            x, _ = b.forward(x.to(torch.bfloat16), attention_bias=bias.to(torch.bfloat16))
        for b in self.blocks:
            x, _ = b.forward(x, attention_bias=bias.to(x.dtype))
        return x

    def hidden(self, states, types, v, logp_v, block_inputs, rep=1):
        """states (N, win, D) final states and block_inputs (N, win, D) inputs of the top K blocks of the current canvas, types (N, win),
        candidates v (N,) and their log-probabilities logp_v (N,) at the window center. Rows come in groups of `rep` candidates of one
        canvas, so the unchanged window is run once per group. Returns the predicted post-placement final states (N, win, D)."""
        dt = next(self.blocks.parameters()).dtype
        x = block_inputs.to(dt)
        a = self.adapt(torch.cat([self.ln_x(x[:, self.c].float()), self.emb[v].float(), logp_v.float()[:, None]], -1))
        x1 = torch.cat([x[:, : self.c], (x[:, self.c].float() + a).to(dt)[:, None], x[:, self.c + 1:]], 1)
        bias = torch.zeros(x.shape[0], 1, 1, x.shape[1], device=x.device, dtype=dt)
        bias = bias.masked_fill((types == TYPES["outside"])[:, None, None, :], float("-inf"))
        y1 = self.run(x1, bias)
        y0 = self.run(x[::rep], bias[::rep])
        if rep > 1:
            y0 = y0.repeat_interleave(rep, 0)
        return states.float() + (self.lnf(y1).float() - self.lnf(y0).float())

    def logits(self, z, vocab_size):
        """(N, D) final states -> (N, vocab_size) logits."""
        return (z.to(self.W.dtype) @ self.W.t()).float()[:, :vocab_size]


def generator_parts(model):
    """The generator's transformer blocks (ordered), final LayerNorm, input embeddings and output projection (LLaDA naming)."""
    blocks = sorted(((int(n.rsplit(".", 1)[1]), m) for n, m in model.named_modules() if re.fullmatch(r".*transformer\.blocks\.\d+", n)),
                    key=lambda t: t[0])
    lnf = next(m for n, m in model.named_modules() if n.endswith("transformer.ln_f"))
    W = next(p for n, p in model.named_parameters() if n.endswith("transformer.ff_out.weight"))
    return [m for _, m in blocks], lnf, model.get_input_embeddings().weight.detach(), W.detach()


def load_predictor(path, model, device="cuda", fp32=False):
    """Builds the predictor on top of the generator `model` and loads a checkpoint {"state", "cfg"} (scripts/train_predictor.py).
    fp32=True keeps the fine-tuned blocks in fp32 as during training (decoding runs them in bf16)."""
    ck = torch.load(path, map_location="cpu")
    cfg = {k: v for k, v in ck["cfg"].items() if k != "arch"}
    blocks, lnf, emb, W = generator_parts(model)
    K, ktrain = int(cfg["K"]), int(cfg["ktrain"])
    top = blocks[len(blocks) - K:]
    top = top[: K - ktrain] + [copy.deepcopy(b) for b in top[K - ktrain:]]      # fine-tuned copies of the top ktrain blocks
    net = ForesightPredictor(emb, W, top, copy.deepcopy(lnf), **cfg)
    net.load_state_dict(ck["state"], strict=False)
    if fp32:
        net.to(device)
        for b in net.blocks:
            b.float()
        net.lnf.float()
    else:
        net.to(device).to_infer()
    return net.eval()


class WindowPredictor(nn.Module):
    """A predictor that does not reuse the generator's blocks (used for Dream): a small bidirectional transformer over the window of
    final states, h = Encoder(LN(r_i) A + type_i + pos_i, plus E[v] B + log p(v) c at the center), with predicted final states
    r + Out(h). The output layer starts at zero, so the untrained predictor returns the current beliefs."""

    def __init__(self, emb, W, win=WINDOW, dm=1024, layers=4, heads=16, ff=4096, drop=0.1):
        super().__init__()
        half = lambda t: t if t.dtype in (torch.float16, torch.bfloat16) else t.half()
        self.register_buffer("emb", half(emb), persistent=False)
        self.register_buffer("W", half(W), persistent=False)
        D = W.shape[1]
        self.win, self.c = win, win // 2
        self.inp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, dm))
        self.vin = nn.Linear(emb.shape[1], dm)
        self.lpin = nn.Linear(1, dm)
        self.typ = nn.Embedding(len(TYPES), dm)
        self.pos = nn.Embedding(win, dm)
        layer = nn.TransformerEncoderLayer(dm, heads, ff, drop, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, D))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)
        self.cfg = {"win": win, "dm": dm, "layers": layers, "heads": heads, "ff": ff, "drop": drop}

    def hidden(self, states, types, v, logp_v, block_inputs=None, rep=1):
        h = self.inp(states.float()) + self.typ(types) + self.pos.weight[None]
        add = self.vin(self.emb[v].float()) + self.lpin(logp_v.float()[:, None])
        h = torch.cat([h[:, : self.c], h[:, self.c: self.c + 1] + add[:, None], h[:, self.c + 1:]], 1)
        return states.float() + self.out(self.enc(h))

    def logits(self, z, vocab_size):
        return (z.to(self.W.dtype) @ self.W.t()).float()[:, :vocab_size]


def load_window_predictor(path, emb, W, device="cuda"):
    ck = torch.load(path, map_location="cpu")
    cfg = {k: v for k, v in ck["cfg"].items() if k != "arch"}
    net = WindowPredictor(emb, W, **cfg).to(device).eval()
    net.load_state_dict(ck["state"])
    return net
