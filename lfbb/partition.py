"""Partitions of the vocabulary into bands and the keyed choice of the green band.

A token v that follows the two-token context c = (c1, c2) is green iff band(c, v) == H_K(c) mod B, where H_K is a keyed hash of the
context token ids. Because the green band is chosen by the key alone, a text that does not depend on the key has each counted unit
green with probability exactly 1/B, for any partition (Proposition 2 of the paper).

Two partitions are implemented:
  LearnedBands    band(c, v) = color[v], a table learned by coordinate ascent (scripts/learn_bands.py);
  KeyedHashBands  band(c, v) = a keyed hash of (c, v), the standard green-red construction (ablation).
Special tokens (end of text, chat-template tokens) are never counted by the detector: their membership is reported as gamma, and the
sampler tilts them as red tokens.
"""
import json

import torch

P31 = 2147483647
HASH_SALT = 7919          # independent salt of the (context, token) hash of KeyedHashBands


def _mix(x):
    for _ in range(4):
        x = (x * 48271) % P31
        x = x ^ (x >> 13)
        x = (x * 69621) % P31
    return x


def fold(tokens):
    """Integer hash of a token tuple (oldest first, -1 = padding) with fold([t]) = t; folds contexts wider than two tokens into the
    first slot of the hashed pair."""
    if len(tokens) == 1:
        return int(tokens[0])
    x = 7
    for t in tokens:
        x = (x * 1_000_003 + (int(t) + 2)) % P31
    return x


def context_hash(c1, c2, key, mod):
    """Keyed hash of the context token ids (int64 tensors, -1 = padding) reduced modulo `mod`. All intermediate values stay below 2^62,
    so the result is the same on CPU and GPU."""
    x = ((c1.long() + 2) * 1_000_003 + (c2.long() + 2)) % P31
    x = (x + (int(key) % P31) * 2654435761) % P31
    return (_mix(x) >> 7) % mod


class Partition:
    """Base class. Contexts are (N, 2) int64 tensors of token ids, -1 marking a position before the response."""

    def __init__(self, nb, key, special_ids, device="cuda"):
        self.nb, self.key = int(nb), int(key)
        self.gamma = 1.0 / self.nb
        self.special = torch.as_tensor(list(special_ids), dtype=torch.long, device=device)
        self.device = device

    def green_band(self, ctx):
        return context_hash(ctx[:, 0], ctx[:, 1], self.key, self.nb)

    def bands(self, ctx, v):
        """(N, K) band of the tokens v (N, K) after the contexts ctx (N, 2)."""
        raise NotImplementedError

    def member(self, ctx, v):
        """(N, K) float: 1 if token v[n, k] is green after context ctx[n], 0 if red, gamma for special tokens."""
        m = (self.bands(ctx, v) == self.green_band(ctx)[:, None]).float()
        return torch.where(torch.isin(v, self.special), torch.full_like(m, self.gamma), m)

    def member_lists(self, contexts, v):
        """member() for contexts given as Python lists [[c1, c2], ...]."""
        return self.member(torch.tensor(contexts, dtype=torch.long, device=v.device), v)


class LearnedBands(Partition):
    def __init__(self, color, nb, key, special_ids, device="cuda"):
        super().__init__(nb, key, special_ids, device)
        self.color = torch.as_tensor(color, dtype=torch.long).to(device)

    def bands(self, ctx, v):
        return self.color[v.long().clamp(min=0)]


class KeyedHashBands(Partition):
    def bands(self, ctx, v):
        x = context_hash(ctx[:, 0], ctx[:, 1], self.key + HASH_SALT, P31)[:, None].expand_as(v)
        x = (x * 1_000_003 + v.long() + 2) % P31
        return (_mix(x) >> 7) % self.nb


def load_special_ids(path):
    with open(path) as f:
        return json.load(f)


def load_partition(spec, key=None, special_ids=(), device="cuda", nb=4):
    """spec = "hash" (or "hash:B" for B bands) for the keyed hash partition, or the path of a learned band table
    (scripts/learn_bands.py). key = None keeps the key stored with the table (the default key is 42)."""
    if spec == "hash" or spec.startswith("hash:"):
        nb = int(spec.split(":")[1]) if ":" in spec else nb
        return KeyedHashBands(nb, 42 if key is None else key, special_ids, device)
    st = torch.load(spec, map_location="cpu")
    cfg = st.get("cfg", st)
    k = int(cfg.get("key", 42)) if key is None else key
    return LearnedBands(st["color"], int(cfg["nb"]), k, special_ids, device)
