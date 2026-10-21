"""A partition learned for LLaDA-8B-Instruct applied to Dream-v0-Instruct-7B through the tokens whose strings are identical in the two
vocabularies (Table 4, transferred rows).

The map sends every Dream token to the LLaDA token with the same byte-level string, or to -1 when there is none; special tokens of
either model (Dream ids from 151643 on, data/special_ids_llada.json) are never matched. scripts/generate.py --transfer MAP applies the
LLaDA partition to Dream ids through it (TransferredBands): the band of a token and the keyed green band of a context are those of the
matched LLaDA ids; the tokens without a match, Dream's special tokens among them, are the partition's special tokens, which the
sampler tilts as red; and a candidate after a context token without a match has membership gamma. The detector (detect_transferred)
skips the positions whose token or context has no match and counts every distinct unit of the matched ids once, with the exact test of
lfbb/detector.py.

Output: {"map": (V_dream,) int64 tensor, "source", "target", "matched"}.

Example:
  python scripts/transfer_partition.py --out checkpoints/llada_to_dream.pt
  python scripts/generate.py --family dream --arms configs/arms/dream_current.json --partition checkpoints/bands.pt \
      --transfer checkpoints/llada_to_dream.pt --arm-prefix transfer_ --prompts data/waterbench600_dream.jsonl --out runs/dream_transfer
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lfbb.detector import context_of, grouped_test  # noqa: E402
from lfbb.dream import DREAM_FIRST_SPECIAL  # noqa: E402
from lfbb.partition import Partition, load_special_ids  # noqa: E402


def build_map(source_vocab, target_vocab, target_first_special, source_special=()):
    """List over the target ids: the source id of the identical token string, -1 if none (special tokens are never matched)."""
    sp = set(int(t) for t in source_special)
    out = [-1] * (max(target_vocab.values()) + 1)
    for s, i in target_vocab.items():
        j = source_vocab.get(s)
        if i < target_first_special and j is not None and j not in sp:
            out[i] = int(j)
    return out


class TransferredBands(Partition):
    """The partition `source` of another vocabulary, applied through the token map (target id -> source id, -1 if unmatched); the
    unmatched target ids are special tokens, in addition to special_ids."""

    def __init__(self, source, mapping, special_ids=(), device="cuda"):
        lut = [int(j) for j in torch.as_tensor(mapping, dtype=torch.long).tolist()]
        special = sorted(set(int(t) for t in special_ids) | {i for i, j in enumerate(lut) if j < 0})
        super().__init__(source.nb, source.key, special, device)
        self.source, self.lut = source, lut
        self.map = torch.as_tensor(lut, dtype=torch.long).to(device)

    def mapped(self, ids):
        """(source ids, unmatched mask) of an int64 tensor of target ids; padding (-1) stays -1, unmatched ids become 0."""
        ids = ids.long()
        m = self.map[ids.clamp(min=0)]
        return torch.where(ids < 0, ids, m.clamp(min=0)), (ids >= 0) & (m < 0)

    def green_band(self, ctx):
        return super().green_band(self.mapped(ctx)[0])

    def bands(self, ctx, v):
        self.source.key = self.key
        return self.source.bands(self.mapped(ctx)[0], self.mapped(v)[0])

    def member(self, ctx, v):
        m = super().member(ctx, v)
        return torch.where(self.mapped(ctx)[1].any(-1)[:, None], torch.full_like(m, self.gamma), m)


@torch.no_grad()
def detect_transferred(part, token_ids, context_width=2):
    """lfbb.detector.detect for TransferredBands: positions whose token or context token has no match are skipped, and the units
    (context, token) are those of the matched source ids. Returns (n, k, z, p)."""
    if context_width not in (1, 2):
        raise ValueError("the transferred partition supports context widths 1 and 2")
    lut = part.lut
    ids = [lut[int(t)] if 0 <= int(t) < len(lut) else -1 for t in token_ids]
    seen, units = set(), []
    for i, t in enumerate(ids):
        if min(ids[max(0, i - context_width): i + 1]) < 0:
            continue
        unit = context_of(ids, i, context_width) + (t,)
        if unit not in seen:
            seen.add(unit)
            units.append(unit)
    if not units:
        return 0, 0, float("nan"), 1.0
    dev = part.device
    ctx = torch.tensor([u[:2] for u in units], dtype=torch.long, device=dev)
    tok = torch.tensor([u[2] for u in units], dtype=torch.long, device=dev)[:, None]
    part.source.key = part.key
    band = part.source.bands(ctx, tok)[:, 0].cpu().tolist()
    green = part.source.green_band(ctx).cpu().tolist()
    groups = {}
    for u, b, g in zip(units, band, green):
        groups.setdefault(u[:2], [g, []])[1].append(b)
    return grouped_test(list(groups.values()), part.nb)


def main():
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-model", default="GSAI-ML/LLaDA-8B-Instruct", help="tokenizer of the model the partition was learned for")
    ap.add_argument("--target-model", default="Dream-org/Dream-v0-Instruct-7B")
    ap.add_argument("--source-special-ids", default=os.path.join(os.path.dirname(__file__), "..", "data", "special_ids_llada.json"))
    ap.add_argument("--target-first-special", type=int, default=DREAM_FIRST_SPECIAL, help="target ids from here on are special tokens")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = AutoTokenizer.from_pretrained(args.source_model, trust_remote_code=True).get_vocab()
    tgt = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True).get_vocab()
    m = build_map(src, tgt, args.target_first_special, load_special_ids(args.source_special_ids))
    matched = sum(j >= 0 for j in m)
    print(f"target tokens {len(m)}, matched {matched}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"map": torch.tensor(m, dtype=torch.long), "source": args.source_model, "target": args.target_model, "matched": matched},
               args.out)


if __name__ == "__main__":
    main()
