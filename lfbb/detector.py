"""Text-only detection. Needs the key and the partition, not the language model.

Every distinct unit (x_{i-2}, x_{i-1}, x_i) of the token sequence is counted once (positions before the text are padding, special
tokens are skipped), and it is green iff band(c, x_i) is the keyed green band of its context c. Tokens that follow the same context
share that context's green band, so under the null hypothesis (text independent of the key) the green count of a context group is the
number of its tokens in a uniformly chosen band, independently across contexts. The exact null of the total count is the convolution of
these group laws, which reduces to Binomial(n, 1/B) when every context is followed by a single token.
"""
import math

import numpy as np
import torch

from .partition import fold


def grouped_test(groups, nb):
    """groups: per distinct context, (green band, [band of each distinct token that follows it]).
    Returns (n, k, z, p): units, green units, z-score under the exact null and the one-sided p-value P(K >= k)."""
    pmf = np.ones(1)
    n = k = 0
    mean = var = 0.0
    for green, bands in groups:
        cnt = np.bincount(np.asarray(bands, dtype=np.int64), minlength=nb)
        m = int(cnt.sum())
        n += m
        k += int(cnt[green])
        law = np.bincount(cnt, minlength=m + 1).astype(np.float64) / nb
        mean += m / nb
        var += float((cnt.astype(np.float64) ** 2).mean()) - (m / nb) ** 2
        pmf = np.convolve(pmf, law)
    p = float(min(1.0, pmf[k:].sum())) if n else 1.0
    z = (k - mean) / math.sqrt(var) if var > 0 else float("nan")
    return n, k, z, p


def context_of(ids, i, width=2):
    """The context pair hashed for position i: (x_{i-2}, x_{i-1}), or (pad, x_{i-1}) for width 1, or (fold(x_{i-h}..x_{i-2}), x_{i-1})
    for width h >= 3; positions before the text are padding (-1)."""
    last = ids[i - 1] if i >= 1 else -1
    if width == 1:
        return -1, last
    if width == 2:
        return (ids[i - 2] if i >= 2 else -1), last
    return fold([ids[i - k] if i - k >= 0 else -1 for k in range(width, 1, -1)]), last


@torch.no_grad()
def detect(partition, token_ids, context_width=2):
    """Detection on the token ids of a text (re-tokenized, without special tokens at the ends). Returns (n, k, z, p)."""
    ids = [int(t) for t in token_ids]
    if not ids:
        return 0, 0, float("nan"), 1.0
    special = set(partition.special.tolist())
    seen, units = set(), []
    for i, t in enumerate(ids):
        if t in special:
            continue
        unit = context_of(ids, i, context_width) + (t,)
        if unit not in seen:
            seen.add(unit)
            units.append(unit)
    if not units:
        return 0, 0, float("nan"), 1.0
    dev = partition.device
    ctx = torch.tensor([u[:2] for u in units], dtype=torch.long, device=dev)
    tok = torch.tensor([u[2] for u in units], dtype=torch.long, device=dev)[:, None]
    band = partition.bands(ctx, tok)[:, 0].cpu().tolist()
    green = partition.green_band(ctx).cpu().tolist()
    groups = {}
    for u, b, g in zip(units, band, green):
        groups.setdefault(u[:2], [g, []])[1].append(b)
    return grouped_test(list(groups.values()), partition.nb)
