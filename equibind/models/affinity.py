"""An affinity head for the IEGMN encoder. Ours, and labelled as such.

EquiBind has no affinity head — it was trained to *place* a ligand, and the head it does have
solves for a rigid transform. That is exactly what makes it the interesting encoder in this
benchmark: every other model here was trained on the number we are asking it to predict.

`gnnb probe` already asked the frozen version of this question and answered R = 0.428, below
the 0.470 a linear model gets from counting heavy atoms. That is a result about a *frozen*
representation read by a *linear* probe. Whether the same representation, fine-tuned, can
compete is a different question, and it needs a head to ask it with.

**The head starts exactly where the embedding ends.** `embed_complexes.py` writes
`cat([pool(h_lig), pool(h_rec)])`, so that concatenation is the head's input and nothing
between the encoder and the head is learned. Any other choice would make the probe number and
the fine-tune describe different representations.

**Sum pooling hands the model molecule size for free.** It is the default because it is what
the declared embedding uses, and because a benchmark should not quietly diverge from what it
recorded — but `--pool mean` is the ablation that removes the shortcut, and the gap between
the two is worth reporting whenever this head is trained.
"""

from __future__ import annotations

import torch
import torch.nn as nn

POOLS = {"sum": lambda t: t.sum(0), "mean": lambda t: t.mean(0), "max": lambda t: t.max(0).values}


class AffinityHead(nn.Module):
    """Pooled ligand and receptor embeddings to one scalar."""

    def __init__(self, feats_dim: int = 64, hidden: tuple[int, ...] = (128, 64),
                 dropout: float = 0.1, pool: str = "sum"):
        super().__init__()
        if pool not in POOLS:
            raise ValueError(f"pool must be one of {sorted(POOLS)}, got {pool!r}")
        self.pool_name = pool
        self.pool = POOLS[pool]
        self.feats_dim = feats_dim

        layers: list[nn.Module] = []
        width = 2 * feats_dim          # ligand and receptor, concatenated
        for size in hidden:
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)]
            width = size
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(width, 1)

    def forward(self, h_lig: torch.Tensor, h_rec: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat([self.pool(h_lig), self.pool(h_rec)])
        return self.out(self.mlp(pooled))


class EquiBindAffinity(nn.Module):
    """The published encoder with an affinity head on top.

    The attribute is named `iegmn` on purpose: that is what the authors' own checkpoint calls
    the encoder, so an encoder cut from it with `gnnb encoder split` loads here under the names
    it already carries rather than needing a rename in between.
    """

    def __init__(self, encoder: nn.Module, feats_dim: int = 64, pool: str = "sum",
                 dropout: float = 0.1, hidden: tuple[int, ...] = (128, 64)):
        super().__init__()
        self.iegmn = encoder
        self.affinity = AffinityHead(feats_dim=feats_dim, hidden=hidden,
                                     dropout=dropout, pool=pool)

    def forward(self, lig_graph, rec_graph, geom_graph) -> torch.Tensor:
        h_lig, h_rec = self.iegmn(lig_graph, rec_graph, geom_graph)[:2]
        # Checked rather than assumed: the head is sized from `feats_dim` at construction, and
        # a mismatch here would otherwise surface as a shape error inside the first Linear,
        # several frames from the configuration that caused it.
        if h_lig.shape[1] != self.affinity.feats_dim:
            raise ValueError(
                f"the encoder emits {h_lig.shape[1]}-dim node features but the head was built "
                f"for {self.affinity.feats_dim}; pass feats_dim to match iegmn_lay_hid_dim")
        return self.affinity(h_lig, h_rec)
