"""Synthetic ligand and receptor graphs, importable without pytest.

Inside the package rather than under `tests/` for two reasons, both of which bit once:
`tests/` imports pytest, which the image does not carry, and `.containerignore` excludes the
directory outright — so a fixture module living there is unreachable from any check that runs
where the model runs. `tests/conftest.py` re-exports these as fixtures for the pytest suite.

Graphs built here are structurally valid and numerically meaningless. They exist so that
checks about *shapes and wiring* — does the encoder detach from its head, does a transferred
encoder embed identically — need no data, no checkpoint and no network.
"""

import torch
from torch_geometric.data import Data, Batch

from equibind.features.process_mols import (lig_feature_dims, rec_atom_feature_dims,
                                            rec_residue_feature_dims)

# ── graph sizes used by all tests ──────────────────────────────────────────
N_LIG, E_LIG = 10, 30
N_REC, E_REC = 20, 60


def _make_lig(n: int = N_LIG, e: int = E_LIG, seed: int = 0) -> Data:
    g = torch.Generator().manual_seed(seed)
    n_cat = len(lig_feature_dims[0])
    n_scal = lig_feature_dims[1]
    x = torch.zeros(n, n_cat + n_scal)                  # all-zero → valid class index 0
    pos = torch.randn(n, 3, generator=g)
    edge_index = torch.randint(0, n, (2, e), generator=g)
    edge_attr = torch.rand(e, 15, generator=g)
    mu_r_norm = torch.rand(n, 5, generator=g) + 0.1     # strictly positive (we take log)
    return Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, mu_r_norm=mu_r_norm)


def _make_rec(n: int = N_REC, e: int = E_REC, seed: int = 1, atom_level: bool = False) -> Data:
    g = torch.Generator().manual_seed(seed)
    dims = rec_atom_feature_dims if atom_level else rec_residue_feature_dims
    n_cat = len(dims[0])
    n_scal = dims[1]
    x = torch.zeros(n, n_cat + n_scal)
    pos = torch.randn(n, 3, generator=g)
    edge_feat_dim = 27
    edge_index = torch.randint(0, n, (2, e), generator=g)
    edge_attr = torch.rand(e, edge_feat_dim, generator=g)
    mu_r_norm = torch.rand(n, 5, generator=g) + 0.1
    return Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, mu_r_norm=mu_r_norm)
