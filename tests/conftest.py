"""Shared pytest fixtures for synthetic molecular graphs."""

import pytest
import torch
from torch_geometric.data import Data, Batch

from equibind.features.process_mols import lig_feature_dims, rec_residue_feature_dims, rec_atom_feature_dims

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


@pytest.fixture
def lig_data() -> Data:
    return _make_lig()


@pytest.fixture
def rec_data() -> Data:
    return _make_rec()


@pytest.fixture
def lig_batch() -> Batch:
    return Batch.from_data_list([_make_lig(8, 20, seed=0), _make_lig(12, 35, seed=2)])


@pytest.fixture
def rec_batch() -> Batch:
    return Batch.from_data_list([_make_rec(15, 40, seed=1), _make_rec(25, 80, seed=3)])
