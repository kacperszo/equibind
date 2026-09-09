"""Shared pytest fixtures for synthetic molecular graphs.

The builders live in `synthetic.py`, which imports no pytest, so scripts that run inside
the image can use them too.
"""

import pytest
from torch_geometric.data import Batch

from equibind.synthetic import N_LIG, E_LIG, N_REC, E_REC, _make_lig, _make_rec  # noqa: F401

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
