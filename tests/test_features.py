"""Tests for AtomEncoder and feature dimension constants."""

import pytest
import torch

from equibind.features.process_mols import (
    AtomEncoder,
    lig_feature_dims,
    rec_residue_feature_dims,
    rec_atom_feature_dims,
)

EMB_DIM = 32


# ── feature dim constants ──────────────────────────────────────────────────

class TestFeatureDims:
    def test_lig_structure(self):
        cat, scal = lig_feature_dims
        assert isinstance(cat, list)
        assert len(cat) == 16          # 16 categorical atom features
        assert scal == 1               # one scalar (Gasteiger charge)
        assert all(v > 1 for v in cat)

    def test_rec_residue_structure(self):
        cat, scal = rec_residue_feature_dims
        assert isinstance(cat, list)
        assert len(cat) == 1           # amino-acid type only
        assert scal == 2               # SASA + B-factor

    def test_rec_atom_structure(self):
        cat, scal = rec_atom_feature_dims
        assert isinstance(cat, list)
        assert len(cat) == 4           # AA type, atomic num, atom-type-2, atom-type-3
        assert scal == 2               # SASA + B-factor


# ── AtomEncoder ───────────────────────────────────────────────────────────

class TestAtomEncoder:
    @pytest.fixture
    def lig_enc(self):
        return AtomEncoder(emb_dim=EMB_DIM, feature_dims=lig_feature_dims)

    @pytest.fixture
    def rec_enc(self):
        return AtomEncoder(emb_dim=EMB_DIM, feature_dims=rec_residue_feature_dims)

    @pytest.fixture
    def rec_atom_enc(self):
        return AtomEncoder(emb_dim=EMB_DIM, feature_dims=rec_atom_feature_dims)

    # --- output shape ----------------------------------------------------------

    def test_lig_output_shape(self, lig_enc):
        n_cat, n_scal = len(lig_feature_dims[0]), lig_feature_dims[1]
        x = torch.zeros(8, n_cat + n_scal)
        out = lig_enc(x)
        assert out.shape == (8, EMB_DIM)

    def test_rec_residue_output_shape(self, rec_enc):
        n_cat, n_scal = len(rec_residue_feature_dims[0]), rec_residue_feature_dims[1]
        x = torch.zeros(12, n_cat + n_scal)
        out = rec_enc(x)
        assert out.shape == (12, EMB_DIM)

    def test_rec_atom_output_shape(self, rec_atom_enc):
        n_cat, n_scal = len(rec_atom_feature_dims[0]), rec_atom_feature_dims[1]
        x = torch.zeros(5, n_cat + n_scal)
        out = rec_atom_enc(x)
        assert out.shape == (5, EMB_DIM)

    # --- no NaN / inf ----------------------------------------------------------

    def test_no_nan(self, lig_enc):
        n_cat, n_scal = len(lig_feature_dims[0]), lig_feature_dims[1]
        x = torch.zeros(4, n_cat + n_scal)
        out = lig_enc(x)
        assert not out.isnan().any()
        assert not out.isinf().any()

    # --- n_feats_to_use truncation --------------------------------------------

    def test_n_feats_to_use(self):
        enc = AtomEncoder(emb_dim=EMB_DIM, feature_dims=lig_feature_dims, n_feats_to_use=3)
        assert len(enc.atom_embedding_list) == 3
        n_cat, n_scal = len(lig_feature_dims[0]), lig_feature_dims[1]
        x = torch.zeros(4, n_cat + n_scal)
        out = enc(x)
        assert out.shape == (4, EMB_DIM)

    # --- scalar features disabled --------------------------------------------

    def test_no_scalar_feat(self):
        enc = AtomEncoder(emb_dim=EMB_DIM, feature_dims=lig_feature_dims, use_scalar_feat=False)
        n_cat, n_scal = len(lig_feature_dims[0]), lig_feature_dims[1]
        x = torch.zeros(4, n_cat + n_scal)
        out = enc(x)
        assert out.shape == (4, EMB_DIM)

    # --- gradients flow -------------------------------------------------------

    def test_gradients(self, lig_enc):
        n_cat, n_scal = len(lig_feature_dims[0]), lig_feature_dims[1]
        x = torch.zeros(4, n_cat + n_scal, requires_grad=False)
        out = lig_enc(x)
        loss = out.sum()
        loss.backward()
        grad_norms = [p.grad.norm().item() for p in lig_enc.parameters() if p.grad is not None]
        assert len(grad_norms) > 0
        assert all(g >= 0 for g in grad_norms)

    # --- single-node batch works ----------------------------------------------

    def test_single_node(self, lig_enc):
        n_cat, n_scal = len(lig_feature_dims[0]), lig_feature_dims[1]
        x = torch.zeros(1, n_cat + n_scal)
        out = lig_enc(x)
        assert out.shape == (1, EMB_DIM)
