"""Tests for EquiBindEncoder end-to-end."""

import pytest
import torch
from torch_geometric.data import Batch

from equibind import EquiBindEncoder
from equibind.features.process_mols import rec_atom_feature_dims

# Default output embedding dimension
OUT_DIM = 64


def _encoder(**kw) -> EquiBindEncoder:
    """Construct a fast (2-layer, Identity norms) encoder for tests."""
    defaults = dict(
        n_lays=2,
        device="cpu",
        layer_norm=0,
        layer_norm_coords=0,
        final_h_layer_norm=0,
        dropout=0.0,
    )
    defaults.update(kw)
    return EquiBindEncoder(**defaults)


# ── single-complex forward ────────────────────────────────────────────────

class TestEncoderSingle:
    def test_output_shapes(self, lig_data, rec_data):
        enc = _encoder()
        h_lig, h_rec, c_lig, c_rec, _ = enc(lig_data, rec_data)
        assert h_lig.shape == (lig_data.x.size(0), OUT_DIM)
        assert h_rec.shape == (rec_data.x.size(0), OUT_DIM)
        assert c_lig.shape == (lig_data.x.size(0), 3)
        assert c_rec.shape == (rec_data.x.size(0), 3)

    def test_no_nan(self, lig_data, rec_data):
        enc = _encoder()
        for t in enc(lig_data, rec_data)[:4]:
            assert not t.isnan().any(), "NaN in encoder output"
            assert not t.isinf().any(), "Inf in encoder output"

    def test_no_mean_node_features(self, lig_data, rec_data):
        enc = _encoder(use_mean_node_features=False)
        h_lig, h_rec, *_ = enc(lig_data, rec_data)
        assert h_lig.shape == (lig_data.x.size(0), OUT_DIM)
        assert h_rec.shape == (rec_data.x.size(0), OUT_DIM)

    def test_no_cross_msgs(self, lig_data, rec_data):
        enc = _encoder(cross_msgs=False)
        h_lig, h_rec, *_ = enc(lig_data, rec_data)
        assert h_lig.shape == (lig_data.x.size(0), OUT_DIM)
        assert not h_lig.isnan().any()

    def test_no_edge_features(self, lig_data, rec_data):
        enc = _encoder(use_edge_features_in_gmn=False)
        h_lig, *_ = enc(lig_data, rec_data)
        assert not h_lig.isnan().any()

    def test_custom_hidden_dim(self, lig_data, rec_data):
        enc = _encoder(residue_emb_dim=32, iegmn_lay_hid_dim=32)
        h_lig, h_rec, *_ = enc(lig_data, rec_data)
        assert h_lig.shape[-1] == 32
        assert h_rec.shape[-1] == 32


# ── batched forward ───────────────────────────────────────────────────────

class TestEncoderBatched:
    def test_output_shapes(self, lig_batch, rec_batch):
        enc = _encoder()
        h_lig, h_rec, c_lig, c_rec, _ = enc(lig_batch, rec_batch)
        assert h_lig.shape == (lig_batch.x.size(0), OUT_DIM)
        assert h_rec.shape == (rec_batch.x.size(0), OUT_DIM)
        assert c_lig.shape == (lig_batch.x.size(0), 3)
        assert c_rec.shape == (rec_batch.x.size(0), 3)

    def test_no_nan_batched(self, lig_batch, rec_batch):
        enc = _encoder()
        for t in enc(lig_batch, rec_batch)[:4]:
            assert not t.isnan().any()

    def test_single_vs_batched_consistency(self, lig_data, rec_data):
        """Single graph and a batch of one must give identical embeddings."""
        enc = _encoder(noise_initial=0.0)
        enc.eval()
        with torch.no_grad():
            h_single, _, _, _, _ = enc(lig_data, rec_data)
            batch = Batch.from_data_list([lig_data])
            rec_b = Batch.from_data_list([rec_data])
            h_batch, _, _, _, _ = enc(batch, rec_b)
        assert torch.allclose(h_single, h_batch, atol=1e-5)


# ── atom-level receptor ───────────────────────────────────────────────────

class TestEncoderRecAtoms:
    def test_atom_level_receptor(self, lig_data):
        from tests.conftest import _make_rec
        rec = _make_rec(atom_level=True)
        enc = _encoder(
            use_rec_atoms=True,
            rec_input_edge_feats_dim=27,
        )
        h_lig, h_rec, *_ = enc(lig_data, rec)
        assert h_lig.shape[-1] == OUT_DIM
        assert h_rec.shape[-1] == OUT_DIM
        assert not h_lig.isnan().any()


# ── gradient flow through encoder ────────────────────────────────────────

class TestEncoderGradients:
    def test_gradients_flow(self, lig_data, rec_data):
        enc = _encoder()
        lig_data.pos = lig_data.pos.requires_grad_(True)
        h_lig, _, _, _, _ = enc(lig_data, rec_data)
        h_lig.sum().backward()
        assert lig_data.pos.grad is not None

    def test_parameter_gradients(self, lig_data, rec_data):
        enc = _encoder()
        h_lig, h_rec, *_ = enc(lig_data, rec_data)
        (h_lig.sum() + h_rec.sum()).backward()
        params_with_grad = [p for p in enc.parameters() if p.grad is not None]
        assert len(params_with_grad) > 0


# ── repr ──────────────────────────────────────────────────────────────────

def test_repr():
    enc = _encoder()
    r = repr(enc)
    assert "EquiBindEncoder" in r
    assert "n_lays=2" in r


# ── geometry regularisation (loss mode) ───────────────────────────────────

class TestGeomRegularisation:
    def test_geom_loss_mode(self, lig_data, rec_data):
        from torch_geometric.data import Data as PyGData

        enc = _encoder(loss_geometry_regularization=True)
        # Build a tiny geometry graph (bond distances between first 4 atoms)
        n = lig_data.x.size(0)
        src = torch.arange(0, min(n, 4))
        dst = (src + 1) % min(n, 4)
        geom = PyGData(
            edge_index=torch.stack([src, dst]),
            edge_attr=torch.ones(src.size(0), 1),
        )
        h_lig, h_rec, c_lig, c_rec, geom_loss = enc(lig_data, rec_data, geom_data=geom)
        assert h_lig.shape[-1] == OUT_DIM
        # geom_loss should be a non-negative scalar tensor
        assert isinstance(geom_loss, torch.Tensor)
        assert geom_loss.item() >= 0.0
