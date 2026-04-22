"""Tests for layer primitives: GraphNorm, CoordsNorm, IEGMN_Layer, utilities."""

import pytest
import torch

from equibind.models.layers import (
    GraphNorm,
    CoordsNorm,
    IEGMN_Layer,
    _scatter_mean,
    cross_attention,
    get_mask,
    apply_norm,
    get_norm,
)

# ── shared constants ───────────────────────────────────────────────────────

H_DIM = 32
OUT_DIM = 32
LIG_EDGE_DIM = 15
REC_EDGE_DIM = 27


def _layer(h=H_DIM, out=OUT_DIM, cross=True, **kw) -> IEGMN_Layer:
    """Construct a minimal IEGMN_Layer for testing (Identity norms, no dropout)."""
    defaults = dict(
        orig_h_feats_dim=h,
        h_feats_dim=h,
        out_feats_dim=out,
        lig_input_edge_feats_dim=LIG_EDGE_DIM,
        rec_input_edge_feats_dim=REC_EDGE_DIM,
        nonlin="lkyrelu",
        cross_msgs=cross,
        layer_norm=0,
        layer_norm_coords=0,
        final_h_layer_norm=0,
        use_dist_in_layers=True,
        skip_weight_h=0.5,
        x_connection_init=0.25,
        leakyrelu_neg_slope=0.1,
        debug=False,
        device="cpu",
        dropout=0.0,
    )
    defaults.update(kw)
    return IEGMN_Layer(**defaults)


def _tensors(n_lig=8, n_rec=15, e_lig=20, e_rec=50, h=H_DIM, seed=42):
    """Synthetic tensors for a single complex."""
    g = torch.Generator().manual_seed(seed)
    h_lig = torch.randn(n_lig, h, generator=g)
    coords_lig = torch.randn(n_lig, 3, generator=g)
    edge_index_lig = torch.randint(0, n_lig, (2, e_lig), generator=g)
    edge_attr_lig = torch.randn(e_lig, LIG_EDGE_DIM, generator=g)
    batch_lig = torch.zeros(n_lig, dtype=torch.long)

    h_rec = torch.randn(n_rec, h, generator=g)
    coords_rec = torch.randn(n_rec, 3, generator=g)
    edge_index_rec = torch.randint(0, n_rec, (2, e_rec), generator=g)
    edge_attr_rec = torch.randn(e_rec, REC_EDGE_DIM, generator=g)
    batch_rec = torch.zeros(n_rec, dtype=torch.long)

    mask = get_mask(batch_lig, batch_rec, "cpu")

    return (
        h_lig, coords_lig, edge_index_lig, edge_attr_lig, batch_lig,
        h_rec, coords_rec, edge_index_rec, edge_attr_rec, batch_rec,
        h_lig.clone(), coords_lig.clone(), h_rec.clone(), coords_rec.clone(),
        mask,
    )


# ── _scatter_mean ─────────────────────────────────────────────────────────

class TestScatterMean:
    def test_basic(self):
        src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        idx = torch.tensor([0, 0, 1])
        out = _scatter_mean(src, idx, dim_size=2)
        assert out.shape == (2, 2)
        assert torch.allclose(out[0], torch.tensor([2.0, 3.0]))   # mean of rows 0,1
        assert torch.allclose(out[1], torch.tensor([5.0, 6.0]))   # only row 2

    def test_empty_src(self):
        src = torch.zeros(0, 4)
        idx = torch.zeros(0, dtype=torch.long)
        out = _scatter_mean(src, idx, dim_size=3)
        assert out.shape == (3, 4)
        assert out.sum() == 0.0

    def test_all_same_index(self):
        src = torch.ones(5, 3)
        idx = torch.zeros(5, dtype=torch.long)
        out = _scatter_mean(src, idx, dim_size=2)
        assert torch.allclose(out[0], torch.ones(3))
        assert torch.allclose(out[1], torch.zeros(3))


# ── GraphNorm ──────────────────────────────────────────────────────────────

class TestGraphNorm:
    def test_output_shape(self):
        gn = GraphNorm(num_features=16)
        h = torch.randn(10, 16)
        batch = torch.zeros(10, dtype=torch.long)
        out = gn(h, batch)
        assert out.shape == (10, 16)

    def test_two_graphs(self):
        gn = GraphNorm(num_features=8)
        h = torch.randn(12, 8)
        batch = torch.cat([torch.zeros(5, dtype=torch.long), torch.ones(7, dtype=torch.long)])
        out = gn(h, batch)
        assert out.shape == (12, 8)
        assert not out.isnan().any()

    def test_no_affine(self):
        gn = GraphNorm(num_features=4, affine=False)
        h = torch.randn(6, 4)
        batch = torch.zeros(6, dtype=torch.long)
        out = gn(h, batch)
        assert out.shape == (6, 4)


# ── CoordsNorm ────────────────────────────────────────────────────────────

class TestCoordsNorm:
    def test_output_shape(self):
        cn = CoordsNorm()
        coords = torch.randn(8, 3)
        out = cn(coords)
        assert out.shape == (8, 3)

    def test_learnable_scale(self):
        cn = CoordsNorm(scale_init=2.0)
        assert cn.scale.item() == pytest.approx(2.0)

    def test_zero_vector_no_nan(self):
        cn = CoordsNorm()
        coords = torch.zeros(4, 3)
        out = cn(coords)
        assert not out.isnan().any()


# ── get_mask ──────────────────────────────────────────────────────────────

class TestGetMask:
    def test_single_complex(self):
        batch_lig = torch.zeros(5, dtype=torch.long)
        batch_rec = torch.zeros(8, dtype=torch.long)
        mask = get_mask(batch_lig, batch_rec, "cpu")
        assert mask.shape == (5, 8)
        assert mask.all()

    def test_two_complexes_block_diagonal(self):
        batch_lig = torch.tensor([0, 0, 1, 1, 1])
        batch_rec = torch.tensor([0, 0, 0, 1, 1])
        mask = get_mask(batch_lig, batch_rec, "cpu")
        assert mask.shape == (5, 5)
        # rows 0-1 (complex 0) should attend only to cols 0-2 (complex 0)
        assert mask[:2, :3].all()
        assert (mask[:2, 3:] == 0).all()
        # rows 2-4 (complex 1) should attend only to cols 3-4 (complex 1)
        assert mask[2:, 3:].all()
        assert (mask[2:, :3] == 0).all()


# ── cross_attention ───────────────────────────────────────────────────────

class TestCrossAttention:
    def test_output_shape(self):
        Q = torch.randn(5, 8)
        K = torch.randn(10, 8)
        V = torch.randn(10, 4)
        mask = torch.ones(5, 10)
        out = cross_attention(Q, K, V, mask, cross_msgs=True)
        assert out.shape == (5, 4)

    def test_disabled_returns_zeros(self):
        Q = torch.randn(5, 8)
        K = torch.randn(10, 8)
        V = torch.randn(10, 4)
        out = cross_attention(Q, K, V, None, cross_msgs=False)
        assert out.shape == (5, 8)
        assert (out == 0).all()


# ── IEGMN_Layer ───────────────────────────────────────────────────────────

class TestIEGMNLayer:
    def test_output_shapes(self):
        layer = _layer()
        args = _tensors()
        c_lig, h_lig, c_rec, h_rec, traj, gloss = layer(*args)
        assert c_lig.shape == (8, 3)
        assert h_lig.shape == (8, OUT_DIM)
        assert c_rec.shape == (15, 3)
        assert h_rec.shape == (15, OUT_DIM)

    def test_no_nan(self):
        layer = _layer()
        c_lig, h_lig, c_rec, h_rec, _, _ = layer(*_tensors())
        for t in (c_lig, h_lig, c_rec, h_rec):
            assert not t.isnan().any(), "NaN in layer output"
            assert not t.isinf().any(), "Inf in layer output"

    def test_cross_msgs_disabled(self):
        layer = _layer(cross=False)
        # When cross_msgs=False, mask is None — should still work
        args = list(_tensors())
        args[14] = None  # set mask to None
        c_lig, h_lig, c_rec, h_rec, _, _ = layer(*args)
        assert h_lig.shape == (8, OUT_DIM)
        assert h_rec.shape == (15, OUT_DIM)

    def test_no_evolve(self):
        layer = _layer(lig_evolve=False, rec_evolve=False)
        args = _tensors()
        c_lig_in = args[1]
        c_rec_in = args[6]
        c_lig, _, c_rec, _, _, _ = layer(*args)
        # Coordinates should be unchanged when evolve=False
        assert torch.allclose(c_lig, c_lig_in)
        assert torch.allclose(c_rec, c_rec_in)

    def test_gradients_flow(self):
        layer = _layer()
        args = list(_tensors())
        args[0] = args[0].requires_grad_(True)   # h_lig
        c_lig, h_lig, c_rec, h_rec, _, _ = layer(*args)
        h_lig.sum().backward()
        assert args[0].grad is not None

    def test_dim_change_first_layer(self):
        """First encoder layer: orig_h=input_dim, h=input_dim, out=hidden_dim."""
        layer = _layer(h=37, out=32)  # h != out  → no skip connection
        n_lig, n_rec = 6, 10
        h_lig = torch.randn(n_lig, 37)
        h_rec = torch.randn(n_rec, 37)
        coords_lig = torch.randn(n_lig, 3)
        coords_rec = torch.randn(n_rec, 3)
        ei_lig = torch.randint(0, n_lig, (2, 15))
        ea_lig = torch.randn(15, LIG_EDGE_DIM)
        ei_rec = torch.randint(0, n_rec, (2, 30))
        ea_rec = torch.randn(30, REC_EDGE_DIM)
        bl = torch.zeros(n_lig, dtype=torch.long)
        br = torch.zeros(n_rec, dtype=torch.long)
        mask = get_mask(bl, br, "cpu")
        _, h_lig_out, _, h_rec_out, _, _ = layer(
            h_lig, coords_lig, ei_lig, ea_lig, bl,
            h_rec, coords_rec, ei_rec, ea_rec, br,
            h_lig.clone(), coords_lig.clone(), h_rec.clone(), coords_rec.clone(),
            mask,
        )
        assert h_lig_out.shape == (n_lig, 32)
        assert h_rec_out.shape == (n_rec, 32)

    def test_batched_two_complexes(self):
        n_lig = 8 + 12   # two complexes, 8 and 12 ligand atoms
        n_rec = 15 + 20
        batch_lig = torch.cat([torch.zeros(8, dtype=torch.long), torch.ones(12, dtype=torch.long)])
        batch_rec = torch.cat([torch.zeros(15, dtype=torch.long), torch.ones(20, dtype=torch.long)])
        g = torch.Generator().manual_seed(7)
        h_lig = torch.randn(n_lig, H_DIM, generator=g)
        h_rec = torch.randn(n_rec, H_DIM, generator=g)
        c_lig = torch.randn(n_lig, 3, generator=g)
        c_rec = torch.randn(n_rec, 3, generator=g)
        ei_lig = torch.randint(0, n_lig, (2, 40), generator=g)
        ea_lig = torch.randn(40, LIG_EDGE_DIM, generator=g)
        ei_rec = torch.randint(0, n_rec, (2, 80), generator=g)
        ea_rec = torch.randn(80, REC_EDGE_DIM, generator=g)
        mask = get_mask(batch_lig, batch_rec, "cpu")
        layer = _layer()
        c_lig_out, h_lig_out, c_rec_out, h_rec_out, _, _ = layer(
            h_lig, c_lig, ei_lig, ea_lig, batch_lig,
            h_rec, c_rec, ei_rec, ea_rec, batch_rec,
            h_lig.clone(), c_lig.clone(), h_rec.clone(), c_rec.clone(),
            mask,
        )
        assert h_lig_out.shape == (n_lig, OUT_DIM)
        assert h_rec_out.shape == (n_rec, OUT_DIM)
        assert not h_lig_out.isnan().any()
