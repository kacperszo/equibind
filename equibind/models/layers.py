# Ported from HannesStark/EquiBind (MIT License)
# https://github.com/HannesStark/EquiBind/blob/main/models/equibind.py
#
# Rewritten for PyTorch Geometric — no DGL dependency.
# Message passing is implemented with native PyTorch scatter operations
# (scatter_add_) so the only hard dependency is torch-geometric itself
# (used for GraphNorm's unbatch and Data types in the encoder).

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.utils import unbatch


# ---------------------------------------------------------------------------
# Scatter helper (pure PyTorch, no torch-scatter required)
# ---------------------------------------------------------------------------

def _scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Mean aggregation: out[i] = mean of src[j] where index[j] == i.

    src   : (E, D)
    index : (E,)   — destination node indices in [0, dim_size)
    """
    d = src.size(-1)
    out = src.new_zeros(dim_size, d)
    cnt = src.new_zeros(dim_size, 1)
    idx = index.unsqueeze(-1).expand(-1, d)
    out.scatter_add_(0, idx, src)
    cnt.scatter_add_(0, index.unsqueeze(-1), src.new_ones(src.size(0), 1))
    return out / cnt.clamp(min=1)


# ---------------------------------------------------------------------------
# Normalisation modules
# ---------------------------------------------------------------------------

class GraphNorm(nn.Module):
    """Per-graph zero-mean / unit-std normalisation using PyG batch indices."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + self.eps)

    def forward(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        normed = torch.cat([self._norm(piece) for piece in unbatch(h, batch)], dim=0)
        return self.gamma * normed + self.beta if self.affine else normed


class CoordsNorm(nn.Module):
    """Normalise coordinate vectors to unit length then rescale by a learned scalar."""

    def __init__(self, eps: float = 1e-8, scale_init: float = 1.0):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(1).fill_(scale_init))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        norms = coords.norm(dim=-1, keepdim=True)
        return coords / norms.clamp(min=self.eps) * self.scale


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def get_non_lin(kind: str, negative_slope: float) -> nn.Module:
    if kind == "swish":
        return nn.SiLU()
    elif kind == "relu":
        return nn.ReLU()
    else:
        assert kind == "lkyrelu"
        return nn.LeakyReLU(negative_slope=negative_slope)


def get_layer_norm(norm_type, dim: int) -> nn.Module:
    if norm_type == "BN":
        return nn.BatchNorm1d(dim)
    elif norm_type == "LN":
        return nn.LayerNorm(dim)
    return nn.Identity()


def get_norm(norm_type, dim: int) -> nn.Module:
    if norm_type == "BN":
        return nn.BatchNorm1d(dim)
    elif norm_type == "LN":
        return nn.LayerNorm(dim)
    elif norm_type == "GN":
        return GraphNorm(dim)
    assert norm_type in ("0", 0)
    return nn.Identity()


def apply_norm(
    h: torch.Tensor,
    norm_type,
    norm_layer: nn.Module,
    batch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if norm_type == "GN":
        assert batch is not None, "GraphNorm requires the batch index tensor"
        return norm_layer(h, batch)
    return norm_layer(h)


# ---------------------------------------------------------------------------
# Attention and mask utilities
# ---------------------------------------------------------------------------

def cross_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: Optional[torch.Tensor],
    cross_msgs: bool,
) -> torch.Tensor:
    """Soft cross-attention.  queries (N,D), keys (M,D), values (M,d), mask (N,M)."""
    if not cross_msgs:
        return queries * 0.0
    assert mask is not None
    a = mask * torch.mm(queries, keys.t()) - 1000.0 * (1.0 - mask)
    return torch.mm(torch.softmax(a, dim=1), values)


def get_mask(
    batch_lig: torch.Tensor,
    batch_rec: torch.Tensor,
    device,
) -> torch.Tensor:
    """Block-diagonal mask: mask[i,j]=1 iff ligand node i and receptor node j share a complex."""
    return (batch_lig.unsqueeze(1) == batch_rec.unsqueeze(0)).float().to(device)


# ---------------------------------------------------------------------------
# IEGMN_Layer
# ---------------------------------------------------------------------------

class IEGMN_Layer(nn.Module):
    """One IEGMN message-passing layer operating on two molecular graphs simultaneously.

    All inputs/outputs are plain tensors extracted from torch_geometric Data objects.
    The ligand and receptor graphs are processed in parallel; cross-attention
    couples their node representations at every layer.
    """

    def __init__(
        self,
        orig_h_feats_dim: int,
        h_feats_dim: int,
        out_feats_dim: int,
        lig_input_edge_feats_dim: int,
        rec_input_edge_feats_dim: int,
        nonlin: str,
        cross_msgs: bool,
        layer_norm,
        layer_norm_coords,
        final_h_layer_norm,
        use_dist_in_layers: bool,
        skip_weight_h: float,
        x_connection_init: float,
        leakyrelu_neg_slope: float,
        debug: bool,
        device,
        dropout: float,
        save_trajectories: bool = False,
        rec_square_distance_scale: float = 1.0,
        standard_norm_order: bool = False,
        normalize_coordinate_update: bool = False,
        lig_evolve: bool = True,
        rec_evolve: bool = True,
        fine_tune: bool = False,
        geometry_regularization: bool = False,
        pre_crossmsg_norm_type=0,
        post_crossmsg_norm_type=0,
        norm_cross_coords_update: bool = False,
        loss_geometry_regularization: bool = False,
        geom_reg_steps: int = 1,
        geometry_reg_step_size: float = 0.1,
    ):
        super().__init__()
        self.cross_msgs = cross_msgs
        self.debug = debug
        self.device = device
        self.lig_evolve = lig_evolve
        self.rec_evolve = rec_evolve
        self.h_feats_dim = h_feats_dim
        self.out_feats_dim = out_feats_dim
        self.fine_tune = fine_tune
        self.normalize_coordinate_update = normalize_coordinate_update
        self.final_h_layer_norm = final_h_layer_norm
        self.use_dist_in_layers = use_dist_in_layers
        self.skip_weight_h = skip_weight_h
        self.x_connection_init = x_connection_init
        self.rec_square_distance_scale = rec_square_distance_scale
        self.geometry_reg_step_size = geometry_reg_step_size
        self.norm_cross_coords_update = norm_cross_coords_update
        self.loss_geometry_regularization = loss_geometry_regularization
        self.geometry_regularization = geometry_regularization
        self.geom_reg_steps = geom_reg_steps
        self.save_trajectories = save_trajectories
        self.standard_norm_order = standard_norm_order
        self.pre_crossmsg_norm_type = pre_crossmsg_norm_type
        self.post_crossmsg_norm_type = post_crossmsg_norm_type
        self.all_sigmas_dist = [1.5 ** x for x in range(15)]

        # ---- Edge MLPs ----
        lig_edge_in = h_feats_dim * 2 + lig_input_edge_feats_dim
        if self.use_dist_in_layers and self.lig_evolve:
            lig_edge_in += len(self.all_sigmas_dist)
        rec_edge_in = h_feats_dim * 2 + rec_input_edge_feats_dim
        if self.use_dist_in_layers and self.rec_evolve:
            rec_edge_in += len(self.all_sigmas_dist)

        def _edge_mlp(in_dim: int) -> nn.Sequential:
            if standard_norm_order:
                return nn.Sequential(
                    nn.Linear(in_dim, out_feats_dim),
                    get_layer_norm(layer_norm, out_feats_dim),
                    get_non_lin(nonlin, leakyrelu_neg_slope),
                    nn.Dropout(dropout),
                    nn.Linear(out_feats_dim, out_feats_dim),
                    get_layer_norm(layer_norm, out_feats_dim),
                )
            return nn.Sequential(
                nn.Linear(in_dim, out_feats_dim),
                nn.Dropout(dropout),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                get_layer_norm(layer_norm, out_feats_dim),
                nn.Linear(out_feats_dim, out_feats_dim),
            )

        self.lig_edge_mlp = _edge_mlp(lig_edge_in)
        self.rec_edge_mlp = _edge_mlp(rec_edge_in)

        # ---- Node MLPs ----
        node_in = orig_h_feats_dim + 2 * h_feats_dim + out_feats_dim

        def _node_mlp() -> nn.Sequential:
            if standard_norm_order:
                return nn.Sequential(
                    nn.Linear(node_in, h_feats_dim),
                    get_layer_norm(layer_norm, h_feats_dim),
                    get_non_lin(nonlin, leakyrelu_neg_slope),
                    nn.Dropout(dropout),
                    nn.Linear(h_feats_dim, out_feats_dim),
                    get_layer_norm(layer_norm, out_feats_dim),
                )
            return nn.Sequential(
                nn.Linear(node_in, h_feats_dim),
                nn.Dropout(dropout),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                get_layer_norm(layer_norm, h_feats_dim),
                nn.Linear(h_feats_dim, out_feats_dim),
            )

        self.node_mlp_lig = _node_mlp()
        self.node_mlp = _node_mlp()
        self.node_norm = nn.Identity()

        # ---- Coordinate MLPs ----
        def _coords_mlp() -> nn.Sequential:
            if standard_norm_order:
                return nn.Sequential(
                    nn.Linear(out_feats_dim, out_feats_dim),
                    get_layer_norm(layer_norm_coords, out_feats_dim),
                    get_non_lin(nonlin, leakyrelu_neg_slope),
                    nn.Dropout(dropout),
                    nn.Linear(out_feats_dim, 1),
                )
            return nn.Sequential(
                nn.Linear(out_feats_dim, out_feats_dim),
                nn.Dropout(dropout),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                get_layer_norm(layer_norm_coords, out_feats_dim),
                nn.Linear(out_feats_dim, 1),
            )

        self.coords_mlp_lig = _coords_mlp()
        self.coords_mlp_rec = _coords_mlp()

        if self.normalize_coordinate_update:
            self.lig_coords_norm = CoordsNorm(scale_init=1e-2)
            self.rec_coords_norm = CoordsNorm(scale_init=1e-2)

        # ---- Fine-tune: cross-coordinate attention ----
        if self.fine_tune:
            _cross_norm = CoordsNorm(scale_init=1e-2) if norm_cross_coords_update else nn.Identity()
            self.lig_cross_coords_norm = _cross_norm
            self.rec_cross_coords_norm = _cross_norm

            def _coors_qk() -> nn.Sequential:
                return nn.Sequential(
                    nn.Linear(h_feats_dim, h_feats_dim, bias=False),
                    get_non_lin(nonlin, leakyrelu_neg_slope),
                )

            def _coors_v() -> nn.Sequential:
                return nn.Sequential(
                    nn.Linear(h_feats_dim, h_feats_dim),
                    get_non_lin(nonlin, leakyrelu_neg_slope),
                    nn.Linear(h_feats_dim, 1),
                )

            self.att_mlp_cross_coors_Q = _coors_qk()
            self.att_mlp_cross_coors_K = _coors_qk()
            self.att_mlp_cross_coors_V = _coors_v()
            self.att_mlp_cross_coors_Q_lig = _coors_qk()
            self.att_mlp_cross_coors_K_lig = _coors_qk()
            self.att_mlp_cross_coors_V_lig = _coors_v()

        # ---- Cross-attention Q/K/V ----
        def _qk() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(h_feats_dim, h_feats_dim, bias=False),
                get_non_lin(nonlin, leakyrelu_neg_slope),
            )

        self.att_mlp_Q_lig = _qk()
        self.att_mlp_K_lig = _qk()
        self.att_mlp_V_lig = nn.Sequential(nn.Linear(h_feats_dim, h_feats_dim, bias=False))
        self.att_mlp_Q = _qk()
        self.att_mlp_K = _qk()
        self.att_mlp_V = nn.Sequential(nn.Linear(h_feats_dim, h_feats_dim, bias=False))

        # ---- Post-update norms ----
        self.final_h_layernorm_layer_lig = get_norm(final_h_layer_norm, out_feats_dim)
        self.final_h_layernorm_layer = get_norm(final_h_layer_norm, out_feats_dim)
        self.pre_crossmsg_norm_lig = get_norm(pre_crossmsg_norm_type, h_feats_dim)
        self.pre_crossmsg_norm_rec = get_norm(pre_crossmsg_norm_type, h_feats_dim)
        self.post_crossmsg_norm_lig = get_norm(post_crossmsg_norm_type, h_feats_dim)
        self.post_crossmsg_norm_rec = get_norm(post_crossmsg_norm_type, h_feats_dim)

    def reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p, gain=1.0)
            else:
                nn.init.zeros_(p)

    def _edge_msgs(
        self,
        h: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_mlp: nn.Module,
        evolve: bool,
        sq_dist_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute edge messages and relative coordinates (x_src - x_dst)."""
        src, dst = edge_index
        x_rel = coords[src] - coords[dst]  # (E, 3)

        if self.use_dist_in_layers and evolve:
            sq = (x_rel ** 2).sum(-1, keepdim=True)
            rbf = torch.cat(
                [torch.exp(-(sq / sq_dist_scale) / s) for s in self.all_sigmas_dist],
                dim=-1,
            )
            edge_in = torch.cat([h[src], h[dst], edge_attr, rbf], dim=-1)
        else:
            edge_in = torch.cat([h[src], h[dst], edge_attr], dim=-1)

        return edge_mlp(edge_in), x_rel  # (E, out_dim), (E, 3)

    def forward(
        self,
        h_lig: torch.Tensor,            # (N_lig, h_dim)
        coords_lig: torch.Tensor,        # (N_lig, 3)
        edge_index_lig: torch.Tensor,    # (2, E_lig)
        edge_attr_lig: torch.Tensor,     # (E_lig, lig_edge_dim)
        batch_lig: torch.Tensor,         # (N_lig,)
        h_rec: torch.Tensor,             # (N_rec, h_dim)
        coords_rec: torch.Tensor,        # (N_rec, 3)
        edge_index_rec: torch.Tensor,    # (2, E_rec)
        edge_attr_rec: torch.Tensor,     # (E_rec, rec_edge_dim)
        batch_rec: torch.Tensor,         # (N_rec,)
        orig_h_lig: torch.Tensor,        # (N_lig, orig_h_dim)
        orig_coords_lig: torch.Tensor,   # (N_lig, 3)
        orig_h_rec: torch.Tensor,        # (N_rec, orig_h_dim)
        orig_coords_rec: torch.Tensor,   # (N_rec, 3)
        mask: Optional[torch.Tensor],    # (N_lig, N_rec) or None
        geom_edge_index: Optional[torch.Tensor] = None,  # (2, E_geom)
        geom_edge_attr: Optional[torch.Tensor] = None,   # (E_geom,) distances
    ) -> tuple:
        n_lig = h_lig.size(0)
        n_rec = h_rec.size(0)

        # ---- Edge messages ----
        msgs_lig, x_rel_lig = self._edge_msgs(
            h_lig, coords_lig, edge_index_lig, edge_attr_lig,
            self.lig_edge_mlp, self.lig_evolve,
        )
        msgs_rec, x_rel_rec = self._edge_msgs(
            h_rec, coords_rec, edge_index_rec, edge_attr_rec,
            self.rec_edge_mlp, self.rec_evolve, self.rec_square_distance_scale,
        )

        # ---- Message aggregation ----
        aggr_msg_lig = _scatter_mean(msgs_lig, edge_index_lig[1], n_lig)
        aggr_msg_rec = _scatter_mean(msgs_rec, edge_index_rec[1], n_rec)

        # ---- Cross-attention ----
        h_lig_norm = apply_norm(h_lig, self.final_h_layer_norm, self.final_h_layernorm_layer, batch_lig)
        h_rec_norm = apply_norm(h_rec, self.final_h_layer_norm, self.final_h_layernorm_layer, batch_rec)

        cross_lig = cross_attention(
            self.att_mlp_Q_lig(h_lig_norm), self.att_mlp_K(h_rec_norm),
            self.att_mlp_V(h_rec_norm), mask, self.cross_msgs,
        )
        cross_rec = cross_attention(
            self.att_mlp_Q(h_rec_norm), self.att_mlp_K_lig(h_lig_norm),
            self.att_mlp_V_lig(h_lig_norm),
            mask.t() if mask is not None else None, self.cross_msgs,
        )
        cross_lig = apply_norm(cross_lig, self.final_h_layer_norm, self.final_h_layernorm_layer, batch_lig)
        cross_rec = apply_norm(cross_rec, self.final_h_layer_norm, self.final_h_layernorm_layer, batch_rec)

        # ---- Coordinate updates ----
        if self.lig_evolve:
            coef_lig = self.coords_mlp_lig(msgs_lig)  # (E_lig, 1)
            xr_lig = self.lig_coords_norm(x_rel_lig) if self.normalize_coordinate_update else x_rel_lig
            x_upd_lig = _scatter_mean(xr_lig * coef_lig, edge_index_lig[1], n_lig)
            coords_lig_new = (
                self.x_connection_init * orig_coords_lig
                + (1.0 - self.x_connection_init) * coords_lig
                + x_upd_lig
            )
        else:
            coords_lig_new = coords_lig

        if self.rec_evolve:
            coef_rec = self.coords_mlp_rec(msgs_rec)  # (E_rec, 1)
            xr_rec = self.rec_coords_norm(x_rel_rec) if self.normalize_coordinate_update else x_rel_rec
            x_upd_rec = _scatter_mean(xr_rec * coef_rec, edge_index_rec[1], n_rec)
            coords_rec_new = (
                self.x_connection_init * orig_coords_rec
                + (1.0 - self.x_connection_init) * coords_rec
                + x_upd_rec
            )
        else:
            coords_rec_new = coords_rec

        # ---- Fine-tune: cross-coordinate attention ----
        if self.fine_tune:
            coords_lig_new = coords_lig_new + self.att_mlp_cross_coors_V_lig(h_lig) * (
                self.lig_cross_coords_norm(
                    coords_lig - cross_attention(
                        self.att_mlp_cross_coors_Q_lig(h_lig),
                        self.att_mlp_cross_coors_K(h_rec),
                        coords_rec, mask, self.cross_msgs,
                    )
                )
            )
            coords_rec_new = coords_rec_new + self.att_mlp_cross_coors_V(h_rec) * (
                self.rec_cross_coords_norm(
                    coords_rec - cross_attention(
                        self.att_mlp_cross_coors_Q(h_rec),
                        self.att_mlp_cross_coors_K_lig(h_lig),
                        coords_lig, mask.t() if mask is not None else None, self.cross_msgs,
                    )
                )
            )

        # ---- Geometry regularisation ----
        trajectory: list = []
        if self.save_trajectories:
            trajectory.append(coords_lig_new.detach().cpu())

        geom_loss: float | torch.Tensor = 0
        if geom_edge_index is not None and geom_edge_attr is not None:
            src_g, dst_g = geom_edge_index
            target_dist = geom_edge_attr.squeeze(-1)  # (E_geom,)

            if self.loss_geometry_regularization:
                d_sq = ((coords_lig_new[src_g] - coords_lig_new[dst_g]) ** 2).sum(1)
                geom_loss = ((d_sq - target_dist ** 2) ** 2).sum()

            if self.geometry_regularization:
                for _ in range(self.geom_reg_steps):
                    d_sq = ((coords_lig_new[src_g] - coords_lig_new[dst_g]) ** 2).sum(1)
                    partial = (
                        2 * (d_sq - target_dist ** 2).unsqueeze(-1)
                        * 2 * (coords_lig_new[src_g] - coords_lig_new[dst_g])
                    )
                    grad_x = coords_lig_new.new_zeros(n_lig, 3)
                    grad_x.scatter_add_(0, dst_g.unsqueeze(-1).expand(-1, 3), partial)
                    coords_lig_new = coords_lig_new + self.geometry_reg_step_size * grad_x
                    if self.save_trajectories:
                        trajectory.append(coords_lig_new.detach().cpu())

        # ---- Node feature updates ----
        inp_lig = torch.cat([self.node_norm(h_lig), aggr_msg_lig, cross_lig, orig_h_lig], dim=-1)
        inp_rec = torch.cat([self.node_norm(h_rec), aggr_msg_rec, cross_rec, orig_h_rec], dim=-1)

        if self.h_feats_dim == self.out_feats_dim:
            h_lig_new = self.skip_weight_h * self.node_mlp_lig(inp_lig) + (1.0 - self.skip_weight_h) * h_lig
            h_rec_new = self.skip_weight_h * self.node_mlp(inp_rec) + (1.0 - self.skip_weight_h) * h_rec
        else:
            h_lig_new = self.node_mlp_lig(inp_lig)
            h_rec_new = self.node_mlp(inp_rec)

        h_lig_new = apply_norm(h_lig_new, self.final_h_layer_norm, self.final_h_layernorm_layer_lig, batch_lig)
        h_rec_new = apply_norm(h_rec_new, self.final_h_layer_norm, self.final_h_layernorm_layer, batch_rec)

        return coords_lig_new, h_lig_new, coords_rec_new, h_rec_new, trajectory, geom_loss

    def __repr__(self) -> str:
        return f"IEGMN_Layer(h={self.h_feats_dim}→{self.out_feats_dim})"
