# Ported and adapted from HannesStark/EquiBind (MIT License)
# https://github.com/HannesStark/EquiBind/blob/main/models/equibind.py
#
# Rewritten for PyTorch Geometric.
# Inputs are torch_geometric.data.Data / Batch objects.
# The docking head (keypoint attention + SVD/Kabsch) is removed so any
# downstream head can be attached to the returned node embeddings.

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data

from equibind.features.process_mols import (
    AtomEncoder,
    lig_feature_dims,
    rec_atom_feature_dims,
    rec_residue_feature_dims,
)
from equibind.models.layers import IEGMN_Layer, get_mask


def _batch_of(data: Data, device) -> torch.Tensor:
    """Return .batch tensor, defaulting to all-zeros for a single (unbatched) graph."""
    if hasattr(data, "batch") and data.batch is not None:
        return data.batch.to(device)
    return torch.zeros(data.x.size(0), dtype=torch.long, device=device)


class EquiBindEncoder(nn.Module):
    """IEGMN encoder: (ligand graph, receptor graph) → node embeddings + 3-D coords.

    This is the EquiBind backbone with the docking-specific head removed.
    Attach any downstream task head to the returned tensors.

    Expected Data attributes
    ------------------------
    lig_data:
        x         (N, 17)   raw atom features  [16 categorical cols + 1 scalar]
        pos       (N, 3)    initial 3-D coordinates
        edge_index (2, E)
        edge_attr  (E, lig_input_edge_feats_dim)   default: 15-dim Gaussian RBF
        mu_r_norm  (N, 5)   direction mean-norm features  [if use_mean_node_features]
        batch      (N,)     graph index per node           [populated by Batch.from_data_list]

    rec_data:
        x         (N, 3)   raw residue features  [1 categorical + 2 scalars]
        pos       (N, 3)   initial 3-D coordinates
        edge_index (2, E)
        edge_attr  (E, rec_input_edge_feats_dim)   default: 27-dim (15 RBF + 12 orientation)
        mu_r_norm  (N, 5)   [if use_mean_node_features]
        batch      (N,)

    geom_data (optional, ligand only):
        edge_index (2, E_g)
        edge_attr  (E_g,) or (E_g, 1)   bond/2-hop distances for regularisation
    """

    def __init__(
        self,
        n_lays: int = 5,
        debug: bool = False,
        device="cpu",
        use_rec_atoms: bool = False,
        shared_layers: bool = False,
        noise_decay_rate: float = 1.0,
        cross_msgs: bool = True,
        noise_initial: float = 0.0,
        use_edge_features_in_gmn: bool = True,
        use_mean_node_features: bool = True,
        residue_emb_dim: int = 64,
        iegmn_lay_hid_dim: int = 64,
        dropout: float = 0.0,
        nonlin: str = "lkyrelu",
        leakyrelu_neg_slope: float = 0.1,
        random_vec_dim: int = 0,
        random_vec_std: float = 1.0,
        use_scalar_features: bool = True,
        num_lig_feats: Optional[int] = None,
        separate_lig: bool = False,
        save_trajectories: bool = False,
        lig_input_edge_feats_dim: int = 15,
        rec_input_edge_feats_dim: int = 27,
        # IEGMN_Layer params with sensible defaults
        use_dist_in_layers: bool = True,
        skip_weight_h: float = 0.5,
        x_connection_init: float = 0.25,
        **kwargs,
    ):
        super().__init__()
        self.debug = debug
        self.cross_msgs = cross_msgs
        self.device = device
        self.save_trajectories = save_trajectories
        self.separate_lig = separate_lig
        self.use_rec_atoms = use_rec_atoms
        self.noise_decay_rate = noise_decay_rate
        self.noise_initial = noise_initial
        self.use_edge_features_in_gmn = use_edge_features_in_gmn
        self.use_mean_node_features = use_mean_node_features
        self.random_vec_dim = random_vec_dim
        self.random_vec_std = random_vec_std

        # ---- Atom / residue embedders ----
        emb_dim = residue_emb_dim - random_vec_dim
        self.lig_atom_embedder = AtomEncoder(
            emb_dim=emb_dim,
            feature_dims=lig_feature_dims,
            use_scalar_feat=use_scalar_features,
            n_feats_to_use=num_lig_feats,
        )
        if self.separate_lig:
            self.lig_separate_atom_embedder = AtomEncoder(
                emb_dim=emb_dim,
                feature_dims=lig_feature_dims,
                use_scalar_feat=use_scalar_features,
                n_feats_to_use=num_lig_feats,
            )
        rec_dims = rec_atom_feature_dims if use_rec_atoms else rec_residue_feature_dims
        self.rec_embedder = AtomEncoder(
            emb_dim=emb_dim,
            feature_dims=rec_dims,
            use_scalar_feat=use_scalar_features,
        )

        # ---- IEGMN layers ----
        # First layer input includes optional mu_r_norm (5 extra dims)
        input_node_dim = residue_emb_dim + (5 if use_mean_node_features else 0)

        layer_kwargs = dict(
            nonlin=nonlin,
            cross_msgs=cross_msgs,
            leakyrelu_neg_slope=leakyrelu_neg_slope,
            debug=debug,
            device=device,
            dropout=dropout,
            save_trajectories=save_trajectories,
            lig_input_edge_feats_dim=lig_input_edge_feats_dim,
            rec_input_edge_feats_dim=rec_input_edge_feats_dim,
            use_dist_in_layers=use_dist_in_layers,
            skip_weight_h=skip_weight_h,
            x_connection_init=x_connection_init,
            **kwargs,
        )

        self.iegmn_layers = nn.ModuleList()
        self.iegmn_layers.append(
            IEGMN_Layer(
                orig_h_feats_dim=input_node_dim,
                h_feats_dim=input_node_dim,
                out_feats_dim=iegmn_lay_hid_dim,
                **layer_kwargs,
            )
        )

        def _make_layer() -> IEGMN_Layer:
            return IEGMN_Layer(
                orig_h_feats_dim=input_node_dim,
                h_feats_dim=iegmn_lay_hid_dim,
                out_feats_dim=iegmn_lay_hid_dim,
                **layer_kwargs,
            )

        if shared_layers:
            shared = _make_layer()
            for _ in range(1, n_lays):
                self.iegmn_layers.append(shared)
        else:
            for _ in range(1, n_lays):
                self.iegmn_layers.append(_make_layer())

        if self.separate_lig:
            self.iegmn_layers_separate = nn.ModuleList()
            self.iegmn_layers_separate.append(
                IEGMN_Layer(
                    orig_h_feats_dim=input_node_dim,
                    h_feats_dim=input_node_dim,
                    out_feats_dim=iegmn_lay_hid_dim,
                    **layer_kwargs,
                )
            )
            if shared_layers:
                shared_sep = _make_layer()
                for _ in range(1, n_lays):
                    self.iegmn_layers_separate.append(shared_sep)
            else:
                for _ in range(1, n_lays):
                    self.iegmn_layers_separate.append(_make_layer())

        self.out_feats_dim = iegmn_lay_hid_dim

    def reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p, gain=1.0)
            else:
                nn.init.zeros_(p)

    def forward(
        self,
        lig_data: Data,
        rec_data: Data,
        geom_data: Optional[Data] = None,
        epoch: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float | torch.Tensor]:
        """Encode ligand and receptor graphs.

        Returns
        -------
        h_feats_lig  : (N_lig, out_feats_dim)
        h_feats_rec  : (N_rec, out_feats_dim)
        coords_lig   : (N_lig, 3)
        coords_rec   : (N_rec, 3)
        geom_losses  : scalar
        """
        dev = self.device

        # --- Extract tensors from PyG Data objects ---
        coords_lig = lig_data.pos.to(dev)
        coords_rec = rec_data.pos.to(dev)
        edge_index_lig = lig_data.edge_index.to(dev)
        edge_attr_lig = lig_data.edge_attr.to(dev)
        edge_index_rec = rec_data.edge_index.to(dev)
        edge_attr_rec = rec_data.edge_attr.to(dev)
        batch_lig = _batch_of(lig_data, dev)
        batch_rec = _batch_of(rec_data, dev)

        geom_edge_index: Optional[torch.Tensor] = None
        geom_edge_attr: Optional[torch.Tensor] = None
        if geom_data is not None:
            geom_edge_index = geom_data.edge_index.to(dev)
            geom_edge_attr = geom_data.edge_attr.to(dev)

        # --- Embed raw features ---
        h_lig = self.lig_atom_embedder(lig_data.x.to(dev))
        h_rec = self.rec_embedder(rec_data.x.to(dev))

        # --- Optional random noise dimensions ---
        if self.random_vec_dim > 0:
            dist = torch.distributions.Normal(0, self.random_vec_std)
            h_lig = torch.cat([h_lig, dist.sample([h_lig.size(0), self.random_vec_dim]).to(dev)], dim=1)
            h_rec = torch.cat([h_rec, dist.sample([h_rec.size(0), self.random_vec_dim]).to(dev)], dim=1)

        # --- Training noise (decays over epochs) ---
        if self.noise_initial > 0:
            noise = self.noise_initial * self.noise_decay_rate ** (epoch + 1)
            h_lig = h_lig + noise * torch.randn_like(h_lig)
            h_rec = h_rec + noise * torch.randn_like(h_rec)
            coords_lig = coords_lig + noise * torch.randn_like(coords_lig)
            coords_rec = coords_rec + noise * torch.randn_like(coords_rec)

        # --- Directional mean-norm features ---
        if self.use_mean_node_features:
            h_lig = torch.cat([h_lig, torch.log(lig_data.mu_r_norm.to(dev))], dim=1)
            h_rec = torch.cat([h_rec, torch.log(rec_data.mu_r_norm.to(dev))], dim=1)

        orig_h_lig = h_lig
        orig_h_rec = h_rec
        orig_coords_lig = coords_lig
        orig_coords_rec = coords_rec

        # --- Optionally zero out edge features ---
        edge_attr_lig = edge_attr_lig * float(self.use_edge_features_in_gmn)
        edge_attr_rec = edge_attr_rec * float(self.use_edge_features_in_gmn)

        # --- Cross-attention mask ---
        mask: Optional[torch.Tensor] = None
        if self.cross_msgs:
            mask = get_mask(batch_lig, batch_rec, dev)

        # --- Separate-lig branch initialisation ---
        if self.separate_lig:
            coords_lig_sep = coords_lig.clone()
            h_lig_sep = h_lig.clone()
            coords_rec_sep = coords_rec.clone()
            h_rec_sep = h_rec.clone()

        geom_losses: float | torch.Tensor = 0

        for layer in self.iegmn_layers:
            coords_lig, h_lig, coords_rec, h_rec, traj, g_loss = layer(
                h_lig=h_lig,
                coords_lig=coords_lig,
                edge_index_lig=edge_index_lig,
                edge_attr_lig=edge_attr_lig,
                batch_lig=batch_lig,
                h_rec=h_rec,
                coords_rec=coords_rec,
                edge_index_rec=edge_index_rec,
                edge_attr_rec=edge_attr_rec,
                batch_rec=batch_rec,
                orig_h_lig=orig_h_lig,
                orig_coords_lig=orig_coords_lig,
                orig_h_rec=orig_h_rec,
                orig_coords_rec=orig_coords_rec,
                mask=mask,
                geom_edge_index=geom_edge_index,
                geom_edge_attr=geom_edge_attr,
            )
            if not self.separate_lig:
                geom_losses = geom_losses + g_loss

        if self.separate_lig:
            for layer in self.iegmn_layers_separate:
                coords_lig_sep, h_lig_sep, coords_rec_sep, h_rec_sep, traj, g_loss = layer(
                    h_lig=h_lig_sep,
                    coords_lig=coords_lig_sep,
                    edge_index_lig=edge_index_lig,
                    edge_attr_lig=edge_attr_lig,
                    batch_lig=batch_lig,
                    h_rec=h_rec_sep,
                    coords_rec=coords_rec_sep,
                    edge_index_rec=edge_index_rec,
                    edge_attr_rec=edge_attr_rec,
                    batch_rec=batch_rec,
                    orig_h_lig=orig_h_lig,
                    orig_coords_lig=orig_coords_lig,
                    orig_h_rec=orig_h_rec,
                    orig_coords_rec=orig_coords_rec,
                    mask=mask,
                    geom_edge_index=geom_edge_index,
                    geom_edge_attr=geom_edge_attr,
                )
                geom_losses = geom_losses + g_loss
            coords_lig = coords_lig_sep

        return h_lig, h_rec, coords_lig, coords_rec, geom_losses

    def __repr__(self) -> str:
        n = len(self.iegmn_layers)
        return (
            f"EquiBindEncoder(n_lays={n}, out_dim={self.out_feats_dim}, "
            f"cross_msgs={self.cross_msgs})"
        )
