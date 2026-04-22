# Ported from HannesStark/EquiBind (MIT License)
# https://github.com/HannesStark/EquiBind/blob/main/commons/process_mols.py
#
# Feature dimensions for ligand atoms and receptor residues/atoms.
# AtomEncoder maps raw feature vectors → fixed-size embeddings used by IEGMN.

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Feature dimension specs
#   Each tuple is (list_of_categorical_vocab_sizes, num_scalar_features)
# ---------------------------------------------------------------------------

# Ligand atom features (16 categorical + 1 scalar Gasteiger charge)
#   Categorical: atomic_num, chirality, degree, formal_charge, implicit_valence,
#                total_H, radical_e, hybridization, is_aromatic, num_rings,
#                in_ring3/4/5/6/7/8
lig_feature_dims = (
    [119, 4, 12, 12, 8, 10, 6, 6, 2, 8, 2, 2, 2, 2, 2, 2],
    1,  # Gasteiger partial charge
)

# Receptor residue-level features (C-alpha graph, default)
#   Categorical: amino_acid_type (37 classes)
#   Scalar: SASA, B-factor
rec_residue_feature_dims = (
    [37],
    2,  # SASA, B-factor
)

# Receptor atom-level features (surface + C-alpha graph)
#   Categorical: amino_acid_type, atomic_num, atom_type_2char, atom_type_3char
#   Scalar: SASA, B-factor
rec_atom_feature_dims = (
    [37, 119, 23, 38],
    2,  # SASA, B-factor
)


# ---------------------------------------------------------------------------
# AtomEncoder
# ---------------------------------------------------------------------------

class AtomEncoder(nn.Module):
    """Embed raw atom/residue feature vectors into a fixed-size space.

    Each categorical feature column gets its own nn.Embedding; all are summed.
    Scalar features are linearly projected and added to the same sum.
    Output dimension is always `emb_dim`.
    """

    def __init__(self, emb_dim: int, feature_dims: tuple, use_scalar_feat: bool = True,
                 n_feats_to_use: int = None):
        super().__init__()
        self.use_scalar_feat = use_scalar_feat
        self.n_feats_to_use = n_feats_to_use
        self.num_categorical_features = len(feature_dims[0])
        self.num_scalar_features = feature_dims[1]

        self.atom_embedding_list = nn.ModuleList()
        for i, dim in enumerate(feature_dims[0]):
            emb = nn.Embedding(dim, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)
            if i + 1 == self.n_feats_to_use:
                break

        if self.num_scalar_features > 0:
            self.linear = nn.Linear(self.num_scalar_features, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.num_categorical_features + self.num_scalar_features
        x_embedding = 0
        for i in range(len(self.atom_embedding_list)):
            x_embedding = x_embedding + self.atom_embedding_list[i](x[:, i].long())
            if i + 1 == self.n_feats_to_use:
                break
        if self.num_scalar_features > 0 and self.use_scalar_feat:
            x_embedding = x_embedding + self.linear(
                x[:, self.num_categorical_features:].float()
            )
        return x_embedding
