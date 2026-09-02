"""The docking head: keypoint attention followed by a Kabsch fit.

Ported from `models/equibind.py` in HannesStark/EquiBind. The encoder evolves coordinates
during message passing, but those are *not* the predicted pose — feeding an RDKit conformer
through the encoder alone moves it further from the crystal structure, not closer. The pose
comes from here: each side attends to K keypoints, and the rigid transform aligning the two
keypoint sets is solved in closed form.

Only the branch the published weights were trained with is implemented, which is the one
selected by `runs/flexible_self_docking/train_arguments.yaml`:

    centroid_keypts_construction*  false   keypoints are att @ coords, no centring
    lig_no_softmax, rec_no_softmax false   attention is softmaxed
    unnormalized_kpt_weights       false   no learned per-atom scaling
    num_att_heads                  30

The other branches exist in the original and are deliberately left out rather than written
untested — a flag nobody exercises is a flag nobody can trust.
"""

import math

import torch
import torch.nn as nn


class DockingHead(nn.Module):
    """Predicts the rigid transform placing a ligand into its pocket."""

    def __init__(self, feats_dim=64, num_att_heads=30, dropout=0.1, leakyrelu_neg_slope=0.01):
        super().__init__()
        self.num_att_heads = num_att_heads
        self.out_feats_dim = feats_dim

        # bias=False matches the checkpoint: these carry weight only
        self.keypts_attention_lig = nn.Sequential(
            nn.Linear(feats_dim, num_att_heads * feats_dim, bias=False))
        self.keypts_queries_lig = nn.Sequential(
            nn.Linear(feats_dim, num_att_heads * feats_dim, bias=False))
        self.keypts_attention_rec = nn.Sequential(
            nn.Linear(feats_dim, num_att_heads * feats_dim, bias=False))
        self.keypts_queries_rec = nn.Sequential(
            nn.Linear(feats_dim, num_att_heads * feats_dim, bias=False))

        self.h_mean_lig = nn.Sequential(
            nn.Linear(feats_dim, feats_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(negative_slope=leakyrelu_neg_slope),
        )
        self.h_mean_rec = nn.Sequential(
            nn.Linear(feats_dim, feats_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(negative_slope=leakyrelu_neg_slope),
        )

    def _keypoints(self, feats, coords, query_mean, attention, queries):
        """K keypoints as an attention-weighted combination of evolved coordinates."""
        d = feats.shape[1]
        weights = (attention(feats).view(-1, self.num_att_heads, d).transpose(0, 1)
                   @ queries(query_mean).view(1, self.num_att_heads, d).transpose(0, 1).transpose(1, 2)
                   / math.sqrt(d)).view(self.num_att_heads, -1)
        weights = torch.softmax(weights, dim=1)
        return weights @ coords

    def forward(self, h_lig, h_rec, coords_lig, coords_rec):
        """Returns (rotation 3x3, translation 1x3, lig_keypts, rec_keypts)."""
        lig_mean = torch.mean(self.h_mean_lig(h_lig), dim=0, keepdim=True)
        rec_mean = torch.mean(self.h_mean_rec(h_rec), dim=0, keepdim=True)

        # each side is queried by the *other* side's mean representation
        lig_keypts = self._keypoints(h_lig, coords_lig, rec_mean,
                                     self.keypts_attention_lig, self.keypts_queries_lig)
        rec_keypts = self._keypoints(h_rec, coords_rec, lig_mean,
                                     self.keypts_attention_rec, self.keypts_queries_rec)

        rec_keypts_mean = rec_keypts.mean(dim=0, keepdim=True)
        lig_keypts_mean = lig_keypts.mean(dim=0, keepdim=True)

        a = ((rec_keypts - rec_keypts_mean).transpose(0, 1)
             @ (lig_keypts - lig_keypts_mean) / float(self.num_att_heads))

        # The original perturbs A until the SVD is well conditioned; a near-degenerate
        # covariance makes the rotation arbitrary rather than merely imprecise.
        u, s, vt = torch.linalg.svd(a)
        for _ in range(10):
            eye = torch.eye(3, device=a.device)
            if (s.min() >= 1e-3
                    and torch.min(torch.abs((s ** 2).view(1, 3) - (s ** 2).view(3, 1) + eye)) >= 1e-2):
                break
            a = a + torch.rand(3, 3, device=a.device) * eye
            u, s, vt = torch.linalg.svd(a)
        else:
            raise RuntimeError("SVD consistently ill-conditioned")

        corr = torch.diag(torch.tensor([1.0, 1.0, float(torch.sign(torch.det(a)))], device=a.device))
        rotation = (u @ corr) @ vt
        translation = rec_keypts_mean - torch.t(rotation @ lig_keypts_mean.t())
        return rotation, translation, lig_keypts, rec_keypts
