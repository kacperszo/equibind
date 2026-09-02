"""Check that the encoder is E(3)-equivariant, which it is by construction.

Rotating and translating the input must leave the node features unchanged and rotate the
output coordinates with it. If it does not, some layer reads absolute coordinates where it
should read relative ones — and matching parameter shapes will never reveal that, because
the model runs perfectly well while being wrong.

**This tests the encoder, not the full docking pipeline**, and that is deliberate. Two parts
of the published method are genuinely non-equivariant and we reproduce them faithfully:

  * Gaussian noise is added to features and coordinates inside the forward pass, scaled by
    `noise_initial * noise_decay_rate ** (epoch + 1)`. Both the authors' inference and ours
    pass `epoch=0`, so published inference runs with noise of 0.5 and is not deterministic.
  * The Kabsch fit perturbs its covariance with `torch.rand` until well conditioned. On real
    complexes the keypoints are nearly coplanar — singular values around 2, 0.02, 6e-05 — so
    it fires every time, and because the perturbation lives in the world frame it makes the
    solved rotation depend on the input orientation.

Together those put roughly 0.3-0.5 A of irreducible spread on the predicted pose. The encoder
underneath is exact to float32, and it is the encoder the embeddings come from.

usage:
    python test_equivariance.py --complex /tmp/one/1c5z --model checkpoints/best_checkpoint.pt
"""

import argparse
import os
import warnings

import numpy as np
import torch
from rdkit import RDLogger
from scipy.spatial.transform import Rotation

from embed_complexes import (ENCODER_CONFIG, LIG_RADIUS, REC_CUTOFF, REC_MAX_NEIGHBOURS,
                             load_encoder, read_molecule)
from equibind.features.featurize import (get_calpha_graph, get_lig_graph_revised,
                                         get_receptor_inference)

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


def main() -> None:
    parser = argparse.ArgumentParser(description="E(3) equivariance check for the encoder")
    parser.add_argument("--complex", required=True, help="one complex directory")
    parser.add_argument("--model", required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    cid = os.path.basename(args.complex.rstrip("/"))
    ENCODER_CONFIG["noise_initial"] = 0  # measure the layers, not the noise
    encoder = load_encoder(args.model, "cpu")

    ligand = read_molecule(f"{args.complex}/{cid}_ligand.mol2", sanitize=True, remove_hs=True)
    if ligand is None:
        ligand = read_molecule(f"{args.complex}/{cid}_ligand.sdf", sanitize=True, remove_hs=True)

    lig_graph = get_lig_graph_revised(ligand, cid, radius=LIG_RADIUS,
                                      max_neighbors=None, use_rdkit_coords=True)
    lig_coords = lig_graph.new_x.numpy().astype(np.float32)
    rec, _c, c_alpha, n, c = get_receptor_inference(f"{args.complex}/{cid}_protein.pdb")
    rec_graph = get_calpha_graph(rec, c_alpha, n, c, cutoff=REC_CUTOFF,
                                 max_neighbor=REC_MAX_NEIGHBOURS)
    rec_coords = rec_graph.pos.numpy().copy()

    def encode(lig_pos, rec_pos):
        lig_graph.pos = torch.from_numpy(lig_pos)
        rec_graph.pos = torch.from_numpy(rec_pos)
        with torch.no_grad():
            return encoder(lig_graph, rec_graph)[:4]

    h_lig, h_rec, coords_lig, coords_rec = encode(lig_coords, rec_coords)

    rng = np.random.RandomState(0)
    worst = 0.0
    for trial in range(args.trials):
        rotation = Rotation.random(random_state=rng.randint(2 ** 31)).as_matrix().astype(np.float32)
        t = rng.randn(1, 3).astype(np.float32)

        def move(x):
            return ((rotation @ x.T).T + t).astype(np.float32)

        h_lig2, h_rec2, coords_lig2, coords_rec2 = encode(move(lig_coords), move(rec_coords))

        checks = {
            "h_lig invariant": float((h_lig - h_lig2).abs().max()),
            "h_rec invariant": float((h_rec - h_rec2).abs().max()),
            "coords_lig equivariant": float(
                (coords_lig2 - torch.from_numpy(move(coords_lig.numpy()))).abs().max()),
            "coords_rec equivariant": float(
                (coords_rec2 - torch.from_numpy(move(coords_rec.numpy()))).abs().max()),
        }
        worst = max(worst, max(checks.values()))
        print(f"trial {trial}: " + "  ".join(f"{k.split()[0]}={v:.2e}" for k, v in checks.items()))

    print(f"\nworst deviation {worst:.2e} (tolerance {args.tolerance})")
    if worst > args.tolerance:
        print("FAIL — a layer is reading absolute coordinates.")
        raise SystemExit(1)
    print("PASS — the encoder is equivariant to float32 precision.")


if __name__ == "__main__":
    main()
