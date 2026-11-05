"""Extract complex-level embeddings from the EquiBind encoder.

EquiBind was never trained on affinity — it learns to place a ligand in a pocket. The head
is already absent from this port, so what comes out is the IEGMN's per-node representations
for both sides, plus the coordinates it evolved them to.

Pooling each side and concatenating gives 2 x 64 dims. EquiBind has no affinity head, so the
pooling is ours, not the authors'.

usage:
    python embed_complexes.py --complexes DIR --model checkpoints/best_checkpoint.pt \
        --out /outputs/embeddings.npz --pool sum
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import torch
from rdkit import RDLogger
from tqdm import tqdm

from equibind.features.featurize import (get_calpha_graph, get_geometry_graph,
                                         get_lig_graph_revised, get_receptor_inference,
                                         read_molecule)
from equibind.models.encoder import EquiBindEncoder

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

POOLS = {"sum": lambda t: t.sum(0), "mean": lambda t: t.mean(0), "max": lambda t: t.max(0).values}

# from runs/flexible_self_docking/train_arguments.yaml — the configuration the published
# weights were trained with. Changing any of these stops the checkpoint from loading.
ENCODER_CONFIG = dict(
    n_lays=8, layer_norm="BN", layer_norm_coords=0, final_h_layer_norm=0, dropout=0.1,
    use_rec_atoms=False, cross_msgs=True, shared_layers=False, noise_decay_rate=0.5,
    noise_initial=1, residue_emb_dim=64, standard_norm_order=True, nonlin="lkyrelu",
    leakyrelu_neg_slope=0.01, use_dist_in_layers=True, normalize_coordinate_update=True,
    # 10, not the port's default of 1.0. Ca-Ca distances reach 30 A, so squared distances
    # reach ~900; without dividing by this the exp(-d^2/sigma) distance features underflow
    # to zero, the receptor edge MLP sees nothing, and receptor coordinates drift to
    # thousands of angstroms. Carries no weights, so a wrong value loads silently.
    rec_square_distance_scale=10,
    # These four also differ from the port's defaults and carry no weights, so a wrong
    # value loads silently and shows up only as bad numbers. use_scalar_features is the
    # dangerous one: it changes what goes into the node features without changing their
    # shape, so the checkpoint still matches 597/597 while the encoder computes something
    # else entirely.
    use_scalar_features=False,
    geometry_regularization=True,
    geometry_reg_step_size=0.001,
)
LIG_RADIUS = 5        # lig_graph_radius
REC_CUTOFF = 30       # rec_graph_radius
REC_MAX_NEIGHBOURS = 10  # c_alpha_max_neighbors


def load_encoder(path: str, device: str, noise: bool = True) -> EquiBindEncoder:
    """Build the encoder and restore the published weights.

    `noise` exists because EquiBind's forward adds `noise_level * randn_like(...)` to features
    and coordinates whenever `noise_initial > 0`, and there is no `self.training` guard — so
    the authors' own settings make *inference* stochastic, at level 0.5 on the first epoch.
    That is faithful, and it is also why two embedding runs of the same complex disagree.
    Passing noise=False keeps everything else identical and makes the encoder deterministic,
    which is what a frozen representation for transfer should be.
    """
    config = dict(ENCODER_CONFIG)
    if not noise:
        config["noise_initial"] = 0
    enc = EquiBindEncoder(device=device, **config).to(device)
    ck = torch.load(path, map_location=device, weights_only=True)
    state = ck["model_state_dict"] if "model_state_dict" in ck else ck
    # the authors prefix the whole encoder with `iegmn.`; what remains after stripping it
    # matches this port exactly, and the 8 ignored keys are the docking head we do not have
    missing, _ignored = enc.load_state_dict(
        {k[len("iegmn."):]: v for k, v in state.items() if k.startswith("iegmn.")}, strict=False)
    if missing:
        raise RuntimeError(f"{len(missing)} parameters were not restored: {missing[:5]}")
    enc.eval()
    return enc


def featurize(complex_dir: str, cid: str):
    ligand = read_molecule(f"{complex_dir}/{cid}_ligand.mol2", sanitize=True, remove_hs=True)
    if ligand is None:
        ligand = read_molecule(f"{complex_dir}/{cid}_ligand.sdf", sanitize=True, remove_hs=True)
    if ligand is None:
        raise ValueError("RDKit could not read the ligand")

    lig_graph = get_lig_graph_revised(ligand, cid, radius=LIG_RADIUS,
                                      max_neighbors=None, use_rdkit_coords=False)
    rec, _coords, c_alpha, n, c = get_receptor_inference(f"{complex_dir}/{cid}_protein.pdb")
    rec_graph = get_calpha_graph(rec, c_alpha, n, c, cutoff=REC_CUTOFF,
                                 max_neighbor=REC_MAX_NEIGHBOURS)
    # The trained weights were fitted with geometry_regularization on, and each IEGMN layer
    # uses this graph to pull evolved coordinates back towards the bonded distances. Omitting
    # it is silent — the encoder simply skips the correction — but it changes the node
    # representations too, because later layers derive their messages from distances.
    geom_graph = get_geometry_graph(ligand)
    return lig_graph, rec_graph, geom_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed complexes with the EquiBind encoder")
    parser.add_argument("--complexes", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pool", default="sum", choices=sorted(POOLS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0,
                        help="pins torch's global generator; load-bearing for reproducibility")
    parser.add_argument("--noise", action="store_true",
                        help="restore the authors' inference-time noise. EquiBind adds it in "
                             "forward with no training guard, so this makes repeated runs of "
                             "the same complex disagree — by up to 1.6e3 per component here")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    # Noise off by default. A frozen representation that changes between two runs of the same
    # complex cannot be transferred from, compared against, or held to a golden — and the
    # noise is a training-time augmentation that reached inference through a missing
    # `self.training` guard rather than a modelling choice.
    torch.manual_seed(args.seed)
    encoder = load_encoder(args.model, device, noise=args.noise)
    pool = POOLS[args.pool]

    dirs = sorted(d for d in os.listdir(args.complexes)
                  if os.path.isdir(os.path.join(args.complexes, d)))
    ids: list[str] = []
    vectors: list[np.ndarray] = []
    failed: list[str] = []

    for cid in tqdm(dirs, desc="Embedding"):
        try:
            lig_graph, rec_graph, geom_graph = featurize(
                os.path.join(args.complexes, cid), cid)
            with torch.no_grad():
                # (h_lig, h_rec, coords_lig, coords_rec, geom_losses) — only the first two
                # are the representation; the coordinates are where it moved the atoms
                h_lig, h_rec = encoder(lig_graph.to(device), rec_graph.to(device),
                                       geom_graph.to(device))[:2]
            vectors.append(torch.cat([pool(h_lig), pool(h_rec)]).cpu().numpy())
            ids.append(cid)
        except Exception as e:
            failed.append(f"{cid}: {type(e).__name__}: {e}")

    if failed:
        print(f"\n{len(failed)} complexes failed:")
        for f in failed[:10]:
            print("  ", f)
    if not vectors:
        raise RuntimeError("nothing embedded")

    matrix = np.stack(vectors)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, ids=np.array(ids), vectors=matrix)
    print(f"\n{len(ids)} embeddings of dimension {matrix.shape[1]} -> {args.out}")


if __name__ == "__main__":
    main()
