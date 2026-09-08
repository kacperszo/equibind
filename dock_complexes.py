"""Predict a binding pose and measure it against the crystal structure.

Runs the full published pipeline: the IEGMN encoder, then the keypoint-attention head, then
the rigid transform it solves for. The evolved coordinates the encoder returns are *not* the
prediction — passing a conformer through the encoder alone moves it away from the true pose.
The placement comes from applying the head's rotation and translation to the input conformer.

The input is an RDKit conformer, as in training. Feeding the crystallographic pose — which
is what the embedding extraction does — would hand the model the answer.

Three numbers per complex, because a bare RMSD says nothing without something to beat:

    rmsd            the predicted pose against the crystal pose — the number to report
    rmsd_conformer  the raw RDKit conformer, unmoved — the do-nothing baseline
    rmsd_aligned    the conformer after an optimal rigid fit onto the true pose: the best
                    any rigid method could do, which separates "wrong shape" from "wrong
                    place" and bounds what this model could possibly achieve

usage:
    python dock_complexes.py --complexes /data --model /ckpt/best_checkpoint.pt \
        --out /outputs/predictions.csv
"""

from __future__ import annotations

import argparse
import os
import warnings

import zlib

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from embed_complexes import ENCODER_CONFIG, LIG_RADIUS, REC_CUTOFF, REC_MAX_NEIGHBOURS, load_encoder
from equibind.models.head import DockingHead
from equibind.features.featurize import (get_calpha_graph, get_lig_graph_revised,
                                         get_rdkit_coords, get_receptor_inference, read_molecule)

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


def random_placement(coords, translation_distance=5.0, rng=None):
    """Place the ligand the way the authors' inference does (inference.py:366).

    The conformer is centred at the origin, randomly rotated, and translated to a random
    point within `translation_distance` of it. The model therefore starts with no idea where
    the pocket is and has to find it from the receptor alone.

    Skipping this and feeding the Kabsch-aligned conformer directly puts the ligand in the
    pocket already, which is off-distribution and makes the model move it *away*.
    """
    rng = rng or np.random
    rotation = Rotation.random(num=1, random_state=rng.randint(2 ** 31)).as_matrix().squeeze()
    t = rng.randn(1, 3)
    t = t / np.sqrt((t * t).sum()) * rng.uniform(0, translation_distance)
    return ((rotation @ (coords - coords.mean(0, keepdims=True)).T).T + t).astype(np.float32)


def kabsch_rmsd(mobile: np.ndarray, target: np.ndarray) -> float:
    """RMSD after the optimal rigid superposition of `mobile` onto `target`."""
    m = mobile - mobile.mean(0)
    t = target - target.mean(0)
    u, _s, vt = np.linalg.svd(m.T @ t)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return float(np.sqrt(((m @ rot.T - t) ** 2).sum(1).mean()))


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(1).mean()))


HEAD_PREFIXES = {"keypts_attention_lig", "keypts_queries_lig", "keypts_attention_rec",
                 "keypts_queries_rec", "h_mean_lig", "h_mean_rec"}


def load_head(path, device):
    """The head lives under the same `iegmn.` prefix as the encoder in the checkpoint."""
    head = DockingHead(num_att_heads=30, dropout=ENCODER_CONFIG["dropout"],
                       leakyrelu_neg_slope=ENCODER_CONFIG["leakyrelu_neg_slope"]).to(device)
    ck = torch.load(path, map_location=device, weights_only=True)
    state = ck["model_state_dict"] if "model_state_dict" in ck else ck
    stripped = {k[len("iegmn."):]: v for k, v in state.items() if k.startswith("iegmn.")}
    selected = {k: v for k, v in stripped.items() if k.split(".")[0] in HEAD_PREFIXES}
    missing, _ = head.load_state_dict(selected, strict=False)
    if missing:
        raise RuntimeError(f"head parameters not restored: {missing}")
    head.eval()
    return head


def main() -> None:
    parser = argparse.ArgumentParser(description="RMSD of the encoder's evolved coordinates")
    parser.add_argument("--complexes", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--noise", action="store_true",
                        help="restore the authors' stochastic inference: EquiBind adds noise "
                             "in forward with no training guard, so this makes repeated runs "
                             "disagree by several angstroms")
    parser.add_argument("--seed", type=int, default=0,
                        help="the starting pose is random; this makes it reproducible, and "
                             "each complex is seeded from it independently")
    args = parser.parse_args()

    # Load-bearing, and this is the second of the two random sources: the Kabsch fit perturbs
    # its covariance with `torch.rand` until it is well conditioned, and on real complexes the
    # keypoints are nearly coplanar so it fires every time. With the encoder's noise off but
    # this unpinned, two runs still disagreed by up to 0.5 A on 14 of 24 complexes. Seeding
    # makes all 24 bit-identical, which is what lets a run of this be a golden.
    torch.manual_seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    # noise off by default, unlike the authors' inference. EquiBind's forward adds
    # `noise_level * randn_like(...)` with no `self.training` guard, so their own settings make
    # docking stochastic: two identical runs of these 24 complexes disagreed by up to 33 A.
    # A result that cannot be reproduced cannot be a golden, and the noise is a training-time
    # augmentation that leaked into inference through a missing guard rather than a modelling
    # choice. `--noise` restores their behaviour for anyone who wants it.
    encoder = load_encoder(args.model, device, noise=args.noise)
    head = load_head(args.model, device)

    dirs = sorted(d for d in os.listdir(args.complexes)
                  if os.path.isdir(os.path.join(args.complexes, d)))
    rows, failed = [], []

    for cid in tqdm(dirs, desc="Docking"):
        d = os.path.join(args.complexes, cid)
        try:
            ligand = read_molecule(f"{d}/{cid}_ligand.mol2", sanitize=True, remove_hs=True)
            if ligand is None:
                ligand = read_molecule(f"{d}/{cid}_ligand.sdf", sanitize=True, remove_hs=True)
            # copy, not a view: get_lig_graph_revised calls get_rdkit_coords, which embeds
            # a new conformer into this same mol object and would otherwise overwrite the
            # crystal coordinates we are about to measure against
            true_coords = np.array(ligand.GetConformer().GetPositions(), dtype=np.float32)

            # use_rdkit_coords=True generates an unbound conformer AND Kabsch-aligns it
            # onto the true pose, which is what puts the ligand in the pocket's frame. The
            # alignment is rigid, so the conformation is still RDKit's guess and the model
            # is not handed the answer — but skipping it leaves the ligand a hundred
            # angstroms from the receptor and every keypoint lands in empty space.
            lig_graph = get_lig_graph_revised(ligand, cid, radius=LIG_RADIUS,
                                              max_neighbors=None, use_rdkit_coords=True)
            # `pos` always holds the crystal pose; the aligned conformer goes to `new_x`,
            # mirroring the original's ndata['new_x']. The original feeds the encoder that
            # field (models/equibind.py:749), so `pos` has to be replaced with it or the
            # model is handed the answer it is supposed to predict.
            aligned = lig_graph.new_x.numpy().astype(np.float32)
            # Seeded per complex, not from a stream shared across the run. A single
            # RandomState consumed in a loop makes every complex's starting pose depend on
            # how many draws the ones before it used, so adding a complex, dropping one, or
            # reordering the directory shifts every pose after it — the recorded RMSDs then
            # differ on all 285 rather than on the one that changed. crc32 of the id, not
            # hash(), which Python randomises per process.
            placement_rng = np.random.RandomState(
                (args.seed + zlib.crc32(cid.encode())) % (2 ** 32))
            conformer = random_placement(aligned, rng=placement_rng)
            lig_graph.pos = torch.from_numpy(conformer)
            rec, _c, ca, n, c = get_receptor_inference(f"{d}/{cid}_protein.pdb")
            rec_graph = get_calpha_graph(rec, ca, n, c, cutoff=REC_CUTOFF,
                                         max_neighbor=REC_MAX_NEIGHBOURS)

            with torch.no_grad():
                h_lig, h_rec, coords_lig, coords_rec = encoder(
                    lig_graph.to(device), rec_graph.to(device))[:4]
                rotation, translation, _lk, _rk = head(h_lig, h_rec, coords_lig, coords_rec)
                # Applied to the *evolved* ligand coordinates, not the input ones
                # (models/equibind.py:1035, with the authors' use_evolved_lig: true). The
                # keypoints the Kabsch fit is solved from come from the evolved coordinates,
                # so applying the transform to the input instead makes the result depend on
                # where the input happened to be placed.
                predicted = ((rotation @ coords_lig.T).T + translation).cpu().numpy()

            rows.append({
                "complex_id": cid,
                "rmsd": rmsd(predicted, true_coords),
                "rmsd_conformer": rmsd(conformer, true_coords),
                "rmsd_aligned": kabsch_rmsd(aligned, true_coords),
            })
        except Exception as e:
            failed.append(f"{cid}: {type(e).__name__}: {e}")

    if failed:
        print(f"\n{len(failed)} failed:")
        for f in failed[:10]:
            print("  ", f)
    if not rows:
        raise RuntimeError("nothing docked")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\n{len(df)} complexes")
    for col in ("rmsd", "rmsd_conformer", "rmsd_aligned"):
        print(f"  {col:16s} median {df[col].median():6.2f}  mean {df[col].mean():6.2f}")
    print()
    for threshold in (2.0, 5.0):
        print(f"  within {threshold:.0f} A: {100 * (df.rmsd < threshold).mean():5.1f}%")
    better = (df.rmsd < df.rmsd_conformer).mean()
    print(f"  beats the untouched conformer in {100 * better:.0f}% of complexes")


if __name__ == "__main__":
    main()
