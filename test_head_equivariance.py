"""Is the docking head equivariant? It is by construction, except for one line — measure which.

`test_equivariance.py` checks the IEGMN encoder and says so in its own docstring. That leaves the
part the authors actually published untested: the keypoint-attention head that solves for the
rigid transform placing the ligand. `dock` is the capability with a golden file, so this is the
output that matters, and a claim about it should be measured rather than reasoned about.

**The reasoning, so the measurement has something to contradict.** Under a global rotation `R`
applied to receptor and ligand together:

    keypoints are a weighted sum of coordinates with weights from *features*, so K -> R K
    a = (rec - rec_mean)^T (lig - lig_mean)  ->  R a R^T
    SVD: a = U S V^T  =>  R a R^T = (R U) S (R V)^T
    rotation = (U corr) V^T  ->  R (U corr) V^T R^T = R rotation R^T

which is exactly the equivariant transformation, so the predicted pose rotates with the input.
One line breaks it:

    a = a + torch.rand(3, 3) * eye          # head.py

The perturbation is diagonal **in the world basis**, and `R a R^T + rand*I` is not
`R (a + rand*I) R^T`. The authors added it because a near-degenerate covariance makes the solved
rotation arbitrary rather than merely imprecise, and on real complexes the keypoints are close to
coplanar, so it fires. That is upstream's trade, reproduced faithfully; what was missing is a
number for what it costs.

**The isolation, and the candidate fix in the same move.** The perturbation cannot simply be
switched off — a near-degenerate covariance is exactly why it exists, and without it the loop
exhausts its ten attempts and raises. What it can be replaced with is a *scalar* multiple of the
identity, because `R (a + cI) R^T = R a R^T + cI`: that conditions the SVD just as well and
commutes with rotation. So the second row of each pair below is both the isolation and a one-line
repair, and the contrast between the two rows is what the random diagonal costs.

usage, inside the image:
    python test_head_equivariance.py --complexes /data --model /ckpt/best_checkpoint.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from rdkit import RDLogger
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dock_complexes import load_head, random_placement
from embed_complexes import featurize, load_encoder

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


def solve(head, h_lig, h_rec, coords_lig, coords_rec):
    """The head's transform, and whether its conditioning fix had to intervene."""
    fired = {"count": 0}
    original = torch.rand

    def counting_rand(*size, **kwargs):
        fired["count"] += 1
        return original(*size, **kwargs)

    torch.rand = counting_rand
    try:
        rotation, translation, _lk, _rk = head(h_lig, h_rec, coords_lig, coords_rec)
    finally:
        torch.rand = original
    return rotation, translation, fired["count"]


def place(rotation, translation, coords_lig):
    """dock_complexes.py:180 — the predicted pose."""
    return ((rotation @ coords_lig.T).T + translation).detach().cpu().numpy()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--complexes", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    encoder = load_encoder(args.model, args.device, noise=False)
    head = load_head(args.model, args.device)
    head.eval()
    rng = np.random.RandomState(args.seed)

    ids = sorted(d for d in os.listdir(args.complexes)
                 if os.path.isdir(os.path.join(args.complexes, d)))[: args.n]

    worst_with, worst_without, fired_total, trials = 0.0, 0.0, 0, 0
    for cid in ids:
        lig_graph, rec_graph, geom_graph = featurize(os.path.join(args.complexes, cid), cid)
        start = random_placement(lig_graph.pos.numpy(), rng=np.random.RandomState(args.seed))
        lig_graph.pos = torch.as_tensor(start)

        for trial in range(args.trials):
            rotation_world = torch.as_tensor(
                Rotation.random(random_state=rng.randint(2 ** 31)).as_matrix(),
                dtype=torch.float32)

            for equivariant_fix in (False, True):
                # `torch.rand(3, 3) * eye` becomes `ones(3, 3) * eye`, i.e. the identity: the
                # same conditioning, but a perturbation that commutes with rotation.
                if equivariant_fix:
                    head_rand = torch.rand
                    torch.rand = lambda *a, **k: torch.ones(*a, **k)
                try:
                    torch.manual_seed(args.seed)
                    base_lig, base_rec = lig_graph.clone(), rec_graph.clone()
                    h_lig, h_rec, x_lig, x_rec = encoder(
                        base_lig, base_rec, geom_graph.clone())[:4]
                    rot_a, trans_a, fired_a = solve(head, h_lig, h_rec, x_lig, x_rec)
                    placed_a = place(rot_a, trans_a, x_lig)

                    turned_lig, turned_rec = lig_graph.clone(), rec_graph.clone()
                    turned_lig.pos = turned_lig.pos @ rotation_world.T
                    turned_rec.pos = turned_rec.pos @ rotation_world.T
                    torch.manual_seed(args.seed)
                    h_lig2, h_rec2, x_lig2, x_rec2 = encoder(
                        turned_lig, turned_rec, geom_graph.clone())[:4]
                    rot_b, trans_b, _fired_b = solve(head, h_lig2, h_rec2, x_lig2, x_rec2)
                    placed_b = place(rot_b, trans_b, x_lig2)
                finally:
                    if equivariant_fix:
                        torch.rand = head_rand

                # equivariance: placing the turned input should give the turned placement
                expected = placed_a @ rotation_world.numpy().T
                deviation = float(np.abs(expected - placed_b).max())
                if equivariant_fix:
                    worst_without = max(worst_without, deviation)
                else:
                    worst_with = max(worst_with, deviation)
                    fired_total += fired_a
                    trials += 1
                label = "identity perturbation" if equivariant_fix else "as published         "
                print(f"  {cid} trial {trial}, {label}: max|Δ| {deviation:.3e} A"
                      + ("" if equivariant_fix else f"   (fired {fired_a}x)"))

    print(f"\nas published (random diagonal):  worst {worst_with:.3e} A, "
          f"fired in {fired_total} of {trials} solves")
    print(f"identity perturbation:           worst {worst_without:.3e} A")
    if worst_with > 1e-2 and worst_without < worst_with / 10:
        print("\nThe head is equivariant; the random diagonal in the conditioning fix is what\n"
              "breaks it, and a scalar multiple of the identity conditions the SVD just as well\n"
              "while commuting with rotation. Changing it would depart from the published\n"
              "method, so it is recorded rather than applied — the weights were trained with\n"
              "the random version in place.")
    elif worst_with < 1e-2:
        print("\nThe head is equivariant to float precision, perturbation and all.")
    else:
        print("\nThe head departs from equivariance even with an equivariant perturbation,\n"
              "which the derivation above says it should not. Something else is wrong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
