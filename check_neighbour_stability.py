"""Would a stable sort make the receptor graph rotation-stable? Measure, rather than argue.

`get_calpha_graph` keeps the `max_neighbor` nearest residues inside a cutoff:

    dst = list(np.where(distances[i, :] < cutoff)[0])
    dst.remove(i)
    if max_neighbor != None and len(dst) > max_neighbor:
        dst = list(np.argsort(distances[i, :]))[1: max_neighbor + 1]

With cutoff 30 A and max_neighbor 10, the second branch fires for nearly every residue, and
`np.argsort` defaults to an unstable quicksort. That invites an obvious fix — pass
`kind="stable"` — and the fix only works if the neighbours being swapped are at *exactly* equal
distances. If instead the two distances differ by an epsilon that a rotation is enough to flip,
no sorting algorithm can help, because the comparison it is given genuinely changed.

So this measures two things per residue, before and after a motion:

    does the selected *set* of neighbours change?   a stable sort cannot fix a changed set
    how large is the gap at the selection boundary? an exact tie is fixable, an epsilon is not

usage, inside the image:
    python check_neighbour_stability.py --complexes /data --n 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.spatial as spa

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embed_complexes import REC_CUTOFF, REC_MAX_NEIGHBOURS  # noqa: E402
from equibind.features.featurize import get_receptor_inference  # noqa: E402


def neighbours(c_alpha: np.ndarray, cutoff: float, max_neighbor: int, stable: bool):
    """The authors' selection, with the sort kind as the one thing that varies."""
    distances = spa.distance.cdist(c_alpha, c_alpha)
    kind = "stable" if stable else "quicksort"
    chosen, boundary_gap = [], []
    for i in range(len(c_alpha)):
        dst = list(np.where(distances[i, :] < cutoff)[0])
        dst.remove(i)
        if max_neighbor is not None and len(dst) > max_neighbor:
            order = list(np.argsort(distances[i, :], kind=kind))
            dst = order[1: max_neighbor + 1]
            # the gap between the last neighbour kept and the first one dropped: if this is
            # zero the boundary is a true tie, and a stable sort decides it reproducibly
            kept, dropped = order[max_neighbor], order[max_neighbor + 1]
            boundary_gap.append(distances[i, dropped] - distances[i, kept])
        chosen.append(frozenset(int(j) for j in dst))
    return chosen, np.asarray(boundary_gap)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--complexes", required=True)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.RandomState(args.seed)
    import os

    ids = sorted(d for d in os.listdir(args.complexes)
                 if os.path.isdir(os.path.join(args.complexes, d)))[: args.n]

    from scipy.spatial.transform import Rotation

    for cid in ids:
        rec, _coords, c_alpha, _n, _c = get_receptor_inference(
            f"{args.complexes}/{cid}/{cid}_protein.pdb")
        c_alpha = np.asarray(c_alpha, dtype=np.float64)

        base_q, gaps = neighbours(c_alpha, REC_CUTOFF, REC_MAX_NEIGHBOURS, stable=False)
        base_s, _ = neighbours(c_alpha, REC_CUTOFF, REC_MAX_NEIGHBOURS, stable=True)

        rotation = Rotation.random(random_state=rng.randint(2 ** 31)).as_matrix()
        turned = c_alpha @ rotation.T
        moved_q, _ = neighbours(turned, REC_CUTOFF, REC_MAX_NEIGHBOURS, stable=False)
        moved_s, _ = neighbours(turned, REC_CUTOFF, REC_MAX_NEIGHBOURS, stable=True)

        changed_q = sum(a != b for a, b in zip(base_q, moved_q))
        changed_s = sum(a != b for a, b in zip(base_s, moved_s))
        sort_matters = sum(a != b for a, b in zip(base_q, base_s))

        print(f"\n{cid}: {len(c_alpha)} residues, {len(gaps)} hit the max_neighbor branch")
        print(f"  neighbour sets changed by a rotation, quicksort: {changed_q}")
        print(f"  neighbour sets changed by a rotation, stable:    {changed_s}")
        print(f"  residues where the sort kind alone changes the set: {sort_matters}")
        if len(gaps):
            exact_ties = int((gaps == 0).sum())
            print(f"  gap at the selection boundary: min {gaps.min():.3e} A, "
                  f"median {np.median(gaps):.3e} A")
            print(f"  exact ties at that boundary: {exact_ties} of {len(gaps)}")
            tiny = int((gaps < 1e-9).sum())
            print(f"  gaps below 1e-9 A (what a stable sort could decide): {tiny}")

    print("\nA stable sort fixes exact ties. A changed *set* under rotation is not a tie —")
    print("the distances themselves moved, and the comparison genuinely reversed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
