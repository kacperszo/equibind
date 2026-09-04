"""Compare this PyG port's featurisation against the authors' original, tensor by tensor.

The port rewrote EquiBind from DGL to PyG and split one 1050-line module into
`features/featurize.py`, `models/{encoder,layers,head}.py`. A rewrite that large is exactly
where a subtle feature-level error survives review: shapes still match, the model still runs,
and the numbers stay plausible. Reading the two files side by side does not settle it — the
port's line offsets differ, so a naive diff reports differences that are only blank lines and
misses a reordered feature entirely.

So this runs both featurisers on the same molecule and compares the arrays they produce.

`commons/process_mols.py` imports dgl at module level. With a real dgl present the authors'
graph builders run and everything below is compared; without one, dgl is stubbed so the
feature tables and the atom featuriser can still be checked inside the port's own container,
and the graph section reports itself skipped rather than silently passing.

The image for the full comparison holds both frameworks (dgl 2.1.0 for their builders, PyG
for ours) and exists only for this script — it never produces a result.

usage:
    python compare_with_upstream.py --ligand /data/1bcu/1bcu_ligand.sdf \
        --receptor /data/1bcu/1bcu_protein.pdb
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch


def load_upstream(repo: Path) -> tuple:
    """Import the authors' process_mols, preferring a real dgl over the stub.

    The stub only exists so the feature tables and atom featuriser can be compared inside the
    port's own container, where dgl is absent. It must never win when a real dgl is present:
    `dgl.graph` would then return None and every graph comparison would die on `.ndata`
    instead of reporting a difference. Returns whether the graph builders are usable.
    """
    try:
        import dgl  # noqa: F401
        have_dgl = True
    except ImportError:
        stub = types.ModuleType("dgl")
        stub.DGLGraph = object
        stub.graph = lambda *a, **k: None
        sys.modules["dgl"] = stub
        sys.modules["dgl.function"] = types.ModuleType("dgl.function")
        have_dgl = False
    sys.path.insert(0, str(repo))
    import commons.process_mols as upstream  # noqa: E402
    return upstream, have_dgl


def compare(name: str, a, b) -> bool:
    """Report one tensor pair, and say *how* they differ rather than just that they do."""
    a = np.asarray(a.detach().cpu() if isinstance(a, torch.Tensor) else a)
    b = np.asarray(b.detach().cpu() if isinstance(b, torch.Tensor) else b)
    if a.shape != b.shape:
        print(f"  {name:<26} SHAPE  upstream {a.shape} vs port {b.shape}")
        return False
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        same = np.allclose(a.astype(float), b.astype(float), atol=1e-6, equal_nan=True)
        delta = float(np.nanmax(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.0
        print(f"  {name:<26} {'OK   ' if same else 'DIFF '} max|Δ|={delta:.3e}  shape {a.shape}")
        return same
    same = bool((a == b).all())
    if same:
        print(f"  {name:<26} OK    exact match, shape {a.shape}")
        return True
    # for integer feature matrices, name the columns that moved — that is what identifies
    # a reordered or dropped feature, which a scalar mismatch count would not
    cols = [i for i in range(a.shape[1])] if a.ndim == 2 else []
    bad = [i for i in cols if not (a[:, i] == b[:, i]).all()]
    print(f"  {name:<26} DIFF  {int((a != b).sum())} cells; columns differing: {bad}")
    return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ligand", required=True)
    p.add_argument("--receptor", default=None, help="protein pdb; enables the receptor half")
    # Defaults are the production settings from embed_complexes.py, not the featuriser's own
    # signature defaults. Comparing at radius=20/cutoff=20 checks a graph the model never
    # actually sees; the neighbour selection is exactly where a port is most likely to differ.
    p.add_argument("--lig-radius", type=float, default=5)
    p.add_argument("--lig-max-neighbors", type=int, default=0, help="0 means None")
    p.add_argument("--rec-cutoff", type=float, default=30)
    p.add_argument("--rec-max-neighbor", type=int, default=10)
    p.add_argument("--upstream", default="/upstream")
    args = p.parse_args()

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    up, have_dgl = load_upstream(Path(args.upstream))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from equibind.features import featurize as port

    mol = None
    for sanitize in (True, False):
        supplier = Chem.SDMolSupplier(args.ligand, sanitize=sanitize, removeHs=False)
        mols = [m for m in supplier if m is not None]
        if mols:
            mol = mols[0]
            print(f"ligand: {args.ligand}  (sanitize={sanitize})  "
                  f"{mol.GetNumAtoms()} atoms, {mol.GetNumBonds()} bonds")
            break
    if mol is None:
        raise SystemExit("no readable ligand")

    ok = True
    print("\n--- allowable_features tables ---")
    up_keys, port_keys = set(up.allowable_features), set(port.allowable_features)
    if up_keys != port_keys:
        print(f"  key sets differ: only upstream {up_keys - port_keys}, "
              f"only port {port_keys - up_keys}")
        ok = False
    else:
        for k in sorted(up_keys):
            if up.allowable_features[k] != port.allowable_features[k]:
                print(f"  {k}: VALUES DIFFER")
                print(f"    upstream: {up.allowable_features[k]}")
                print(f"    port    : {port.allowable_features[k]}")
                ok = False
        print(f"  {len(up_keys)} tables, all identical" if ok else "")

    print("\n--- feature dimension tuples ---")
    for name in ("lig_feature_dims", "rec_atom_feature_dims", "rec_residue_feature_dims"):
        u, q = getattr(up, name, None), getattr(port, name, None)
        if u is None or q is None:
            print(f"  {name:<26} MISSING  upstream={u is not None} port={q is not None}")
            ok = False
        elif u != q:
            print(f"  {name:<26} DIFF  upstream {u} vs port {q}")
            ok = False
        else:
            print(f"  {name:<26} OK    {u}")

    print("\n--- ligand atom features on a real molecule ---")
    ok &= compare("lig_atom_featurizer", up.lig_atom_featurizer(mol),
                  port.lig_atom_featurizer(mol))

    print("\n--- ligand graph construction (DGL vs PyG) ---")
    if not have_dgl:
        print("  SKIPPED: no dgl in this image, so the authors' graph builders cannot run")
        print("\n" + ("FEATURES MATCH UPSTREAM (graphs not checked)" if ok
                      else "FEATURISATION DIVERGES"))
        return 0 if ok else 1
    # use_rdkit_coords=False on purpose: the rdkit conformer path is stochastic, and a
    # difference there would say nothing about the port.
    kw = dict(name="cmp", radius=args.lig_radius,
              max_neighbors=args.lig_max_neighbors or None, use_rdkit_coords=False)
    print(f"  settings: radius={kw['radius']} max_neighbors={kw['max_neighbors']}")
    g_up = up.get_lig_graph_revised(mol, **kw)
    g_pt = port.get_lig_graph_revised(mol, **kw)

    n_up, n_pt = g_up.num_nodes(), int(g_pt.num_nodes)
    print(f"  {'num_nodes':<26} {'OK   ' if n_up == n_pt else 'DIFF '} "
          f"upstream {n_up} vs port {n_pt}")
    ok &= n_up == n_pt

    src_up, dst_up = (t.long() for t in g_up.edges())
    src_pt, dst_pt = g_pt.edge_index[0], g_pt.edge_index[1]
    # Order first: DGL and PyG both take (src, dst), so a port that swapped them would still
    # produce the same *set* of edges on a symmetric graph and only differ in direction.
    ok &= compare("edge src (ordered)", src_up, src_pt)
    ok &= compare("edge dst (ordered)", dst_up, dst_pt)

    ok &= compare("node features", g_up.ndata["feat"], g_pt.x)
    ok &= compare("node coordinates", g_up.ndata["x"], g_pt.pos)
    ok &= compare("edge features", g_up.edata["feat"], g_pt.edge_attr)

    if args.receptor:
        print("\n--- receptor parsing (biopandas path, identical code both sides) ---")
        r_up = up.get_receptor_inference(args.receptor)
        r_pt = port.get_receptor_inference(args.receptor)
        # (rec, coords, c_alpha, n, c) — compare the coordinate arrays, which is what the
        # graph builder consumes; the Bio.PDB structure object itself is not comparable.
        for i, nm in enumerate(["rec_coords", "c_alpha_coords", "n_coords", "c_coords"], start=1):
            a, b = r_up[i], r_pt[i]
            if isinstance(a, list):
                a = np.concatenate(a) if len(a) and hasattr(a[0], "shape") else np.array(a)
                b = np.concatenate(b) if len(b) and hasattr(b[0], "shape") else np.array(b)
            ok &= compare(nm, a, b)

        print("\n--- receptor graph construction (DGL vs PyG) ---")
        gk = dict(cutoff=args.rec_cutoff, max_neighbor=args.rec_max_neighbor or None)
        print(f"  settings: cutoff={gk['cutoff']} max_neighbor={gk['max_neighbor']}")
        rg_up = up.get_calpha_graph(r_up[0], r_up[2], r_up[3], r_up[4], **gk)
        rg_pt = port.get_calpha_graph(r_pt[0], r_pt[2], r_pt[3], r_pt[4], **gk)

        n_up, n_pt = rg_up.num_nodes(), int(rg_pt.num_nodes)
        print(f"  {'num_nodes':<26} {'OK   ' if n_up == n_pt else 'DIFF '} "
              f"upstream {n_up} vs port {n_pt}")
        ok &= n_up == n_pt
        su, du = (t.long() for t in rg_up.edges())
        ok &= compare("edge src (ordered)", su, rg_pt.edge_index[0])
        ok &= compare("edge dst (ordered)", du, rg_pt.edge_index[1])
        ok &= compare("node features", rg_up.ndata["feat"], rg_pt.x)
        ok &= compare("node coordinates", rg_up.ndata["x"], rg_pt.pos)
        ok &= compare("edge features", rg_up.edata["feat"], rg_pt.edge_attr)
        ok &= compare("mu_r_norm", rg_up.ndata["mu_r_norm"], rg_pt.mu_r_norm)

    print("\n" + ("FEATURISATION MATCHES UPSTREAM" if ok else "FEATURISATION DIVERGES"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
