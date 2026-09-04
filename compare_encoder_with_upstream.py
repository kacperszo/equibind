"""Compare this port's encoder against the authors' IEGMN, on identical inputs and weights.

`compare_with_upstream.py` already showed the inputs are bit-identical: feature tables, atom
features, and both graphs (nodes, edge order, edge features, mu_r_norm) match exactly. So if
the port diverges, it diverges in the arithmetic — which is what this checks.

Two things make the comparison possible at all:

  * the port kept the authors' parameter names. `embed_complexes.py` loads the published
    checkpoint by stripping the `iegmn.` prefix and nothing else, and asserts that nothing is
    missing. So the same tensors can be loaded into both implementations.
  * the authors' `IEGMN.forward` returns docking outputs (rotations, keypoints, evolved
    coordinates), not the representations this port's encoder returns. A forward hook on the
    last `iegmn_layers` entry captures `(x_evolved_lig, node_upd_ligand, x_evolved_rec,
    node_upd_receptor, ...)` — exactly what the port's encoder returns.

**Noise is forced off.** `IEGMN.forward` adds `noise_level * randn_like(...)` to features and
coordinates whenever `noise_initial > 0`, with no `self.training` guard, so the published
settings make inference stochastic (level 0.5 at epoch 0). The port reproduces that faithfully.
Comparing with it on would compare two different random draws and say nothing about the port,
so `noise_initial` is set to 0 here and both sides are seeded anyway.

usage (in the image holding both dgl and PyG):
    python compare_encoder_with_upstream.py --complex-dir /data/1bcu --id 1bcu
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def build_graphs(port, cid: str, complex_dir: str, lig_radius: float, rec_cutoff: float,
                 rec_max_neighbor: int):
    """Graphs at the settings embed_complexes.py actually uses, built once by the port."""
    from equibind.features.featurize import read_molecule

    ligand = read_molecule(f"{complex_dir}/{cid}_ligand.mol2", sanitize=True, remove_hs=True)
    if ligand is None:
        ligand = read_molecule(f"{complex_dir}/{cid}_ligand.sdf", sanitize=True, remove_hs=True)
    if ligand is None:
        raise SystemExit("RDKit could not read the ligand")
    lig = port.get_lig_graph_revised(ligand, cid, radius=lig_radius, max_neighbors=None,
                                     use_rdkit_coords=False)
    rec, _c, c_alpha, n, c = port.get_receptor_inference(f"{complex_dir}/{cid}_protein.pdb")
    recg = port.get_calpha_graph(rec, c_alpha, n, c, cutoff=rec_cutoff,
                                 max_neighbor=rec_max_neighbor)
    return lig, recg


def to_dgl(pyg_graph, is_ligand: bool):
    """Rebuild the same graph as DGL, so both implementations see identical tensors.

    Deliberately converted rather than built twice: `compare_with_upstream.py` established the
    two builders agree, and converting removes any doubt that a difference here is an input
    difference rather than an arithmetic one.
    """
    import dgl

    src, dst = pyg_graph.edge_index
    g = dgl.graph((src.int(), dst.int()), num_nodes=int(pyg_graph.num_nodes),
                  idtype=torch.int32)
    g.ndata["feat"] = pyg_graph.x
    g.ndata["x"] = pyg_graph.pos
    g.edata["feat"] = pyg_graph.edge_attr
    if hasattr(pyg_graph, "mu_r_norm"):
        g.ndata["mu_r_norm"] = pyg_graph.mu_r_norm
    if is_ligand:
        # IEGMN.forward reads ndata['new_x'] unconditionally; with use_rdkit_coords=False the
        # builder never sets it, and the authors' inference path seeds it from the true coords
        g.ndata["new_x"] = pyg_graph.pos
    return g


def alias_old_dgl_functions() -> list[str]:
    """Restore the DGL 0.x message-function names the authors' code calls.

    `dgl.function.copy_edge` and `copy_src` were renamed to `copy_e` and `copy_u` in DGL 1.0.
    These are renames, not behaviour changes, so aliasing them lets the original run unmodified
    under DGL 2.1 — which is the point: the comparison must exercise the authors' code, not a
    version of it we edited.
    """
    import dgl.function as fn

    restored = []
    for old_name, new_name in (("copy_edge", "copy_e"), ("copy_src", "copy_u")):
        if not hasattr(fn, old_name) and hasattr(fn, new_name):
            setattr(fn, old_name, getattr(fn, new_name))
            restored.append(f"{old_name}->{new_name}")
    return restored


def report(name: str, a: torch.Tensor, b: torch.Tensor) -> bool:
    a, b = a.detach().cpu().double(), b.detach().cpu().double()
    if a.shape != b.shape:
        print(f"  {name:<22} SHAPE  upstream {tuple(a.shape)} vs port {tuple(b.shape)}")
        return False
    d = float((a - b).abs().max())
    scale = max(float(a.abs().max()), 1e-12)
    ok = d / scale < 1e-5
    print(f"  {name:<22} {'OK  ' if ok else 'DIFF'}  max|Δ|={d:.3e}  "
          f"relative={d / scale:.2e}  shape {tuple(a.shape)}")
    return ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--complex-dir", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--checkpoint", default="checkpoints/best_checkpoint.pt")
    p.add_argument("--args", default="checkpoints/train_arguments.yaml")
    p.add_argument("--upstream", default="/upstream")
    p.add_argument("--lig-radius", type=float, default=5)
    p.add_argument("--rec-cutoff", type=float, default=30)
    p.add_argument("--rec-max-neighbor", type=int, default=10)
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    sys.path.insert(0, args.upstream)

    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    params = yaml.safe_load(Path(args.args).read_text())["model_parameters"]
    params["noise_initial"] = 0          # see module docstring
    params["debug"] = False
    # Geometry regularization is disabled here because no geometry graph is passed, and
    # upstream dereferences it unconditionally while the port guards on None. That guard is
    # itself a finding — `embed_complexes.py` calls the encoder without a geometry graph, so
    # production embeddings skip a correction the trained model had enabled, and the port
    # never ported `get_geometry_graph`. Isolating the arithmetic means switching it off on
    # both sides; comparing it on requires porting that builder first.
    params["geometry_regularization"] = False
    params["loss_geometry_regularization"] = False
    print(f"config: n_lays={params['n_lays']} residue_emb_dim={params['residue_emb_dim']} "
          f"num_att_heads={params['num_att_heads']} noise_initial=0 (forced)")

    import equibind.features.featurize as port_feat
    from equibind.models.encoder import EquiBindEncoder
    from equibind.models.layers import IEGMN_Layer as PortLayer
    from models.equibind import IEGMN, IEGMN_Layer  # noqa: E402  (upstream)

    def filtered(cls, extra=None, inner_classes=()):
        """Keep only what the constructor accepts — unless it takes **kwargs.

        Upstream's IEGMN forwards several arguments to IEGMN_Layer through **kwargs rather
        than naming them, so filtering on the explicit signature alone drops them and the
        layer constructor fails on missing positionals.
        """
        allowed = set(inspect.signature(cls.__init__).parameters)
        for inner in inner_classes or ():
            # IEGMN names some arguments only in IEGMN_Layer and forwards them via **kwargs,
            # so the union of both signatures is what it really accepts. Passing everything
            # instead fails the other way: IEGMN_Layer rejects IEGMN-only keys.
            allowed |= set(inspect.signature(inner.__init__).parameters)
        out = {k: v for k, v in params.items() if k in allowed}
        out.update(extra or {})
        return out

    # Graphs first: the edge feature widths are properties of the data, not of the config,
    # and both constructors need them.
    lig_pyg, rec_pyg = build_graphs(port_feat, args.id, args.complex_dir,
                                    args.lig_radius, args.rec_cutoff, args.rec_max_neighbor)
    print(f"graphs: ligand {int(lig_pyg.num_nodes)} nodes / {lig_pyg.edge_index.shape[1]} edges, "
          f"receptor {int(rec_pyg.num_nodes)} nodes / {rec_pyg.edge_index.shape[1]} edges")
    dims = dict(device="cpu",
                lig_input_edge_feats_dim=int(lig_pyg.edge_attr.shape[1]),
                rec_input_edge_feats_dim=int(rec_pyg.edge_attr.shape[1]))
    print(f"edge feature dims: ligand {dims['lig_input_edge_feats_dim']}, "
          f"receptor {dims['rec_input_edge_feats_dim']}")

    torch.manual_seed(0)
    up_model = IEGMN(**filtered(IEGMN, dims, inner_classes=(IEGMN_Layer,)))
    torch.manual_seed(0)
    pt_model = EquiBindEncoder(**filtered(EquiBindEncoder, dims,
                                          inner_classes=(PortLayer,)))

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = ck.get("model_state_dict", ck)
    encoder_state = {k[len("iegmn."):]: v for k, v in state.items() if k.startswith("iegmn.")}
    for name, model in (("upstream", up_model), ("port", pt_model)):
        missing, unexpected = model.load_state_dict(encoder_state, strict=False)
        real_missing = [m for m in missing if "num_batches_tracked" not in m]
        print(f"  {name:<8} loaded {len(encoder_state)} tensors, "
              f"{len(real_missing)} missing, {len(unexpected)} unexpected")
        if real_missing:
            print(f"    missing: {real_missing[:5]}")
    up_model.eval(), pt_model.eval()

    lig_dgl, rec_dgl = to_dgl(lig_pyg, True), to_dgl(rec_pyg, False)

    restored = alias_old_dgl_functions()
    if restored:
        print(f"dgl compatibility aliases: {', '.join(restored)}")

    captured: list = []
    up_model.iegmn_layers[-1].register_forward_hook(
        lambda _m, _i, out: captured.append(out))

    with torch.no_grad():
        torch.manual_seed(0)
        up_model(lig_dgl, rec_dgl, None, [args.id], 0)
        torch.manual_seed(0)
        h_lig, h_rec, coords_lig, coords_rec, _g = pt_model(lig_pyg, rec_pyg, None, 0)

    if not captured:
        raise SystemExit("the hook captured nothing — did IEGMN's layer list change name?")
    x_lig, h_lig_up, x_rec, h_rec_up = captured[-1][:4]

    print("\n--- encoder outputs after the final IEGMN layer ---")
    ok = True
    ok &= report("h_feats_lig", h_lig_up, h_lig)
    ok &= report("h_feats_rec", h_rec_up, h_rec)
    ok &= report("coords_lig", x_lig, coords_lig)
    ok &= report("coords_rec", x_rec, coords_rec)

    print("\n" + ("ENCODER MATCHES UPSTREAM" if ok else "ENCODER DIVERGES"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
