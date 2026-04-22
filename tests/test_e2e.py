"""End-to-end test: embed a real single complex.

Builds a ligand graph from aspirin (RDKit, SMILES → 3-D conformer → PyG Data)
paired with a small synthetic receptor stub, then runs the full encoder pipeline
and checks output shapes and numerical validity.

Skipped automatically when rdkit is not installed.
"""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data

from equibind import EquiBindEncoder

# ---------------------------------------------------------------------------
# Feature-mapping helpers (matching EquiBind / process_mols conventions)
# ---------------------------------------------------------------------------

_CHIRAL = None   # populated lazily after rdkit import
_HYBRID = None


def _init_enums():
    global _CHIRAL, _HYBRID
    from rdkit.Chem import rdchem
    _CHIRAL = [
        rdchem.ChiralType.CHI_UNSPECIFIED,
        rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        rdchem.ChiralType.CHI_OTHER,
    ]
    _HYBRID = [
        rdchem.HybridizationType.SP,
        rdchem.HybridizationType.SP2,
        rdchem.HybridizationType.SP3,
        rdchem.HybridizationType.SP3D,
        rdchem.HybridizationType.SP3D2,
        rdchem.HybridizationType.OTHER,
    ]


def _safe_idx(lst, val):
    try:
        return lst.index(val)
    except ValueError:
        return len(lst) - 1  # clamp unknowns to last bucket


def _atom_features(atom, ring_info) -> list:
    """Return 17-element feature row for one atom (16 categorical + 1 scalar).

    ring_info must be mol.GetRingInfo() — RDKit ≥2022 removed GetRingInfo() from Atom.
    """
    idx     = atom.GetIdx()
    anum    = min(atom.GetAtomicNum(), 118)              # vocab 119 → idx 0..118
    chir    = _safe_idx(_CHIRAL, atom.GetChiralTag())    # vocab 4
    deg     = min(atom.GetTotalDegree(), 11)             # vocab 12
    fcharge = min(max(atom.GetFormalCharge() + 5, 0), 11)  # shift → 0..11, vocab 12
    ival    = min(atom.GetTotalValence() - atom.GetNumExplicitHs(), 7)  # vocab 8
    th      = min(atom.GetTotalNumHs(), 9)               # vocab 10
    rad     = min(atom.GetNumRadicalElectrons(), 5)      # vocab 6
    hyb     = _safe_idx(_HYBRID, atom.GetHybridization())  # vocab 6
    arom    = int(atom.GetIsAromatic())                  # vocab 2
    nrings  = min(ring_info.NumAtomRings(idx), 7)        # vocab 8
    r3, r4, r5, r6, r7, r8 = (int(ring_info.IsAtomInRingOfSize(idx, s))
                               for s in (3, 4, 5, 6, 7, 8))  # vocab 2 each

    gasteiger = 0.0
    if atom.HasProp("_GasteigerCharge"):
        g = atom.GetDoubleProp("_GasteigerCharge")
        gasteiger = float(g) if np.isfinite(g) else 0.0

    return [anum, chir, deg, fcharge, ival, th, rad, hyb, arom, nrings,
            r3, r4, r5, r6, r7, r8, gasteiger]


def _gaussian_rbf(dists: torch.Tensor, n: int = 15, max_d: float = 6.0) -> torch.Tensor:
    """(E,) distances → (E, n) Gaussian RBF features."""
    centers = torch.linspace(0.0, max_d, n)
    gamma = (max_d / n) ** -2
    return torch.exp(-gamma * (dists.unsqueeze(-1) - centers) ** 2)


def _mu_r_norm(pos: torch.Tensor, n_shells: int = 5) -> torch.Tensor:
    """(N, 5) strictly-positive directional density features.

    For each atom and each of 5 distance shells, count how many other atoms
    fall within that shell (plus 1 to stay positive). Log is taken by the
    encoder, so values must be > 0.
    """
    n = pos.size(0)
    dists = torch.cdist(pos, pos)                    # (N, N)
    dists.fill_diagonal_(float("inf"))

    finite = dists[dists.isfinite()]
    result = torch.ones(n, n_shells)
    if finite.numel() > 0:
        lo, hi = finite.min().item(), finite.max().item() + 1e-6
        edges = torch.linspace(lo, hi, n_shells + 1)
        for k in range(n_shells):
            in_shell = ((dists >= edges[k]) & (dists < edges[k + 1])).float()
            result[:, k] = in_shell.sum(dim=1) + 1.0   # strictly > 0
    return result


def mol_to_lig_data(mol) -> Data:
    """Convert an RDKit mol (with 3-D conformer + Gasteiger charges) to PyG Data."""
    conf = mol.GetConformer()
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
    n = mol.GetNumAtoms()

    ring_info = mol.GetRingInfo()
    feat_rows = [_atom_features(a, ring_info) for a in mol.GetAtoms()]
    x = torch.tensor(feat_rows, dtype=torch.float32)  # (N, 17)

    # Bidirectional bond edges
    bonds = mol.GetBonds()
    src_list, dst_list = [], []
    for b in bonds:
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        src_list += [i, j]
        dst_list += [j, i]

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    dists = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)
    edge_attr = _gaussian_rbf(dists, n=15)             # (E, 15)

    mu_r_norm = _mu_r_norm(pos)                        # (N, 5), strictly > 0

    return Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr,
                mu_r_norm=mu_r_norm)


def _make_rec_stub(n: int = 12, e: int = 36, seed: int = 42) -> Data:
    """Synthetic receptor residue graph with valid feature ranges."""
    g = torch.Generator().manual_seed(seed)
    # 1 categorical (AA type 0..36) + 2 scalars (SASA, B-factor)
    aa   = torch.randint(0, 37, (n, 1), generator=g).float()
    sasa = torch.rand(n, 1, generator=g) * 150.0          # Å² — realistic range
    bfac = torch.rand(n, 1, generator=g) * 50.0           # B-factor — realistic range
    x = torch.cat([aa, sasa, bfac], dim=1)                # (n, 3)

    pos  = torch.randn(n, 3, generator=g) * 5.0           # scattered Cα positions
    edge_index = torch.randint(0, n, (2, e), generator=g)
    dists = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)
    rbf   = _gaussian_rbf(dists, n=15)
    # Receptor edge features: 15 RBF + 12 backbone orientation features (zeros ok)
    edge_attr = torch.cat([rbf, torch.zeros(e, 12)], dim=1)   # (e, 27)

    mu_r_norm = _mu_r_norm(pos)                               # (n, 5)
    return Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr,
                mu_r_norm=mu_r_norm)


# ---------------------------------------------------------------------------
# Fixtures / skip guard
# ---------------------------------------------------------------------------

def _rdkit_available() -> bool:
    try:
        import rdkit  # noqa: F401
        return True
    except ImportError:
        return False


requires_rdkit = pytest.mark.skipif(
    not _rdkit_available(), reason="rdkit not installed"
)


@pytest.fixture(scope="module")
def aspirin_mol():
    from rdkit import Chem
    from rdkit.Chem import AllChem
    _init_enums()
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    mol = Chem.RemoveHs(mol)
    AllChem.ComputeGasteigerCharges(mol)
    return mol


@pytest.fixture(scope="module")
def aspirin_lig(aspirin_mol):
    return mol_to_lig_data(aspirin_mol)


@pytest.fixture(scope="module")
def receptor_stub():
    return _make_rec_stub()


@pytest.fixture(scope="module")
def fast_encoder():
    return EquiBindEncoder(
        n_lays=2, device="cpu",
        layer_norm=0, layer_norm_coords=0, final_h_layer_norm=0,
        dropout=0.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@requires_rdkit
class TestE2ESingleComplex:
    def test_ligand_graph_feature_shape(self, aspirin_mol, aspirin_lig):
        n = aspirin_mol.GetNumAtoms()
        assert aspirin_lig.x.shape == (n, 17), \
            f"Expected ({n}, 17), got {aspirin_lig.x.shape}"
        assert aspirin_lig.pos.shape == (n, 3)
        assert aspirin_lig.edge_attr.shape[1] == 15
        assert aspirin_lig.mu_r_norm.shape == (n, 5)
        # All categorical cols must be within their vocab bounds
        from equibind.features.process_mols import lig_feature_dims
        cats, _ = lig_feature_dims
        for col, vocab in enumerate(cats):
            vals = aspirin_lig.x[:, col].long()
            assert vals.min() >= 0, f"col {col}: negative index"
            assert vals.max() < vocab, \
                f"col {col}: index {vals.max().item()} >= vocab {vocab}"

    def test_mu_r_norm_positive(self, aspirin_lig):
        assert (aspirin_lig.mu_r_norm > 0).all(), "mu_r_norm must be strictly positive (log is taken)"

    def test_encoder_output_shapes(self, aspirin_mol, aspirin_lig, receptor_stub, fast_encoder):
        n_lig = aspirin_mol.GetNumAtoms()
        n_rec = receptor_stub.x.size(0)

        fast_encoder.eval()
        with torch.no_grad():
            h_lig, h_rec, c_lig, c_rec, _ = fast_encoder(aspirin_lig, receptor_stub)

        assert h_lig.shape == (n_lig, 64), f"h_lig: {h_lig.shape}"
        assert h_rec.shape == (n_rec, 64), f"h_rec: {h_rec.shape}"
        assert c_lig.shape == (n_lig, 3),  f"c_lig: {c_lig.shape}"
        assert c_rec.shape == (n_rec, 3),  f"c_rec: {c_rec.shape}"

    def test_no_nan_or_inf(self, aspirin_lig, receptor_stub, fast_encoder):
        fast_encoder.eval()
        with torch.no_grad():
            outputs = fast_encoder(aspirin_lig, receptor_stub)
        for name, t in zip(("h_lig", "h_rec", "c_lig", "c_rec"), outputs[:4]):
            assert not t.isnan().any(), f"NaN in {name}"
            assert not t.isinf().any(), f"Inf in {name}"

    def test_embeddings_not_all_zero(self, aspirin_lig, receptor_stub, fast_encoder):
        fast_encoder.eval()
        with torch.no_grad():
            h_lig, h_rec, *_ = fast_encoder(aspirin_lig, receptor_stub)
        assert h_lig.abs().sum() > 0, "ligand embeddings are all zero"
        assert h_rec.abs().sum() > 0, "receptor embeddings are all zero"

    def test_coords_evolved(self, aspirin_lig, receptor_stub, fast_encoder):
        """Evolved coords must differ from the initial coords (network should move them)."""
        fast_encoder.eval()
        with torch.no_grad():
            _, _, c_lig, _, _ = fast_encoder(aspirin_lig, receptor_stub)
        assert not torch.allclose(c_lig, aspirin_lig.pos), \
            "Encoder left coordinates unchanged"

    def test_gradient_through_real_ligand(self, aspirin_lig, receptor_stub):
        enc = EquiBindEncoder(
            n_lays=2, device="cpu",
            layer_norm=0, layer_norm_coords=0, final_h_layer_norm=0,
        )
        pos = aspirin_lig.pos.clone().requires_grad_(True)
        lig = Data(x=aspirin_lig.x, pos=pos,
                   edge_index=aspirin_lig.edge_index,
                   edge_attr=aspirin_lig.edge_attr,
                   mu_r_norm=aspirin_lig.mu_r_norm)
        h_lig, *_ = enc(lig, receptor_stub)
        h_lig.sum().backward()
        assert pos.grad is not None, "No gradient w.r.t. input coordinates"
        assert pos.grad.abs().sum() > 0
