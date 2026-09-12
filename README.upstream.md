# equibind — the authors' original README

Kept for attribution and for the authors' own description of the method. **Its
instructions are not current for this fork** — see [README.md](README.md) for how to
build and run what is actually here.

---

# EquiBind Embedding Layer — PyTorch Geometric Port

A faithful PyTorch Geometric port of the **IEGMN encoder** from
[HannesStark/EquiBind](https://github.com/HannesStark/EquiBind)
([arXiv:2202.05146](https://arxiv.org/abs/2202.05146)).

The docking-specific **head** (keypoint attention + SVD/Kabsch alignment) is
deliberately omitted, exposing clean per-atom node embeddings so you can attach
any downstream task head you need.

> **License** — EquiBind is released under the MIT license.
> This port inherits the same license.

---

## What is EquiBind?

EquiBind is a structure-based drug-discovery model that predicts how a small
molecule (ligand) binds to a protein (receptor) **without any expensive
sampling or docking search**. It achieves this by jointly encoding both
molecules with an equivariant graph neural network and then solving for the
rigid-body transformation analytically.

### Architecture overview

```
Ligand atoms   ──► AtomEncoder ──► h_lig ─┐
                                           ├─ IEGMN_Layer × n_lays ──► embeddings + evolved coords
Receptor C-α   ──► AtomEncoder ──► h_rec ─┘
```

#### AtomEncoder

Each atom/residue feature vector is mapped to a fixed-size embedding by:

1. One `nn.Embedding(vocab_size, emb_dim)` per **categorical** feature  
   (atomic number, chirality, degree, hybridisation, …)
2. One `nn.Linear(n_scalars, emb_dim)` for **scalar** features  
   (Gasteiger charge, SASA, B-factor)

All contributions are **summed** (not concatenated), so the output is always
`emb_dim`-dimensional regardless of the number of features.

#### IEGMN_Layer (Iterative Equivariant Graph Matching Network)

Each layer performs four coupled updates on both molecular graphs simultaneously:

| Step | What happens |
|------|-------------|
| **Edge messages** | `m_{i→j} = MLP(h_i, h_j, e_{ij}, RBF(‖xᵢ-xⱼ‖²))` |
| **Cross-attention** | Ligand nodes attend softly to all receptor nodes (and vice versa) via Q/K/V MLPs |
| **Coordinate update** | `x_new = α·x_orig + (1-α)·x_now + mean_j[(xᵢ-xⱼ)·φˣ(m_{i→j})]` |
| **Node update** | `h_new = MLP(h_now ‖ aggr_msg ‖ cross_attn ‖ h_orig)` |

The coordinate update is **SE(3)-equivariant**: rotating / translating the input
coordinates rotates / translates the output by the same amount.

---

## Installation

### Requirements

- Python ≥ 3.11  
- PyTorch ≥ 2.3  
- PyTorch Geometric ≥ 2.6

### With `uv` (recommended)

```bash
# Install uv if you don't have it
curl -Ls https://astral.sh/uv/install.sh | sh

# Clone and install (CPU build)
git clone <this-repo>
cd equibind
uv sync

# CUDA build — override the torch index (example: CUDA 12.1)
uv sync --index-url https://download.pytorch.org/whl/cu121
```

Development extras (pytest, coverage):

```bash
uv sync --extra dev
```

### With pip

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
pip install rdkit biopython biopandas scipy
```

---

## Quick start

### 1 — Prepare graph data

The encoder expects `torch_geometric.data.Data` objects for ligand and receptor.
Use standard PyG graph construction or adapt the original EquiBind dataset code.

**Required node attributes**

| Graph | `.x` | `.pos` | `.edge_attr` | `.mu_r_norm` |
|-------|-------|--------|-------------|--------------|
| ligand | `(N, 17)` float — 16 categorical cols (int-valued) + 1 Gasteiger charge | `(N, 3)` — 3-D coords | `(E, 15)` — Gaussian RBF distances | `(N, 5)` — direction mean-norms |
| receptor | `(N, 3)` float — 1 AA-type col (int-valued) + SASA + B-factor | `(N, 3)` — C-α coords | `(E, 27)` — 15 RBF + 12 backbone-frame features | `(N, 5)` |

The `lig_feature_dims` and `rec_residue_feature_dims` constants exported from
`equibind.features.process_mols` document the exact vocabulary sizes for each
categorical column.

### 2 — Run the encoder

```python
import torch
from torch_geometric.data import Data
from equibind import EquiBindEncoder

encoder = EquiBindEncoder(
    n_lays=5,                    # number of IEGMN layers
    residue_emb_dim=64,          # node embedding dimension
    iegmn_lay_hid_dim=64,        # hidden dim in every layer
    cross_msgs=True,             # enable ligand ↔ receptor cross-attention
    use_mean_node_features=True, # prepend log(mu_r_norm) to node features
    device="cpu",
)

# lig_data / rec_data: torch_geometric.data.Data objects (see table above)
h_lig, h_rec, coords_lig, coords_rec, geom_loss = encoder(lig_data, rec_data)

# h_lig   : (N_lig, 64)  per-atom ligand embeddings
# h_rec   : (N_rec, 64)  per-residue receptor embeddings
# coords_lig : (N_lig, 3)  evolved ligand coordinates
# coords_rec : (N_rec, 3)  evolved receptor coordinates
```

### 3 — Attach your own head

```python
import torch.nn as nn

class MyBindingAffinity(nn.Module):
    def __init__(self, emb_dim=64):
        super().__init__()
        self.encoder = EquiBindEncoder(n_lays=5)
        # Global pooling + MLP head
        self.head = nn.Sequential(
            nn.Linear(emb_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, lig_data, rec_data):
        h_lig, h_rec, *_ = self.encoder(lig_data, rec_data)
        # Mean-pool each graph to a single vector
        z_lig = h_lig.mean(0)   # (emb_dim,)
        z_rec = h_rec.mean(0)   # (emb_dim,)
        return self.head(torch.cat([z_lig, z_rec]))
```

### 4 — Batched inference

`torch_geometric.data.Batch` is fully supported:

```python
from torch_geometric.data import Batch

ligs = Batch.from_data_list([lig1, lig2, lig3])
recs = Batch.from_data_list([rec1, rec2, rec3])

h_lig, h_rec, coords_lig, coords_rec, _ = encoder(ligs, recs)
# h_lig shape: (N_lig_total, 64)
```

---

## Feature dimensions reference

### Ligand atoms (`lig_feature_dims`)

| # | Feature | Vocabulary |
|---|---------|-----------|
| 0 | Atomic number | 119 |
| 1 | Chirality | 4 |
| 2 | Total degree | 12 |
| 3 | Formal charge | 12 |
| 4 | Implicit valence | 8 |
| 5 | Total H count | 10 |
| 6 | Radical electrons | 6 |
| 7 | Hybridisation | 6 |
| 8 | Is aromatic | 2 |
| 9 | Num rings | 8 |
| 10–15 | In ring of size 3/4/5/6/7/8 | 2 each |
| 16 | Gasteiger charge *(scalar)* | — |

Total feature vector width: **17** (16 categorical + 1 scalar).

### Receptor residues (`rec_residue_feature_dims`, default)

| # | Feature | Vocabulary |
|---|---------|-----------|
| 0 | Amino-acid type | 37 |
| 1 | SASA *(scalar)* | — |
| 2 | B-factor *(scalar)* | — |

Total width: **3**.

### Receptor atoms (`rec_atom_feature_dims`, when `use_rec_atoms=True`)

| # | Feature | Vocabulary |
|---|---------|-----------|
| 0 | Amino-acid type | 37 |
| 1 | Atomic number | 119 |
| 2 | Atom type (2-char) | 23 |
| 3 | Atom type (3-char) | 38 |
| 4 | SASA *(scalar)* | — |
| 5 | B-factor *(scalar)* | — |

Total width: **6**.

---

## Key constructor parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_lays` | `5` | Number of IEGMN layers |
| `residue_emb_dim` | `64` | Node embedding dimension after `AtomEncoder` |
| `iegmn_lay_hid_dim` | `64` | Hidden dimension in every IEGMN layer |
| `cross_msgs` | `True` | Enable ligand ↔ receptor cross-attention |
| `use_mean_node_features` | `True` | Append `log(mu_r_norm)` (5-dim directional features) |
| `use_rec_atoms` | `False` | Atom-level receptor graph instead of C-α |
| `lig_input_edge_feats_dim` | `15` | Ligand edge feature width (Gaussian RBF) |
| `rec_input_edge_feats_dim` | `27` | Receptor edge feature width (15 RBF + 12 backbone) |
| `use_dist_in_layers` | `True` | Append dynamic distance RBF inside each layer |
| `layer_norm` | `0` | Normalisation in MLPs: `0`=none, `'BN'`, `'LN'`, `'GN'` |
| `final_h_layer_norm` | `0` | Normalisation applied after each node update |
| `dropout` | `0.0` | Dropout rate inside all MLPs |
| `x_connection_init` | `0.25` | Weight for coordinate residual to initial position |
| `skip_weight_h` | `0.5` | Node feature skip-connection weight |
| `noise_initial` | `0.0` | Gaussian noise std added to features/coords during training |
| `noise_decay_rate` | `1.0` | Noise std ← `noise_initial × decay_rate^(epoch+1)` |
| `geometry_regularization` | `False` | Enforce bond-geometry constraints via gradient steps |
| `shared_layers` | `False` | Share weights across all IEGMN layers |
| `device` | `'cpu'` | Torch device string |

---

## Running the tests

```bash
# Install dev deps then run
uv sync --extra dev
uv run pytest

# With coverage
uv run pytest --cov=equibind --cov-report=term-missing
```

The test suite covers:

| File | What is tested |
|------|----------------|
| `test_features.py` | `AtomEncoder` output shapes, truncation (`n_feats_to_use`), disabled scalars, gradient flow |
| `test_layers.py` | `_scatter_mean` edge cases, `GraphNorm`, `CoordsNorm`, `get_mask` block-diagonal structure, `cross_attention`, `IEGMN_Layer` (single & batched, no-evolve, first-layer dim change) |
| `test_encoder.py` | `EquiBindEncoder` single and batched forward, single-vs-batch consistency, atom-level receptor, no-cross-msgs, gradient flow through encoder, geometry regularisation loss |

---

## Project structure

```
equibind/
├── __init__.py                   # Public API
├── features/
│   └── process_mols.py           # AtomEncoder + feature dim constants
└── models/
    ├── layers.py                 # GraphNorm, CoordsNorm, IEGMN_Layer + helpers
    └── encoder.py                # EquiBindEncoder (backbone without docking head)

tests/
├── conftest.py                   # Shared synthetic Data/Batch fixtures
├── test_features.py
├── test_layers.py
└── test_encoder.py

pyproject.toml                    # uv-compatible project manifest
```

---

## Differences from the original

| Aspect | Original EquiBind | This port |
|--------|-------------------|-----------|
| Graph library | DGL | PyTorch Geometric |
| Graph input | `dgl.DGLGraph` | `torch_geometric.data.Data` / `Batch` |
| Message passing | `apply_edges` + `update_all` | `scatter_add_` (native PyTorch) |
| Batch normalisation | `GraphNorm` via DGL local scope | `GraphNorm` via PyG `unbatch` |
| Cross-attention mask | Built from `batch_num_nodes()` | Built from `.batch` index tensor |
| Docking head | Included (keypoints + SVD) | **Removed** — attach your own |
| Package manager | conda + pip | uv + pyproject.toml |

All mathematical operations are numerically equivalent to the original.

---

## Citation

```bibtex
@article{stark2022equibind,
  title   = {EquiBind: Geometric Deep Learning for Drug Binding Structure Prediction},
  author  = {Hannes St{\"a}rk and Octavian-Eugen Ganea and Lagnajit Pattanaik
             and Regina Barzilay and Tommi Jaakkola},
  journal = {arXiv preprint arXiv:2202.05146},
  year    = {2022}
}
```
