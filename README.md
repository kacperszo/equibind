# EquiBind — an E(3)-equivariant encoder trained to place a ligand

## Upstream

This is a PyTorch Geometric port of the IEGMN encoder and docking head from
[HannesStark/EquiBind](https://github.com/HannesStark/EquiBind) (Stärk et al.,
[arXiv:2202.05146](https://arxiv.org/abs/2202.05146)), which is released under the MIT licence.
The code here is a rewrite, not a patched copy of theirs. The authors' published weights load into
it unchanged; see [Weights](#weights).

[README.upstream.md](README.upstream.md) is the README this port had at an earlier stage. It is
**not** the authors' README, which lives in their repository.

## What it is

A PyTorch Geometric reimplementation of the IEGMN encoder: eight equivariant layers
over separate ligand and receptor graphs with cross-messages between them, evolving coordinates as
well as features. The head solves for a rigid transform by keypoint attention plus a Kabsch fit. Not a
patched copy but a rewrite, which is why it carries 52 tests (`pytest tests/`) and no DGL — and the authors' published weights
load into it unchanged, 597/597 parameters.

EquiBind has no affinity head, which is what makes its encoder interesting to transfer from: a
representation trained purely on geometry.

## State

| | |
|---|---|
| docking | **11.7 A median** on 285/285 CASF-2016 core-set complexes, beating the do-nothing baseline on 96%. The paper reports a ~6.2 A median on its own test split, so this is about twice as far off; CASF-2016 is a different distribution, but that does not explain all of it (see Porting notes) |
| embedding | 128d, **ours** (max-pooled), linear probe R 0.577 on the core set — clears the 0.470 that counting heavy atoms and bonds reaches |
| regression check | against stored reference outputs: dock 3.6e-15, embed **0.0** |
| equivariance | encoder 3.8e-05; the docking head 0.15 A, from one line in the Kabsch stabiliser |
| training | `train` and `finetune`, with an affinity head that is ours, not theirs |

## Build

```bash
podman build --format=docker -t equibind:latest .
```

## Weights

The authors' `best_checkpoint.pt` and its `train_arguments.yaml` are not in this repository and
are not redistributed here (`checkpoints/` is gitignored). They come from the authors' published
release in [HannesStark/EquiBind](https://github.com/HannesStark/EquiBind); this repository does
not record an exact download location, so follow the instructions there. Put both files in
`checkpoints/`. Treat any downloaded checkpoint as pickle: it is code execution unless loaded
with `torch.load(..., weights_only=True)`, which `embed_complexes.py` and `dock_complexes.py` do.

## Run it

The commands below run with `--network=none` and a read-only root filesystem. `embed` and `dock`
need the weights from the previous section; `train` needs none.

Input is one directory per complex:

    <complexes>/<id>/<id>_protein.pdb
    <complexes>/<id>/<id>_ligand.sdf      # or .mol2; several models try both

Reads the ligand with RDKit and generates its own conformer for docking — the crystallographic pose is the *answer*, so docking must not be given it. Embedding does use the crystal pose, deliberately.

```bash
# embed
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v "$PWD/checkpoints:/ckpt:ro" \
    localhost/equibind:latest \
    python embed_complexes.py --complexes /data --model /ckpt/best_checkpoint.pt --out /outputs/embeddings.npz --pool max --device cpu

# dock
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v "$PWD/checkpoints:/ckpt:ro" \
    localhost/equibind:latest \
    python dock_complexes.py --complexes /data --model /ckpt/best_checkpoint.pt --out /outputs/predictions.csv --device cpu

# train
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v /path/to/splits:/splits:ro \
    -v /path/to/cache:/cache:rw,U \
    --shm-size 4g \
    localhost/equibind:latest \
    python train_affinity.py --complexes /data --labels /splits/train.csv --out /outputs --pool max --seed 0 --device cpu --val-labels /splits/val.csv --cache /cache --epochs 30

# finetune  (encoder frozen; drop --freeze-encoder to tune all of it)
# encoders/encoder.pt: the authors' checkpoint cut down to encoder tensors only (see below)
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v "$PWD/encoders:/ckpt:ro" \
    -v /path/to/splits:/splits:ro \
    -v /path/to/cache:/cache:rw,U \
    --shm-size 4g \
    localhost/equibind:latest \
    python train_affinity.py --complexes /data --labels /splits/train.csv --out /outputs --pool max --seed 0 --device cpu --val-labels /splits/val.csv --cache /cache --epochs 30 --init-encoder /ckpt/encoder.pt --freeze-encoder
```

`encoder.pt` is the authors' state dict with the docking-head tensors removed (the prefixes are
listed as `DOCKING_HEAD_PREFIXES` in `transfer.py`); the `iegmn.` prefix may be kept or stripped,
both load. `transfer.py` refuses a file that still contains docking-head tensors.

## What comes out

| file | holds |
|---|---|
| `predictions.csv` | docking: `complex_id,rmsd,rmsd_conformer,rmsd_aligned` — the prediction, the do-nothing baseline, and the best a rigid method could manage. A bare RMSD says nothing without the other two |
| `embeddings.npz` | `ids` and `vectors`, 128-dim — pooled ligand and receptor node embeddings concatenated. Ours, not the authors' |
| `model.pt` | training only; the IEGMN plus the affinity head from `equibind/models/affinity.py` |

## Before you trust the numbers

**Published inference is not deterministic.** `noise_initial > 0` adds `randn_like` to features and coordinates inside `forward` with no `self.training` guard, so the authors' own settings make a single RMSD a sample rather than a measurement — roughly 0.3-0.5 Å of irreducible spread, and the Kabsch fit perturbs a near-degenerate covariance on top. Our scripts turn it off by default and take `--noise` to restore it.

**The affinity head is ours.** EquiBind has none: it was trained to place a ligand, not to score one, which is what makes it the interesting encoder here and why a fine-tune needs the random-init control run beside it.

## Porting notes

What breaks if this port is changed, or if you reuse the model with other settings.

**Four config parameters carry no weights, so a wrong value loads cleanly and silently changes
what the encoder computes.** The authors' `train_arguments.yaml` has `use_scalar_features: False`,
`rec_square_distance_scale: 10`, and geometry regularisation on with step 0.001. With this port's
old defaults, receptor coordinates grew from 83 A to 2400 A across the 8 layers and RMSD read ~250
A. `use_scalar_features` changes feature *content* without changing any shape, so a 597/597
parameter match proves nothing. Read `train_arguments.yaml` before concluding anything about
compatibility. Built with arbitrary defaults this port matched 171/597 parameters; with the
authors' configuration it matches 597/597. The encoder is built with their `runs/flexible_self_docking`
configuration:

```python
EquiBindEncoder(
    n_lays=8, layer_norm="BN", layer_norm_coords=0, final_h_layer_norm=0,
    dropout=0.1, use_rec_atoms=False, cross_msgs=True, shared_layers=False,
    noise_decay_rate=0.5, noise_initial=1, residue_emb_dim=64,
    standard_norm_order=True, nonlin="lkyrelu", leakyrelu_neg_slope=0.01,
    use_dist_in_layers=True, normalize_coordinate_update=True,
)
```

Their state dict prefixes everything with `iegmn.`; strip it and the names line up exactly.

**Conformer generation fails on hydrogen-stripped ligands** under current RDKit, and the authors'
fallback only catches an embed returning -1, not one that throws. It then falls back to the
crystal pose and hands the model the answer. Embed with explicit hydrogens and remove them
afterwards.

**The docking input must be randomly placed** (authors' `inference.py:366`): centre at the origin,
rotate, translate up to 5 A. Feeding the aligned conformer is off-distribution and the model
moves it away. The predicted transform applies to the *evolved* ligand coordinates, not the input
ones (authors' `models/equibind.py:1035`, `use_evolved_lig: true`).

**The residue vocabulary is 38 classes, not 37.** The authors' `possible_amino_acids` is the 20
standard residues plus 17 non-standard ones (HIP, HIE, TPO, HID, LEV, MEU, PTR, GLV, CYT, SEP,
HIZ, CYM, GLM, ASQ, TYS, CYX, GLZ) and a trailing `misc`. Off by one shifts every residue index:
the model runs and encodes every residue as the wrong one. Whatever produces the features must use
the authors' list in their order.

**`standard_norm_order` decides where BatchNorm sits inside the MLPs.** The authors trained with
`true` (Linear, Norm, NonLin, Dropout, Linear, Norm). The other ordering leaves 160 parameter
names unmatched, which looks like a broken port rather than a wrong flag.

**Docking accuracy.** 11.7 A median on the CASF-2016 core set, 4.6% within 5 A, against the
paper's ~6.2 A median and ~20% within 5 A. Not explained by the starting position (correlation
with final RMSD 0.14). Settling it needs the authors' original run side by side.

**Inference is not deterministic by the authors' design.** Two causes, both reproduced faithfully
and switched off by default here:

- `noise_initial * noise_decay_rate ** (epoch + 1)` adds noise to features and coordinates inside
  `forward` with no `self.training` guard: N(0, 0.5) at inference at the authors' settings.
  `--noise` restores it.
- the Kabsch fit perturbs its covariance with `torch.rand` until well conditioned, and on real
  complexes the keypoints are near-coplanar so it fires every time. Because the perturbation is
  diagonal in the world frame it breaks equivariance, measured at 0.15 A by
  `test_head_equivariance.py`. A scalar multiple of the identity conditions the SVD identically
  and commutes with rotation (1.1e-05 A), but is not applied because the weights were trained with
  the random version. The encoder alone is equivariant to float precision (4e-05,
  `test_equivariance.py`).

**The embedding pooling is ours.** `max` was chosen by probe over the full core set:

| pool | Pearson | Spearman | c-index | sensitivity to 5e-4 A of coordinate noise |
|---|---|---|---|---|
| sum | 0.533 | 0.582 | 0.692 | 6.05% |
| mean | 0.544 | 0.545 | 0.671 | 1.93% |
| **max** | **0.577** | **0.591** | **0.693** | **0.81%** |

`max` wins on every metric and is the most stable; `sum` also encodes molecule size, which is the
trivial baseline (0.470). A geometry-only representation therefore does carry affinity linearly, so
read the readout before concluding anything about the representation. Embeddings are extracted
from the crystallographic pose, not the RDKit conformer the model was trained on, so they are
off-distribution; no randomly initialised encoder has been probed to separate what is learned
from what is architectural.

**Three helper functions were called but never ported** at one stage; an AST check for
called-but-undefined names guards that now.
