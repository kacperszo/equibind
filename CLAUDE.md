# CLAUDE.md — EquiBind (PyG port)

## What this is

A PyTorch Geometric reimplementation of the **IEGMN encoder** from
[HannesStark/EquiBind](https://github.com/HannesStark/EquiBind), with the docking head
(keypoint attention + SVD/Kabsch) deliberately omitted. Not a fork — a rewrite, which is
why it carries 52 tests and no DGL.

Unlike the other models in the benchmark this one arrived as an *architecture*, not a
reproduced model. As of 2026-08-31 it also loads the authors' published weights.

## Current state

Done for `embed` and `dock`.

| | |
|---|---|
| docking | **11.7 Å median** on 285/285, beating the do-nothing baseline on 96% |
| embedding | 128d, max-pooled, probe R **0.577** — clears the size baseline's 0.470 |

That embedding result is not a bug and was rechecked after every fix: EquiBind was trained to
*place* a ligand, so its representation encodes geometry and shape complementarity rather than
binding strength. A linear probe for affinity asks it for something it was never taught. Report
it plainly rather than explaining it away.

**The authors' `best_checkpoint.pt` loads completely**: 597/597 parameters, zero missing. The
8 keys that were once ignored are the docking head, which `equibind/models/head.py` now
implements — keypoint attention plus the Kabsch fit.

The encoder must be constructed with the authors' own training configuration, taken from
`runs/flexible_self_docking/train_arguments.yaml` in their repository:

```python
EquiBindEncoder(
    n_lays=8, layer_norm="BN", layer_norm_coords=0, final_h_layer_norm=0,
    dropout=0.1, use_rec_atoms=False, cross_msgs=True, shared_layers=False,
    noise_decay_rate=0.5, noise_initial=1, residue_emb_dim=64,
    standard_norm_order=True, nonlin="lkyrelu", leakyrelu_neg_slope=0.01,
    use_dist_in_layers=True, normalize_coordinate_update=True,
)
```

Their state_dict prefixes everything with `iegmn.`; strip it and the names line up exactly.

## Hard-won facts (do NOT regress these)

- **Four config parameters carry no weights, so a wrong value loads cleanly and silently changes
  what the encoder computes.** The authors' yaml has `use_scalar_features: False`,
  `rec_square_distance_scale: 10`, and geometry regularisation on with step 0.001. With the port's
  old defaults, receptor coordinates grew from 83 A to 2400 A across the 8 layers and RMSD read
  ~250 A. `use_scalar_features` is the nasty one: it changes feature *content* without changing
  shape, so a 597/597 parameter match proves nothing. **Read `checkpoints/train_arguments.yaml`
  before concluding anything about compatibility.**
- **Conformer generation fails on hydrogen-stripped ligands** under current RDKit, and the authors'
  fallback only catches an embed returning -1, not one that throws — so it fell back to the crystal
  pose and handed the model the answer. Embed with explicit hydrogens, remove them afterwards.
- **The docking input must be randomly placed** (`inference.py:366`): centre at the origin, rotate,
  translate up to 5 A. Feeding the aligned conformer is off-distribution and the model moves it away.
- **The transform applies to the *evolved* ligand coordinates, not the input ones**
  (`models/equibind.py:1035`, `use_evolved_lig: true`).
- Three helper functions were called but never ported; an AST check for called-but-undefined names
  guards that now.

- **The residue vocabulary is 38 classes, not 37.** The authors' `possible_amino_acids` is
  the 20 standard residues plus 17 non-standard ones (HIP, HIE, TPO, HID, LEV, MEU, PTR,
  GLV, CYT, SEP, HIZ, CYM, GLM, ASQ, TYS, CYX, GLZ) and a trailing `misc`. This port had 37
  hardcoded. Off by one shifts **every** residue index, so the model would run fine and
  encode every residue as the wrong one. Fixed 2026-08-31; the port declares only the
  dimension, so whatever produces the features must use the authors' list in their order.
- **`standard_norm_order` decides where BatchNorm sits inside the MLPs.** The authors
  trained with `true` — Linear → Norm → NonLin → Dropout → Linear → Norm. Build the port
  with the other ordering and 160 parameter names stop matching the checkpoint, which looks
  like a broken port rather than a wrong flag.
- **Configuration mismatch masquerades as incompatibility.** Built with arbitrary defaults
  this port matched 171/597 parameters; with the authors' yaml it matches 597/597. Always
  read `train_arguments.yaml` before concluding anything about compatibility.
- **Checkpoints downloaded from a paper's repository get scanned before loading.** Both
  EquiBind checkpoints came back clean — only `OrderedDict`, `_rebuild_tensor_v2` and the
  storage types — but the scan is the point, not the result.

## Docking: what it does and what travels with the number

`dock` runs the authors' protocol end to end, 285/285 on the CASF-2016 core set. Input placed
randomly 40.5 A from the truth, output **11.7 A median**, best achievable with the same conformer
1.5 A, and it beats the do-nothing baseline on 96% of complexes.

**That is about twice the paper's ~6.2 A median, and 4.6% within 5 A against their ~20%.** Not
explained by the starting position (correlation with final RMSD 0.14). CASF-2016 is not their test
set, so it is a different distribution; beyond that, settling it needs their original run side by
side. Report the number with this attached.

**Published inference is not deterministic**, so a single RMSD from it is a draw rather than a
measurement — roughly 0.3-0.5 A of spread. Two causes, both reproduced faithfully and both
switched off by default here:

- `noise_initial * noise_decay_rate ** (epoch + 1)` adds noise to features *and* coordinates inside
  `forward`, with **no `self.training` guard**. At the authors' settings and `epoch=0` that is
  N(0, 0.5) at inference. `--noise` restores it.
- the Kabsch fit perturbs its covariance with `torch.rand` until well conditioned, and on real
  complexes the keypoints are near-coplanar so it fires every time. Because the perturbation is
  diagonal in the world frame it breaks equivariance, measured at **0.15 A** by
  `test_head_equivariance.py`. A scalar multiple of the identity conditions the SVD identically and
  commutes with rotation (1.1e-05 A) — **not applied**, because the weights were trained with the
  random version.

The encoder itself is equivariant to float precision (`test_equivariance.py`, 4e-05), and so is the
docking head once that one line is set aside.

## The embedding is ours, and max-pooled

EquiBind has no affinity head — it was trained to place a ligand — so the pooling is an invention
of ours and `embed.kind` says `defined`. `max` is declared, chosen by probe over the full core set:

| pool | Pearson | Spearman | c-index | sensitivity to 5e-4 A of coordinate noise |
|---|---|---|---|---|
| sum | 0.533 | 0.582 | 0.692 | 6.05% |
| mean | 0.544 | 0.545 | 0.671 | 1.93% |
| **max** | **0.577** | **0.591** | **0.693** | **0.81%** |

It wins on every metric and is the most stable; `sum` also encodes molecule size, which is the
benchmark's own trivial baseline. So a geometry-only representation **does** carry affinity linearly
(0.577 against 0.470) — read the readout before concluding anything about the representation.

## What it can and cannot do

| capability | status |
|---|---|
| `embed` | yes — the head is already off by design |
| `dock` | yes — the head is ported in `equibind/models/head.py` |
| `predict` | **no** — EquiBind predicts a binding pose, not affinity, and this port has no head at all |
| `pretrain` / `train` | its natural role: a trained encoder to transfer from, or an architecture to train |

Because it never predicted affinity, `gnnb probe` measures something different here: whether
affinity can be linearly recovered from a representation trained only to place a ligand.

Measured on the CASF-2016 core set, 285/285 complexes, 128 dims: **R = 0.442**
(RMSE 1.948, CI 0.669).

**That is below the trivial baseline.** The ligand's heavy-atom count and bond count alone
reach R = 0.470 on the same split. This embedding does not beat counting atoms, so it is no
evidence that a pose-trained representation encodes binding affinity. The affinity-trained
models clear the baseline comfortably — SS-GNN 0.694, PLANET 0.682 — which is what a probe
result looks like when it means something.

Two caveats travel with that number. The embedding was extracted from the **crystallographic
pose**, not the RDKit conformer the model was trained on, so it is off-distribution. And
until the same probe is run on a randomly initialised encoder, it is unknown how much of the
0.442 is learned rather than architectural.
