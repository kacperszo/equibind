# CLAUDE.md — EquiBind (PyG port)

## What this is

A PyTorch Geometric reimplementation of the **IEGMN encoder** from
[HannesStark/EquiBind](https://github.com/HannesStark/EquiBind), with the docking head
(keypoint attention + SVD/Kabsch) deliberately omitted. Not a fork — a rewrite, which is
why it carries 52 tests and no DGL.

Unlike the other models in the benchmark this one arrived as an *architecture*, not a
reproduced model. As of 2026-08-31 it also loads the authors' published weights.

## Current state

**The authors' `best_checkpoint.pt` loads completely**: 597/597 parameters, zero missing,
8 keys ignored (the docking head this port does not have).

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

## Docking works, and getting there found four silent bugs

`dock` reproduces the authors' protocol end to end. On the full CASF-2016 core set, 285/285
complexes with no failures: input placed randomly 40.5 A from the truth, model output
**11.7 A median**, best achievable with the same conformer 1.5 A. It beats the do-nothing
baseline on 96% of complexes, so the model is clearly working.

**But it is about twice the paper's ~6.2 A median, and 4.6% within 5 A against their ~20%.** The gap is not explained
by the random starting position (its correlation with the final RMSD is 0.14). Three
candidates remain, in order of likelihood: CASF-2016 is not their test set and is a
different distribution; some protocol detail is still wrong; or another silent config
mismatch survives. **Settling it needs the original run side by side** — everything up to
here shows the port is close, not that it is right.

Getting there required fixing four things that all failed *silently* — the model ran, the
weights loaded, and only the numbers were wrong:

1. **Four config parameters carry no weights**, so a wrong value loads cleanly and changes
   what the encoder computes. The port's defaults disagreed with the authors' yaml on
   `use_scalar_features` (True vs **False**), `rec_square_distance_scale` (1.0 vs **10**),
   `geometry_regularization` and `geometry_reg_step_size`. With those wrong, receptor
   coordinates grew from 83 A to **2400 A** across 8 layers and keypoints landed in empty
   space, giving RMSD around 250 A. `use_scalar_features` is the nasty one: it alters node
   feature *content* without altering shape, so a 597/597 parameter match proves nothing.
2. **Three helper functions were called but never ported** — `get_rdkit_coords`,
   `rigid_transform_Kabsch_3D`, `log`. An AST check for called-but-undefined names now
   guards this.
3. **Conformer generation fails on hydrogen-stripped ligands** under RDKit 2026
   ("Cannot be a chiral center", "nbrs.size() >= 3"), and the authors' fallback only catches
   an embed returning -1, not one that throws. Every test ligand failed, and
   `get_lig_graph_revised` quietly fell back to the crystal pose — handing the model the
   answer. Embedding with explicit hydrogens and removing them afterwards fixes it.
4. **The input must be randomly placed** (`inference.py:366`): centre the conformer at the
   origin, rotate randomly, translate up to 5 A. Feeding the Kabsch-aligned conformer puts
   the ligand in the pocket already, which is off-distribution — the model then moves it
   *away* and scores worse than doing nothing.

5. **The transform is applied to the *evolved* ligand coordinates, not the input ones**
   (`models/equibind.py:1035`, with the authors' `use_evolved_lig: true`). The Kabsch fit is
   solved from keypoints built out of the evolved coordinates, so applying it to the input
   makes the answer depend on where the input happened to be placed.

## The port is equivariant; the published method is not

`test_equivariance.py` confirms the encoder is exact to float32 — node features invariant and
coordinates equivariant to within 4e-05 under random rotation and translation. That is the
gate that matters, because the encoder is what the embeddings come from.

The full docking pipeline is *not* deterministic, and that is faithful rather than broken.
Two parts of the published method introduce randomness:

- **Noise at inference.** `noise_initial * noise_decay_rate ** (epoch + 1)` with the authors'
  `noise_initial: 1`, `noise_decay_rate: 0.5`, and `epoch=0` passed by their own
  `inference.py:385` — so published inference adds N(0, 0.5) to features *and* coordinates.
- **Kabsch stabilisation.** The fit perturbs its covariance with `torch.rand` until well
  conditioned, and on real complexes the keypoints are nearly coplanar (singular values around
  2, 0.02, 6e-05) so it fires every time. The perturbation lives in the world frame, which
  makes the solved rotation depend on the input orientation.

Together these put roughly 0.3-0.5 A of irreducible spread on the predicted pose. **Any single
RMSD from this model is a draw from a distribution, not a measurement** — including the
authors' published figures.

**The embeddings were recomputed after every fix and the conclusion did not change.** The
probe gives R=0.428 against a trivial baseline of 0.470 — before the fixes it was 0.442, so
the difference is noise. The exploding coordinates were not why this representation fails to
predict affinity.

That is worth stating plainly rather than explaining away: EquiBind was trained to *place* a
ligand, so its representation encodes geometry and shape complementarity. A linear probe for
binding affinity is asking it for something it was never taught, and the honest reading is
that it does not carry that information in linearly accessible form. SS-GNN and PLANET, both
trained on affinity, clear the same baseline comfortably.

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
