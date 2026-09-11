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
| embedding | 128d, probe R **0.428** — **does not clear** the size baseline's 0.470 |

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
coordinates equivariant to within 4e-05 under random rotation and translation.

**`test_head_equivariance.py` now covers the docking head too**, which is the part the authors
published and the part `dock` has a golden for. The head is equivariant by construction:
keypoints are a weighted sum of coordinates with weights taken from *features*, so `K -> RK`;
`a = (rec - mean)^T (lig - mean) -> R a R^T`; and the SVD carries that to
`rotation -> R rotation R^T`. Measured by rotating receptor and ligand together and checking the
predicted pose follows:

| perturbation in the Kabsch stabiliser | worst departure | fired |
|---|---|---|
| as published, `rand(3, 3) * eye` | **1.49e-01 A** | 4 of 4 solves |
| a scalar multiple of the identity | **1.14e-05 A** | 4 of 4 |

So one line accounts for all of it. A random diagonal does not commute with `R`; `cI` does —
`R (a + cI) R^T = R a R^T + cI` — and it conditions the SVD just as well, which makes the
equivariant version a strictly better stabiliser mathematically.

**It is not applied.** The published weights were trained with the random version in place, so
changing it would depart from the method being reproduced. It is the right choice for any
encoder we train ourselves, and the test exists so the cost of leaving it alone stays a number:
0.15 A, against a median docking RMSD of 11.7 A.

**The published method is stochastic; this port runs it deterministically by default.** Two
parts introduce randomness:

- **Noise at inference.** `noise_initial * noise_decay_rate ** (epoch + 1)` with the authors'
  `noise_initial: 1`, `noise_decay_rate: 0.5`, and `epoch=0` passed by their own
  `inference.py:385` — so published inference adds N(0, 0.5) to features *and* coordinates.
- **Kabsch stabilisation.** The fit perturbs its covariance with `torch.rand` until well
  conditioned, and on real complexes the keypoints are nearly coplanar (singular values around
  2, 0.02, 6e-05) so it fires every time. The perturbation lives in the world frame, which
  makes the solved rotation depend on the input orientation — **measured at 0.15 A** by
  `test_head_equivariance.py`, so the rest of the spread below is the noise rather than this.

Together these put roughly 0.3-0.5 A of spread on the predicted pose as the authors run it.
**Any single RMSD from published EquiBind is a draw from a distribution, not a measurement** —
including their own figures.

Both are now off by default, and `--noise` restores the authors' behaviour:

- the encoder's noise is a *training* augmentation that reached inference through a missing
  `self.training` guard, so switching it off is closer to what the model was meant to do than
  leaving it on. It is also better: on 24 complexes, within-5 A went from 3.2% to 58.3% and
  the model beat the untouched conformer on 100% of them rather than 95%.
- the Kabsch perturbation cannot be removed — it is what keeps the fit conditioned — so it is
  pinned instead, with `torch.manual_seed`. Without that line, and with the encoder noise
  already off, two identical runs still disagreed by up to 0.5 A on 14 of 24 complexes.
  Threads were not the cause; this was.

The placement seed is now derived per complex, from `crc32(complex_id)` rather than drawn from
one stream shared across the run. A shared stream made every pose depend on how many draws the
complexes before it consumed, so adding or dropping one shifted all of them — which is why the
first recorded golden disagreed with a rerun on all 285 rather than on the one that changed.

`gnnb verify equibind.modern` now reproduces docking to 3.6e-15.

**A number that has to be remeasured:** the probe figures below were taken with the encoder
noise on. They are a draw from a distribution, and the deterministic path may not give the
same answer.

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
