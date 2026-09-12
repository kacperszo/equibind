# EquiBind — an E(3)-equivariant encoder trained to place a ligand

> A fork maintained for [gnn-benchmark](../../README.md). The authors' own README is
> kept as [README.upstream.md](README.upstream.md) for attribution and for their
> description of the method — **its build and run instructions are not current for
> this fork.**

## What it is

A PyTorch Geometric reimplementation of the IEGMN encoder: eight equivariant layers
over separate ligand and receptor graphs with cross-messages between them, evolving coordinates as
well as features. The head solves for a rigid transform by keypoint attention plus a Kabsch fit. Not a
fork but a rewrite, which is why it carries 52 tests and no DGL — and the authors' published weights
load into it unchanged, 597/597 parameters.

It is the only model here with no affinity head, which is what makes its encoder the interesting one
to transfer from: a representation trained purely on geometry.

## State

| | |
|---|---|
| docking | **11.7 A median** on 285/285, beating the do-nothing baseline on 96% |
| embedding | 128d, **ours** (max-pooled), probe R 0.577 — clears the size baseline's 0.470 |
| `gnnb verify` | dock 3.6e-15, embed **0.0** |
| equivariance | encoder 3.8e-05; the docking head 0.15 A, from one line in the Kabsch stabiliser |
| training | `train` and `finetune`, with an affinity head that is ours, not theirs |

## Build

```bash
podman build --format=docker -t equibind:latest .
```

## Run it, without the harness

Generated from this model's adapter by `gnnb howto`, so these are the exact commands
the benchmark issues — regenerate with `python tools/sync_model_readmes.py`. Every one
runs with `--network=none` and a read-only root filesystem.

Input is one directory per complex:

    <complexes>/<id>/<id>_protein.pdb
    <complexes>/<id>/<id>_ligand.sdf      # or .mol2; several models try both

Reads the ligand with RDKit and generates its own conformer for docking — the crystallographic pose is the *answer*, so docking must not be given it. Embedding does use the crystal pose, deliberately.

```bash
# equibind.modern — localhost/equibind:latest
# source: models/equibind

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
    -v "$PWD/checkpoints:/ckpt:ro" \
    -v /path/to/splits:/splits:ro \
    -v /path/to/cache:/cache:rw,U \
    --shm-size 4g \
    localhost/equibind:latest \
    python train_affinity.py --complexes /data --labels /splits/train.csv --out /outputs --pool max --seed 0 --device cpu --val-labels /splits/val.csv --cache /cache --epochs 30

# finetune  (encoder frozen; drop --freeze-encoder to tune all of it)
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v "$PWD/checkpoints:/ckpt:ro" \
    -v /path/to/splits:/splits:ro \
    -v /path/to/cache:/cache:rw,U \
    --shm-size 4g \
    localhost/equibind:latest \
    python train_affinity.py --complexes /data --labels /splits/train.csv --out /outputs --pool max --seed 0 --device cpu --val-labels /splits/val.csv --cache /cache --epochs 30 --init-encoder /ckpt/encoder.pt --freeze-encoder
```

## What comes out

| file | holds |
|---|---|
| `predictions.csv` | docking: `complex_id,rmsd,rmsd_conformer,rmsd_aligned` — the prediction, the do-nothing baseline, and the best a rigid method could manage. A bare RMSD says nothing without the other two |
| `embeddings.npz` | `ids` and `vectors`, 128-dim — pooled ligand and receptor node embeddings concatenated. Ours, not the authors' |
| `model.pt` | training only; the IEGMN plus the affinity head from `equibind/models/affinity.py` |

## Before you trust the numbers

**Published inference is not deterministic.** `noise_initial > 0` adds `randn_like` to features and coordinates inside `forward` with no `self.training` guard, so the authors' own settings make a single RMSD a sample rather than a measurement — roughly 0.3-0.5 Å of irreducible spread, and the Kabsch fit perturbs a near-degenerate covariance on top. Our scripts turn it off by default and take `--noise` to restore it.

**The affinity head is ours.** EquiBind has none: it was trained to place a ligand, not to score one, which is what makes it the interesting encoder here and why a fine-tune needs the random-init control run beside it.

## Maintainer notes

`CLAUDE.md` in this directory holds what breaks if it is changed back.
