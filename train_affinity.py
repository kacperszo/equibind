"""Fine-tune the EquiBind encoder on affinity, with a head it never had.

EquiBind predicts a binding pose. It has no affinity head and no affinity loss, so unlike a
reproduction trainer this one reproduces nothing — it is the transfer
experiment itself, and `equibind/models/affinity.py` says which part is ours.

The question it exists to answer: a linear probe on the *frozen*, max-pooled representation
reaches R = 0.577 on the CASF-2016 core set, against 0.470 for counting heavy atoms and bonds.
That is a result about frozen geometry embeddings. It says nothing about what the same encoder does when
its weights are allowed to move, and the two runs worth comparing are:

    --freeze-encoder    the probe again, with a non-linear head. Does the representation carry
                        affinity information a ridge could not reach?
    (unfrozen)          does a geometry-pretrained initialisation beat a random one?

The second needs its control run — the same head on the same split with `--init-encoder`
omitted — or the number means nothing.

**Noise is off unless asked for.** EquiBind adds `noise_level * randn_like(...)` to features
and coordinates inside `forward` with no `self.training` guard, so it is on at inference too.
For training that is an augmentation and defensible; for a frozen encoder it means the head
learns against a representation resampled every step, so freezing implies noise off.

usage:
    python train_affinity.py --complexes /data --labels /splits/train.csv \
        --val-labels /splits/val.csv --out /outputs --cache /cache \
        --init-encoder /ckpt/encoder.pt --freeze-encoder
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from rdkit import RDLogger
from tqdm import tqdm

# LIG_RADIUS and REC_CUTOFF are named in the cache filename so a run cannot silently reuse
# graphs built with other radii — the geometry is baked into them.
from embed_complexes import ENCODER_CONFIG, LIG_RADIUS, REC_CUTOFF, featurize
from equibind.models.affinity import EquiBindAffinity
from equibind.models.encoder import EquiBindEncoder
from transfer import set_training_mode, transfer_encoder

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

#: node feature width the published configuration produces, per side
FEATS_DIM = 64


def build_graphs(complexes: str, labels: dict[str, float]) -> list:
    """Featurise every complex once. The three graphs and the label travel together."""
    built, failed = [], []
    for cid in tqdm(sorted(labels), desc="Featurising"):
        d = os.path.join(complexes, cid)
        if not os.path.isdir(d):
            failed.append(f"{cid}: no directory")
            continue
        try:
            lig, rec, geom = featurize(d, cid)
            built.append({"id": cid, "lig": lig, "rec": rec, "geom": geom,
                          "y": float(labels[cid])})
        except Exception as e:
            failed.append(f"{cid}: {type(e).__name__}: {e}")
    if failed:
        print(f"{len(failed)} complexes failed to featurise:")
        for line in failed[:10]:
            print("  ", line)
    return built


def load_or_build(cache: str | None, complexes: str, labels_path: str) -> list:
    """Featurise, or reuse a cached copy keyed by the split.

    `weights_only=False` reading back: these are graph objects rather than plain tensors, and
    the file was written by this script in this container. That is the whole difference from
    the rule about checkpoints that came from elsewhere.
    """
    with open(labels_path) as f:
        labels = {r["complex_id"]: float(r["y_true"]) for r in csv.DictReader(f)}

    if not cache:
        return build_graphs(complexes, labels)

    path = Path(cache) / f"equibind_{Path(labels_path).stem}_lig{LIG_RADIUS}_rec{REC_CUTOFF}.pt"
    if path.exists():
        data = torch.load(path, weights_only=False)
        print(f"reusing {path}: {len(data)} complexes")
        return data
    data = build_graphs(complexes, labels)
    torch.save(data, path)
    print(f"cached {len(data)} complexes -> {path}")
    return data



def run_epoch(model, data, device, optimiser, batch_size, frozen) -> dict:
    """One pass. `optimiser=None` means validation.

    Gradients accumulate over `batch_size` complexes and the step is taken once, because the
    encoder takes one complex at a time — the ligand and receptor graphs are separate objects
    with cross-messages between them, and there is no batched form of that here. The arithmetic
    is a mean over the batch either way; only the memory profile differs.
    """
    training = optimiser is not None
    set_training_mode(model, training, frozen_encoder=frozen)

    order = torch.randperm(len(data)).tolist() if training else list(range(len(data)))
    criterion = nn.MSELoss()
    total, n, preds, trues = 0.0, 0, [], []

    for start in range(0, len(order), batch_size):
        chunk = [data[i] for i in order[start:start + batch_size]]
        if training:
            optimiser.zero_grad()
        for item in chunk:
            with torch.set_grad_enabled(training):
                out = model(item["lig"].to(device), item["rec"].to(device),
                            item["geom"].to(device))
                y = torch.tensor([item["y"]], device=device, dtype=out.dtype)
                loss = criterion(out.view(1), y) / len(chunk)
            if training:
                loss.backward()
            total += float(loss) * len(chunk)
            n += 1
            preds.append(float(out))
            trues.append(item["y"])
        if training:
            optimiser.step()

    p, t = np.array(preds), np.array(trues)
    r = float(np.corrcoef(p, t)[0, 1]) if len(p) > 1 and p.std() and t.std() else float("nan")
    return {"loss": total / max(n, 1), "r": r,
            "rmse": float(np.sqrt(((p - t) ** 2).mean()))}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--complexes", required=True)
    p.add_argument("--labels", required=True, help="csv with complex_id,y_true")
    p.add_argument("--val-labels", dest="val_labels", default=None)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--cache", default=None)
    p.add_argument("--init-encoder", dest="init_encoder", default=None,
                   help="encoder-only .pt (no head tensors); omit for the random-init control")
    p.add_argument("--freeze-encoder", dest="freeze_encoder", action="store_true")
    p.add_argument("--pool", default="sum", choices=["sum", "mean", "max"],
                   help="must match what embed_complexes.py wrote for the probe to be "
                        "comparable; `mean` is the ablation that removes the size shortcut")
    p.add_argument("--noise", action="store_true",
                   help="restore EquiBind's forward-pass noise as an augmentation. Refused "
                        "with --freeze-encoder, where it would resample the frozen "
                        "representation every step")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if args.noise and args.freeze_encoder:
        raise SystemExit("--noise with --freeze-encoder would resample the representation the "
                         "head is being fitted on; pick one")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    train = load_or_build(args.cache, args.complexes, args.labels)
    if not train:
        raise SystemExit("no training graphs were built")
    val = (load_or_build(args.cache, args.complexes, args.val_labels)
           if args.val_labels else None)
    print(f"{len(train)} training complexes" + (f", {len(val)} validation" if val else ""))

    config = dict(ENCODER_CONFIG)
    if not args.noise:
        config["noise_initial"] = 0
    encoder = EquiBindEncoder(device=device, **config)
    model = EquiBindAffinity(encoder, feats_dim=FEATS_DIM, pool=args.pool,
                             dropout=args.dropout).to(device)

    provenance: dict = {"init_encoder": None, "frozen_params": 0}
    if args.init_encoder:
        provenance = transfer_encoder(model, args.init_encoder, args.freeze_encoder)
    elif args.freeze_encoder:
        raise SystemExit("--freeze-encoder without --init-encoder would fit the head against a "
                         "random representation; that is not an experiment")
    else:
        print("no encoder transferred: this is the random-init control")

    optimiser = torch.optim.Adam([q for q in model.parameters() if q.requires_grad],
                                 lr=args.lr, weight_decay=args.weight_decay)

    out = Path(args.out)
    out_dir = out.parent if out.suffix == ".pt" else out
    model_path = out if out.suffix == ".pt" else out / "model.pt"
    out_dir.mkdir(parents=True, exist_ok=True)

    history, best, best_epoch = [], float("inf"), -1
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train, device, optimiser, args.batch_size, args.freeze_encoder)
        row = {"epoch": epoch, "train_loss": tr["loss"], "train_r": tr["r"],
               "val_loss": "", "val_r": "", "train_rmse": tr["rmse"], "val_rmse": ""}
        line = f"epoch {epoch:4d}  train mse {tr['loss']:7.4f}  R {tr['r']:+.3f}"
        if val:
            va = run_epoch(model, val, device, None, args.batch_size, args.freeze_encoder)
            row.update(val_loss=va["loss"], val_r=va["r"], val_rmse=va["rmse"])
            line += f"  |  val mse {va['loss']:7.4f}  R {va['r']:+.3f}"
            score = va["loss"]
        else:
            score = tr["loss"]
        history.append(row)
        print(line, flush=True)

        if score < best:
            best, best_epoch = score, epoch
            torch.save({"model_state_dict": model.state_dict(),
                        "feats_dim": FEATS_DIM, "pool": args.pool,
                        "dropout": args.dropout, "noise": bool(args.noise),
                        "epoch": epoch, "score": best}, model_path)

    with (out_dir / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (out_dir / "summary.json").write_text(json.dumps({
        "model": "equibind",
        "head": "ours — EquiBind has no affinity head; see equibind/models/affinity.py",
        "pool": args.pool,
        "noise": bool(args.noise),
        "control": "a run with --init-encoder omitted is the random-init control this needs",
        "epochs": len(history), "best_epoch": best_epoch, "best_score": best,
        "selected_on": "val_mse" if val else "train_mse",
        "train_complexes": len(train), "val_complexes": len(val) if val else 0,
        "seed": args.seed, "freeze_encoder": args.freeze_encoder,
        "lr": args.lr, "batch_size": args.batch_size,
        "seconds": round(time.time() - started, 1),
        **provenance,
    }, indent=2) + "\n")

    print(f"\nbest {best:.4f} at epoch {best_epoch} -> {model_path}  "
          f"({time.time() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
