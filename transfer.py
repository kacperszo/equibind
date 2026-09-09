"""Move the published IEGMN encoder into a model with an affinity head on top.

EquiBind is the one model here that is a *source* of encoders rather than a target. It was
trained on geometry, has no affinity head, and its own head is a docking algorithm — so the
transfer question it answers is the one the whole programme is about: does a representation
learned by placing ligands carry anything about how tightly they bind.

**The boundary is unusually clean, and it was already measured.** Everything in the authors'
checkpoint sits under `iegmn.`, and of those 605 tensors exactly 8 belong to the docking head:

    h_mean_lig, h_mean_rec, keypts_attention_lig, keypts_attention_rec,
    keypts_queries_lig, keypts_queries_rec

The remaining 597 are precisely what `equibind/models/encoder.py` implements — the port was
written to the encoder and stops there — and precisely what produces the embedding
`embed_complexes.py` writes. Encoder boundary, port boundary and embedding boundary are the
same cut, which is as good as this gets.

`prefix` is the mirror of the harness's `strip`: the split removes `iegmn.` on the way out so
the file's names match the port, and this puts it back on the way in so they match the
composite model. Cutting and loading have to agree about naming or the load silently moves
nothing.
"""

from __future__ import annotations

import torch
import torch.nn as nn

#: The affinity head, which is ours. Named as a prefix of the composite model rather than of
#: the checkpoint, because the checkpoint has no affinity head to name.
HEAD_PREFIXES = ("affinity",)

#: The docking head inside the authors' checkpoint, under `iegmn.` and stripped along with it.
#: Must agree with `encoder.head` for equibind.modern in harness/registry.toml.
DOCKING_HEAD_PREFIXES = ("h_mean_lig", "h_mean_rec", "keypts_attention_lig",
                         "keypts_attention_rec", "keypts_queries_lig", "keypts_queries_rec")


def is_head(name: str) -> bool:
    return any(name == p or name.startswith(p + ".") for p in HEAD_PREFIXES)


def read_encoder(path: str) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and isinstance(payload.get("encoder"), dict):
        payload = payload["encoder"]
    if isinstance(payload, dict) and isinstance(payload.get("model_state_dict"), dict):
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: expected a state dict, got {type(payload).__name__}")
    return {k: v for k, v in payload.items() if torch.is_tensor(v)}


def transfer_encoder(model: nn.Module, path: str, freeze: bool = False,
                     prefix: str = "iegmn.") -> dict[str, object]:
    """Overlay the encoder onto a fresh model, leaving the affinity head untouched.

    Loaded strictly after overlaying, so a name that does not line up is an error rather than a
    silent skip — which here would mean fine-tuning a randomly initialised IEGMN and reporting
    it as a transfer result.
    """
    raw = read_encoder(path)
    # Accept a file that already carries the prefix, so an encoder cut without `--strip` and
    # one cut with it both load. Mixed files are refused rather than half-renamed.
    already = [k for k in raw if k.startswith(prefix)]
    if already and len(already) != len(raw):
        raise SystemExit(f"{path}: {len(already)} of {len(raw)} keys carry {prefix!r}; the file "
                         f"mixes two namings and cannot be loaded unambiguously")
    encoder = raw if already else {prefix + k: v for k, v in raw.items()}

    target = model.state_dict()
    docking = [k for k in encoder
               if any(k.startswith(prefix + p) for p in DOCKING_HEAD_PREFIXES)]
    if docking:
        raise SystemExit(
            f"{path} carries {len(docking)} docking-head tensors ({docking[:3]}). That is the "
            f"whole checkpoint — cut it with `gnnb encoder split`, which knows the boundary."
        )
    unknown = [k for k in encoder if k not in target]
    if unknown:
        raise SystemExit(f"{path}: {len(unknown)} tensors have no slot in this model: {unknown[:5]}")
    mismatched = [k for k, v in encoder.items() if target[k].shape != v.shape]
    if mismatched:
        raise SystemExit(
            f"{path}: shape mismatch on {len(mismatched)} tensors: "
            + ", ".join(f"{k} {tuple(encoder[k].shape)} into {tuple(target[k].shape)}"
                        for k in mismatched[:3])
            + ". The encoder was built with a different configuration; read "
              "checkpoints/train_arguments.yaml before concluding anything about compatibility."
        )
    uncovered = [k for k in target if k not in encoder and not is_head(k)]
    if uncovered:
        raise SystemExit(
            f"{path}: {len(uncovered)} model tensors are neither in the encoder nor in the "
            f"head — the boundary is wrong: {uncovered[:5]}"
        )

    model.load_state_dict({**target, **encoder}, strict=True)
    moved = sum(v.numel() for v in encoder.values())
    print(f"transferred {len(encoder)} tensors / {moved:,} params from {path}")

    frozen = 0
    if freeze:
        for name, param in model.named_parameters():
            if not is_head(name):
                param.requires_grad_(False)
                frozen += param.numel()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"froze {frozen:,} encoder params; {trainable:,} trainable in the affinity head")

    return {"init_encoder": path, "transferred_tensors": len(encoder),
            "transferred_params": moved, "frozen_params": frozen}


def set_training_mode(model: nn.Module, training: bool, frozen_encoder: bool) -> None:
    """Keep a frozen encoder deterministic.

    Two sources of randomness here, not one. Dropout at 0.1 in every IEGMN layer is the usual
    one; the other is EquiBind's own — `noise_initial > 0` adds `randn_like` to features and
    coordinates inside `forward` **with no `self.training` guard**, so eval mode does not stop
    it. Only building the encoder with `noise_initial = 0` does, which is why the trainer does
    that whenever the encoder is frozen and says so in `summary.json`.
    """
    if not training or not frozen_encoder:
        model.train(training)
        return
    model.eval()
    for name, module in model.named_modules():
        if name and is_head(name) and any(p.requires_grad for p in module.parameters()):
            module.train()
