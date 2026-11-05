"""Check that the encoder detaches from its head and carries its representation with it.

`test_equivariance.py` asks whether the port computes what EquiBind computes. This asks whether
the encoder can be lifted out and reused, which is a different way to be wrong: every failure
below still trains and still prints a curve, just a worse one for a reason nothing reports.

The load-bearing case is **the embedding surviving transfer** — after a transfer the new model
must embed exactly as the old one did, because only the head was replaced. That is what makes
the linear probe's R = 0.577 a statement about the thing a fine-tune actually starts from, rather
than about a vector nobody transfers.

A plain script rather than pytest, like `test_equivariance.py` beside it: the image carries no
pytest, and a check that cannot run where the model runs is a check that stops being run.

usage:
    python test_transfer.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

from embed_complexes import ENCODER_CONFIG, POOLS
from equibind.models.affinity import AffinityHead, EquiBindAffinity
from equibind.models.encoder import EquiBindEncoder
from equibind.synthetic import _make_lig, _make_rec
from transfer import DOCKING_HEAD_PREFIXES, is_head, set_training_mode, transfer_encoder

FEATS_DIM = 64
PASSED: list[str] = []
FAILED: list[str] = []


def case(name):
    def wrap(fn):
        try:
            fn()
        except AssertionError as e:
            FAILED.append(f"{name}: {e}")
        except Exception as e:
            FAILED.append(f"{name}: {type(e).__name__}: {e}")
        else:
            PASSED.append(name)
        return fn
    return wrap


def refuses(fn, fragment: str) -> None:
    """Require `fn` to bail out, for the stated reason."""
    try:
        fn()
    except (SystemExit, ValueError) as e:
        assert fragment in str(e), f"refused, but for the wrong reason: {e}"
        return
    raise AssertionError(f"accepted what it should refuse (expected {fragment!r})")


def build(seed: int, pool: str = "sum") -> EquiBindAffinity:
    torch.manual_seed(seed)
    config = dict(ENCODER_CONFIG)
    config["noise_initial"] = 0        # the noise has no self.training guard; off for a test
    encoder = EquiBindEncoder(device="cpu", **config)
    return EquiBindAffinity(encoder, feats_dim=FEATS_DIM, pool=pool)


def encoder_state(model: EquiBindAffinity) -> dict:
    """The encoder half, named as a split of the full checkpoint that strips `iegmn.` writes it."""
    return {k[len("iegmn."):]: v for k, v in model.state_dict().items()
            if k.startswith("iegmn.")}


def write_encoder(model: EquiBindAffinity, directory: Path, name: str = "encoder.pt") -> str:
    path = directory / name
    torch.save({"encoder": encoder_state(model), "variant": "equibind.modern"}, path)
    return str(path)


def graphs():
    return _make_lig(), _make_rec(), None


@case("the head is ours and the encoder is theirs")
def _():
    names = list(build(seed=0).state_dict())
    head = [n for n in names if is_head(n)]
    assert head, "no parameter matched the affinity head"
    assert all(n.startswith("affinity.") for n in head), head[:5]
    assert all(n.startswith("iegmn.") for n in names if not is_head(n))


@case("the docking-head prefixes are exactly the keys the port does not implement")
def _():
    """If these disagree, a split either drops encoder weights or smuggles the
    docking head across, and the strict load is the only thing that would notice."""
    encoder_names = set(encoder_state(build(seed=0)))
    for prefix in DOCKING_HEAD_PREFIXES:
        assert not any(n == prefix or n.startswith(prefix + ".") for n in encoder_names), \
            f"{prefix} is implemented by the encoder port, so it is not the docking head"


@case("transfer moves the encoder and leaves the head")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        source = build(seed=0)
        path = write_encoder(source, Path(tmp))
        fresh = build(seed=99)
        before = {k: v.clone() for k, v in fresh.state_dict().items()}

        transfer_encoder(fresh, path, freeze=False)
        after = fresh.state_dict()

        for name, value in source.state_dict().items():
            if name.startswith("iegmn."):
                assert torch.equal(after[name], value), f"{name} did not transfer"
        for name in (n for n in after if is_head(n)):
            assert torch.equal(after[name], before[name]), f"{name} was overwritten; it is ours"


@case("the embedding survives transfer")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        source = build(seed=0)
        path = write_encoder(source, Path(tmp))
        fresh = build(seed=99)
        transfer_encoder(fresh, path, freeze=False)

        lig, rec, geom = graphs()
        source.eval()
        fresh.eval()
        with torch.no_grad():
            h_lig_a, h_rec_a = source.iegmn(lig, rec, geom)[:2]
            h_lig_b, h_rec_b = fresh.iegmn(lig, rec, geom)[:2]
            score_a, score_b = source(lig, rec, geom), fresh(lig, rec, geom)

        pool = POOLS["sum"]
        a = torch.cat([pool(h_lig_a), pool(h_rec_a)])
        b = torch.cat([pool(h_lig_b), pool(h_rec_b)])
        assert torch.allclose(a, b, atol=1e-5), \
            "the transferred model embeds differently — the boundary cuts through the encoder"
        assert not torch.allclose(score_a, score_b, atol=1e-6), \
            "the affinity is unchanged, so the head came across too"


@case("freezing leaves the encoder untouched, buffers included")
def _():
    """A frozen encoder must not move at all — and `requires_grad=False` is not enough.

    Every IEGMN MLP carries BatchNorm, whose `running_mean` and `running_var` are buffers, not
    parameters: they update on every forward in train mode no matter what the gradients do. So
    the encoder drifts, the representation the head is being fitted on drifts with it, and
    nothing in the training loop reports anything. `set_training_mode` is what prevents it, and
    the case below shows the drift happening without it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = write_encoder(build(seed=0), Path(tmp))
        model = build(seed=99)
        transfer_encoder(model, path, freeze=True)

        assert all(not p.requires_grad for n, p in model.named_parameters() if not is_head(n))
        assert all(p.requires_grad for n, p in model.named_parameters() if is_head(n))

        set_training_mode(model, training=True, frozen_encoder=True)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        optimiser = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.1)
        lig, rec, geom = graphs()
        model(lig, rec, geom).sum().backward()
        optimiser.step()

        after = model.state_dict()
        for name in (n for n in after if not is_head(n)):
            assert torch.equal(after[name], before[name]), f"{name} moved while frozen"
        assert not torch.equal(after["affinity.out.weight"], before["affinity.out.weight"]), \
            "the head did not learn"


@case("set_training_mode is what stops the batchnorm drift")
def _():
    """The negative control for the case above: without it the buffers move.

    Worth pinning because the guard is one line in the trainer and looks removable. If this
    ever stops failing, BatchNorm has changed and the guard's justification needs rechecking —
    not that the guard has become unnecessary.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = write_encoder(build(seed=0), Path(tmp))
        model = build(seed=99)
        transfer_encoder(model, path, freeze=True)

        model.train()                    # deliberately not set_training_mode
        before = {k: v.clone() for k, v in model.state_dict().items()}
        lig, rec, geom = graphs()
        with torch.no_grad():
            model(lig, rec, geom)

        moved = [k for k, v in model.state_dict().items()
                 if not is_head(k) and not torch.equal(v, before[k])]
        assert moved, ("no frozen tensor moved in train mode, so the eval-mode guard in "
                       "set_training_mode no longer has a hazard to guard against")
        assert all("running_" in k or "num_batches_tracked" in k for k in moved), \
            f"something other than batchnorm buffers moved: {moved[:5]}"


@case("a whole checkpoint is refused")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        state = encoder_state(build(seed=0))
        state["keypts_attention_lig.0.weight"] = torch.zeros(1920, 64)
        path = Path(tmp) / "whole.pt"
        torch.save({"encoder": state}, path)
        refuses(lambda: transfer_encoder(build(seed=1), str(path), freeze=False), "docking-head")


@case("an already-prefixed encoder loads")
def _():
    """a split that does not strip writes iegmn.-prefixed names; both shapes must work."""
    with tempfile.TemporaryDirectory() as tmp:
        source = build(seed=0)
        state = {k: v for k, v in source.state_dict().items() if k.startswith("iegmn.")}
        path = Path(tmp) / "prefixed.pt"
        torch.save({"encoder": state}, path)
        model = build(seed=99)
        transfer_encoder(model, str(path), freeze=False)
        for name, value in state.items():
            assert torch.equal(model.state_dict()[name], value)


@case("the head pools the way the embedding did")
def _():
    """A head pooling differently from embed_complexes.py fine-tunes a different vector."""
    head = AffinityHead(feats_dim=FEATS_DIM, pool="sum")
    h = torch.arange(12, dtype=torch.float).reshape(3, 4)
    assert torch.equal(head.pool(h), POOLS["sum"](h))
    refuses(lambda: AffinityHead(feats_dim=FEATS_DIM, pool="median"), "pool must be")


if __name__ == "__main__":
    for name in PASSED:
        print(f"  ok    {name}")
    for line in FAILED:
        print(f"  FAIL  {line}")
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    sys.exit(1 if FAILED else 0)
