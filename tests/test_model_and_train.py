import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradwarden.config import PatchError, apply_patch, validate_patch
from gradwarden.model import Transformer, init_params
from gradwarden.train import Trainer


def test_shapes_and_stats(tiny_cfg):
    params = init_params(tiny_cfg.model)
    tokens = jnp.zeros((3, tiny_cfg.model.seq_len), dtype=jnp.int32)
    logits, stats = Transformer(tiny_cfg.model).apply({"params": params}, tokens)
    assert logits.shape == (3, tiny_cfg.model.seq_len, tiny_cfg.model.vocab_size)
    for key in ("attn_logit_max", "attn_entropy", "act_rms"):
        assert stats[key].shape == (tiny_cfg.model.n_layers,)
    # entropy is normalized so it has to sit in [0, 1]
    assert np.all((stats["attn_entropy"] >= 0) & (stats["attn_entropy"] <= 1.0001))


def test_causal(tiny_cfg):
    params = init_params(tiny_cfg.model)
    model = Transformer(tiny_cfg.model)
    t = tiny_cfg.model.seq_len
    a = jnp.arange(t, dtype=jnp.int32)[None] % tiny_cfg.model.vocab_size
    b = a.at[0, t // 2 :].set(0)  # change only the second half
    la, _ = model.apply({"params": params}, a)
    lb, _ = model.apply({"params": params}, b)
    np.testing.assert_allclose(la[0, : t // 2], lb[0, : t // 2], atol=1e-5)


def test_qk_norm_flag_keeps_param_tree(tiny_cfg):
    on = dataclasses.replace(tiny_cfg.model, qk_norm=True)
    s1 = jax.tree_util.tree_structure(init_params(tiny_cfg.model))
    s2 = jax.tree_util.tree_structure(init_params(on))
    assert s1 == s2  # needed so the agent can flip qk_norm mid run


def test_loss_goes_down(tiny_cfg, data):
    tr = Trainer(tiny_cfg, data)
    tr.run(60)
    first = np.mean([r["loss"] for r in tr.history[:5]])
    last = np.mean([r["loss"] for r in tr.history[-5:]])
    assert last < first - 0.3
    assert "val_loss" in tr.history[19]
    assert set(tr.history[0]["grad_norm_by_module"]) >= {"embed", "block_0", "lm_head"}


def test_rewind_is_exact(tiny_cfg, data):
    # rewinding and replaying with no patch should give the same losses back
    tr = Trainer(tiny_cfg, data)
    tr.run(30)
    original = [r["loss"] for r in tr.history[20:30]]
    restored = tr.rewind(20)
    assert restored == 20
    tr.run(10)
    replay = [r["loss"] for r in tr.history[20:30]]
    np.testing.assert_allclose(original, replay, rtol=1e-5)


def test_patch_then_rewind(tiny_cfg, data):
    tr = Trainer(tiny_cfg, data)
    tr.run(25)
    tr.apply_patch({"qk_norm": True, "lr": 1e-3, "z_loss": 1e-4})
    assert tr.cfg.model.qk_norm and tr.cfg.train.lr == 1e-3
    tr.rewind(10)
    tr.run(5)
    assert all(math.isfinite(r["loss"]) for r in tr.history)


def test_patch_validation():
    assert validate_patch({"lr": 1e-3}) == []
    assert validate_patch({}) == ["patch is empty"]
    assert validate_patch({"lr": 5.0})
    assert validate_patch({"n_layers": 8})  # not patchable
    assert validate_patch({"qk_norm": "yes"})
    with pytest.raises(PatchError):
        apply_patch(None, {"lr": -1})
