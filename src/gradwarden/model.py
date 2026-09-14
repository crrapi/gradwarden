"""tiny pre-LN decoder. same shape as everything else but it also hands you the
per-layer stats on every forward pass so detectors have something to look at.

per layer:
  attn_logit_max   max |q.k/sqrt(d)| over unmasked positions. this is the one
                   that blows up at high lr (dehghani 2023, wortsman 2023)
  attn_entropy     mean attn entropy / log(#visible keys) -> 0..1. heads going to
                   ~0 = entropy collapse (zhai 2023)
  act_rms          residual stream rms after the block, mostly a sanity dial
"""

from __future__ import annotations

import math

import flax.linen as nn
import jax
import jax.numpy as jnp

from gradwarden.config import ModelConfig


class CausalSelfAttention(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x):
        cfg = self.cfg
        b, t, d = x.shape
        h = cfg.n_heads
        hd = d // h

        qkv = nn.Dense(3 * d, use_bias=False, name="qkv")(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(b, t, h, hd)
        k = k.reshape(b, t, h, hd)
        v = v.reshape(b, t, h, hd)

        # have to build these even when the flag is off, otherwise flipping qk_norm
        # mid run changes the param tree and optax dies. ugly but w/e
        q_normed = nn.LayerNorm(use_bias=False, name="q_norm")(q)
        k_normed = nn.LayerNorm(use_bias=False, name="k_norm")(k)
        if cfg.qk_norm:
            q, k = q_normed, k_normed

        logits = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(hd)
        mask = jnp.tril(jnp.ones((t, t), dtype=bool))
        masked = jnp.where(mask, logits, -jnp.inf)
        probs = jax.nn.softmax(masked, axis=-1)

        out = jnp.einsum("bhts,bshd->bthd", probs, v).reshape(b, t, d)
        out = nn.Dense(d, use_bias=False, name="proj")(out)

        visible = jnp.arange(1, t + 1, dtype=jnp.float32)  # #keys that query i can see
        plogp = jnp.where(mask, probs * jnp.log(jnp.clip(probs, 1e-30)), 0.0)
        entropy = -plogp.sum(-1)  # [b, h, t]
        # pos 0 only sees itself, entropy=0 by construction, drop it or the norm nans
        norm_entropy = entropy[..., 1:] / jnp.log(visible[1:])
        stats = {
            "attn_logit_max": jnp.max(jnp.where(mask, jnp.abs(logits), 0.0)),
            "attn_entropy": jnp.mean(norm_entropy),
        }
        return out, stats


class MLP(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.cfg.d_ff, use_bias=False, name="up")(x)
        x = nn.gelu(x)
        return nn.Dense(self.cfg.d_model, use_bias=False, name="down")(x)


class Block(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x):
        attn_out, stats = CausalSelfAttention(self.cfg, name="attn")(nn.LayerNorm(name="ln1")(x))
        x = x + attn_out
        x = x + MLP(self.cfg, name="mlp")(nn.LayerNorm(name="ln2")(x))
        stats["act_rms"] = jnp.sqrt(jnp.mean(jnp.square(x)))
        return x, stats


class Transformer(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, tokens):
        cfg = self.cfg
        _, t = tokens.shape
        x = nn.Embed(cfg.vocab_size, cfg.d_model, name="embed")(tokens)
        pos = self.param("pos_embed", nn.initializers.normal(0.02), (cfg.seq_len, cfg.d_model))
        x = x + pos[:t]

        per_layer: dict[str, list] = {"attn_logit_max": [], "attn_entropy": [], "act_rms": []}
        for i in range(cfg.n_layers):
            x, stats = Block(cfg, name=f"block_{i}")(x)
            for key, value in stats.items():
                per_layer[key].append(value)

        x = nn.LayerNorm(name="ln_f")(x)
        logits = nn.Dense(cfg.vocab_size, use_bias=False, name="lm_head")(x)
        return logits, {k: jnp.stack(v) for k, v in per_layer.items()}


def init_params(cfg: ModelConfig, seed: int = 0):
    model = Transformer(cfg)
    dummy = jnp.zeros((1, cfg.seq_len), dtype=jnp.int32)
    return model.init(jax.random.PRNGKey(seed), dummy)["params"]


def count_params(params) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
