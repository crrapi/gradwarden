"""training. jitted step that also returns per-module grad/update stats, plus a
Trainer that keeps in-memory checkpoints so a run can be patched and rewound.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from gradwarden.config import RunConfig, TrainConfig, apply_patch
from gradwarden.data import CharDataset
from gradwarden.model import Transformer, init_params

NO_CLIP = 1e12  # effectively "off" but keeps the opt state tree identical so
#                 patches switching clip on/off dont crash optax


def lr_schedule(tc: TrainConfig) -> optax.Schedule:
    end = tc.lr * tc.min_lr_ratio
    decay_steps = max(tc.total_steps - tc.warmup_steps, 1)
    cosine = optax.cosine_decay_schedule(tc.lr, decay_steps, alpha=end / tc.lr)
    if tc.warmup_steps == 0:
        return cosine
    warmup = optax.linear_schedule(0.0, tc.lr, tc.warmup_steps)
    return optax.join_schedules([warmup, cosine], [tc.warmup_steps])


def _decay_mask(params):
    # no weight decay on norms / biases / anything 1d. standard
    return jax.tree_util.tree_map(lambda p: p.ndim >= 2, params)


def make_optimizer(tc: TrainConfig) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(tc.clip_norm if tc.clip_norm is not None else NO_CLIP),
        optax.adamw(
            lr_schedule(tc),
            b1=tc.b1,
            b2=tc.b2,
            eps=tc.eps,
            weight_decay=tc.weight_decay,
            mask=_decay_mask,
        ),
    )


def _group_norms(tree) -> dict[str, jax.Array]:
    """l2 norm per top level module (embed, block_0, ..., lm_head)"""
    return {
        name: optax.tree.norm(sub) if isinstance(sub, dict) else jnp.linalg.norm(sub)
        for name, sub in tree.items()
    }


def make_train_step(cfg: RunConfig, opt: optax.GradientTransformation):
    model = Transformer(cfg.model)
    z_weight = cfg.train.z_loss

    def loss_fn(params, inputs, targets):
        logits, stats = model.apply({"params": params}, inputs)
        logits = logits.astype(jnp.float32)
        log_z = jax.nn.logsumexp(logits, axis=-1)
        target_logits = jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
        ce = jnp.mean(log_z - target_logits)
        z_loss = jnp.mean(jnp.square(log_z))
        loss = ce + z_weight * z_loss
        aux = {
            "ce": ce,
            "log_z_mean": jnp.mean(log_z),
            "output_logit_max": jnp.max(jnp.abs(logits)),
            **stats,
        }
        return loss, aux

    @jax.jit
    def train_step(params, opt_state, inputs, targets):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, inputs, targets)
        updates, new_opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        grad_norm = optax.tree.norm(grads)
        n_params = sum(x.size for x in jax.tree_util.tree_leaves(grads))
        g_layers = _group_norms(grads)
        p_layers = _group_norms(params)
        u_layers = _group_norms(updates)
        metrics = {
            "loss": loss,
            "grad_norm": grad_norm,
            "grad_rms": grad_norm / math.sqrt(n_params),
            "param_norm": optax.tree.norm(params),
            "update_norm": optax.tree.norm(updates),
            "grad_norm_by_module": g_layers,
            "update_ratio_by_module": {k: u_layers[k] / (p_layers[k] + 1e-12) for k in u_layers},
            **aux,
        }
        return new_params, new_opt_state, metrics

    @jax.jit
    def eval_step(params, inputs, targets):
        loss, aux = loss_fn(params, inputs, targets)
        return aux["ce"]

    return train_step, eval_step


def to_host(metrics: dict[str, Any]) -> dict[str, Any]:
    """metrics pytree -> plain floats/lists for json"""
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            out[key] = {k: float(v) for k, v in value.items()}
        else:
            arr = np.asarray(value)
            out[key] = float(arr) if arr.ndim == 0 else [float(x) for x in arr]
    return out


@dataclass
class Checkpoint:
    step: int
    params: Any
    opt_state: Any
    config: RunConfig


@dataclass
class Trainer:
    """owns params + opt state, steps, and does patch/rewind"""

    cfg: RunConfig
    data: CharDataset
    max_checkpoints: int = 40
    step: int = 0
    params: Any = None
    opt_state: Any = None
    checkpoints: deque = field(default_factory=deque)
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if self.cfg.model.vocab_size != self.data.vocab_size:
            raise ValueError(
                f"model vocab_size={self.cfg.model.vocab_size} but dataset has "
                f"{self.data.vocab_size} symbols"
            )
        self.params = init_params(self.cfg.model, self.cfg.train.seed)
        self._build()
        self.opt_state = self.opt.init(self.params)
        self._save_checkpoint()

    def _build(self):
        self.opt = make_optimizer(self.cfg.train)
        self._train_step, self._eval_step = make_train_step(self.cfg, self.opt)

    def _save_checkpoint(self):
        # ring buffer in-memory, no disk. for a 1000 step run w/ ckpt_every=25
        # that's ~40 ckpts * ~1MB -> fine on cpu
        ckpt = Checkpoint(
            step=self.step,
            params=jax.device_get(self.params),
            opt_state=jax.device_get(self.opt_state),
            config=self.cfg,
        )
        self.checkpoints.append(ckpt)
        while len(self.checkpoints) > self.max_checkpoints:
            self.checkpoints.popleft()

    def train_step(self) -> dict[str, Any]:
        tc = self.cfg.train
        inputs, targets = self.data.batch(self.step, tc.batch_size, self.cfg.model.seq_len, tc.seed)
        t0 = time.perf_counter()
        self.params, self.opt_state, metrics = self._train_step(
            self.params, self.opt_state, inputs, targets
        )
        record = to_host(metrics)
        record["step"] = self.step
        record["lr"] = float(lr_schedule(tc)(self.step))
        record["step_time_s"] = time.perf_counter() - t0
        self.step += 1
        if tc.eval_every and self.step % tc.eval_every == 0:
            record["val_loss"] = self.evaluate()
        if tc.checkpoint_every and self.step % tc.checkpoint_every == 0 and _finite(record):
            self._save_checkpoint()
        self.history.append(record)
        return record

    def evaluate(self) -> float:
        tc = self.cfg.train
        losses = []
        for i in range(tc.eval_batches):
            x, y = self.data.batch(i, tc.batch_size, self.cfg.model.seq_len, tc.seed, split="val")
            losses.append(float(self._eval_step(self.params, x, y)))
        return float(np.mean(losses))

    def run(self, n_steps: int, callback: Callable[[dict[str, Any]], bool] | None = None) -> None:
        """train n_steps. callback returning True = pause (the session uses this)"""
        for _ in range(n_steps):
            record = self.train_step()
            if callback is not None and callback(record):
                return

    def latest_checkpoint_at_or_before(self, step: int) -> Checkpoint:
        candidates = [c for c in self.checkpoints if c.step <= step]
        if not candidates:
            raise ValueError(f"no checkpoint at or before step {step}")
        return candidates[-1]

    def rewind(self, to_step: int) -> int:
        """restore newest ckpt <= to_step, returns the step we actually landed on"""
        ckpt = self.latest_checkpoint_at_or_before(to_step)
        self.params = ckpt.params
        self.opt_state = ckpt.opt_state
        self.step = ckpt.step
        while self.checkpoints and self.checkpoints[-1].step > ckpt.step:
            self.checkpoints.pop()
        self.history = [r for r in self.history if r["step"] < ckpt.step]
        return ckpt.step

    def apply_patch(self, changes: dict[str, Any]) -> RunConfig:
        """swap in patched config. params + opt state carry over as is"""
        self.cfg = apply_patch(self.cfg, changes)
        self._build()
        return self.cfg


def _finite(record: dict[str, Any]) -> bool:
    return math.isfinite(record["loss"]) and math.isfinite(record["grad_norm"])
