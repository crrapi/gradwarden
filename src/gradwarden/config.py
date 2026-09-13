"""configs. frozen so each run is hashable and easy to diff.

the agent patches go thru apply_patch(). only PATCHABLE keys are allowed and every
value gets range checked. mostly so the llm cant set lr=50 or invent some new knob
that doesnt exist (both have happened in testing lol)
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 128
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    seq_len: int = 128
    # layernorm on q and k before the logits (dehghani 2023). see model.py — the
    # norm params always exist so this can be flipped mid run
    qk_norm: bool = False


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 3e-3
    warmup_steps: int = 100
    total_steps: int = 1000
    min_lr_ratio: float = 0.1
    batch_size: int = 32
    weight_decay: float = 0.1
    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    # global norm clipping, None = off
    clip_norm: float | None = 1.0
    # z-loss weight, mean(logZ^2), from palm (chowdhery 2022)
    z_loss: float = 0.0
    seed: int = 0
    eval_every: int = 50
    eval_batches: int = 4
    checkpoint_every: int = 25


@dataclass(frozen=True)
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunConfig:
        return cls(model=ModelConfig(**d["model"]), train=TrainConfig(**d["train"]))


# key -> (section, validator, range str for the error msg).
# if you add a knob here it also needs type info in agent/tools.py:tool_schemas
PATCHABLE: dict[str, tuple[str, Any, str]] = {
    "lr": ("train", lambda v: isinstance(v, (int, float)) and 1e-6 <= v <= 1.0, "1e-6..1.0"),
    "warmup_steps": ("train", lambda v: isinstance(v, int) and 0 <= v <= 10_000, "0..10000"),
    "clip_norm": (
        "train",
        lambda v: v is None or (isinstance(v, (int, float)) and 0.01 <= v <= 100.0),
        "null or 0.01..100",
    ),
    "z_loss": ("train", lambda v: isinstance(v, (int, float)) and 0.0 <= v <= 1e-2, "0..1e-2"),
    "eps": ("train", lambda v: isinstance(v, (int, float)) and 1e-20 <= v <= 1e-4, "1e-20..1e-4"),
    "weight_decay": (
        "train",
        lambda v: isinstance(v, (int, float)) and 0.0 <= v <= 1.0,
        "0..1",
    ),
    "qk_norm": ("model", lambda v: isinstance(v, bool), "true/false"),
}


class PatchError(ValueError):
    pass


def validate_patch(changes: dict[str, Any]) -> list[str]:
    """returns whats wrong w/ a patch, empty list = ok"""
    problems = []
    if not changes:
        problems.append("patch is empty")
    for key, value in changes.items():
        if key not in PATCHABLE:
            problems.append(f"{key!r} is not patchable; allowed: {sorted(PATCHABLE)}")
            continue
        _, ok, rng = PATCHABLE[key]
        if key == "warmup_steps" and isinstance(value, float) and value.is_integer():
            value = int(value)
        if not ok(value):
            problems.append(f"{key}={value!r} is out of range ({rng})")
    return problems


def apply_patch(cfg: RunConfig, changes: dict[str, Any]) -> RunConfig:
    problems = validate_patch(changes)
    if problems:
        raise PatchError("; ".join(problems))
    model_changes, train_changes = {}, {}
    for key, value in changes.items():
        section = PATCHABLE[key][0]
        if key == "warmup_steps":
            value = int(value)
        elif isinstance(value, int) and not isinstance(value, bool) and key != "warmup_steps":
            value = float(value)
        (model_changes if section == "model" else train_changes)[key] = value
    return RunConfig(
        model=dataclasses.replace(cfg.model, **model_changes),
        train=dataclasses.replace(cfg.train, **train_changes),
    )
