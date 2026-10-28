"""named broken configs. each one is a known way to wreck a run at this scale so the
policies can be compared on more than just the one demo config.

all of them picked from probe runs (300-600 steps, seed 0), see docs/NOTES.md for
what fires and when. `--scenario name` on train/watch applies the overrides on top
of whatever else is on the cli
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from gradwarden.config import RunConfig


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    train: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)

    def apply(self, cfg: RunConfig) -> RunConfig:
        return RunConfig(
            model=dataclasses.replace(cfg.model, **self.model),
            train=dataclasses.replace(cfg.train, **self.train),
        )


SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in (
        Scenario(
            "logit_growth",
            "the readme demo. lr 3e-2 w/ warmup, no clip. attn logits grow until "
            "attention goes one hot, ~step 90",
            train={"lr": 3e-2, "warmup_steps": 100, "clip_norm": None},
        ),
        Scenario(
            "hot_start",
            "lr 3e-1, no warmup, no clip. everything fires in the first few steps "
            "incl. log Z divergence, then grad norm spikes later",
            train={"lr": 3e-1, "warmup_steps": 0, "clip_norm": None},
        ),
        Scenario(
            "qk_not_enough",
            "qk_norm already on but lr 1.0. logits still cross 200 so the first fix "
            "in the table is a no-op, tests escalation",
            train={"lr": 1.0, "warmup_steps": 0, "clip_norm": None},
            model={"qk_norm": True},
        ),
        Scenario(
            "big_eps",
            "adam eps 1e-4, grad rms is ~1e-3 so updates get damped. only a warn, "
            "nothing should get patched. here as a false positive check",
            train={"eps": 1e-4},
        ),
    )
}


def get(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError:
        raise ValueError(f"unknown scenario {name!r}, have {sorted(SCENARIOS)}") from None
