"""tools the agent can call. everything reads the run store and returns small json.

keep outputs SHORT. on a 9b with queued requests every tool token is prompt time
on the next call. long series get downsampled to ~24 points.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from gradwarden.config import PATCHABLE, validate_patch
from gradwarden.store import PER_LAYER, PER_MODULE, RunStore

SCALAR_METRICS = (
    "loss",
    "ce",
    "val_loss",
    "grad_norm",
    "grad_rms",
    "param_norm",
    "update_norm",
    "log_z_mean",
    "output_logit_max",
    "lr",
)
ALL_METRICS = SCALAR_METRICS + PER_LAYER + PER_MODULE
MAX_POINTS = 24


def _r(x: float) -> float | str:
    if not math.isfinite(x):
        return str(x)
    return float(f"{x:.4g}")


def downsample(points: list[tuple[int, float]], max_points: int = MAX_POINTS):
    if len(points) <= max_points:
        return points
    stride = math.ceil(len(points) / max_points)
    picked = points[::stride]
    if picked[-1] != points[-1]:
        picked.append(points[-1])
    return picked


@dataclass
class Decision:
    action: str  # "patch" or "no_action"
    rationale: str
    cause: str = ""
    changes: dict[str, Any] = field(default_factory=dict)
    rewind_to_step: int | None = None


@dataclass
class Toolbox:
    store: RunStore
    # steps w/ an in memory ckpt. trainer fills this in when live, offline diagnose
    # synthesizes it from checkpoint_every. agent uses it to pick a rewind target
    checkpoint_steps: list[int] = field(default_factory=list)
    decision: Decision | None = None

    # tool implementations

    def get_run_overview(self) -> dict[str, Any]:
        recs = self.store.records
        last = recs[-1] if recs else {}
        finite = [r["loss"] for r in recs if math.isfinite(r["loss"])]
        return {
            "current_step": self.store.current_step,
            "last_loss": _r(last.get("loss", float("nan"))),
            "best_loss": _r(min(finite)) if finite else None,
            "n_layers": len(last.get("attn_logit_max", [])),
            "config": self.store.config,
            "alerts": [f"step {a.step}: {a.kind} ({a.message})" for a in self.store.alerts[-8:]],
            "interventions": len(self.store.interventions),
            "checkpoint_steps": self.checkpoint_steps[-10:],
        }

    def get_metric_window(
        self,
        metric: str,
        start_step: int = 0,
        end_step: int | None = None,
        layer: int | str | None = None,
    ) -> dict[str, Any]:
        if metric not in ALL_METRICS:
            return {"error": f"unknown metric {metric!r}", "valid": list(ALL_METRICS)}
        end = self.store.current_step if end_step is None else int(end_step)
        pts = [
            (s, v) for s, v in self.store.series(metric, layer=layer) if int(start_step) <= s <= end
        ]
        layer_note = "max over layers" if layer is None and metric in PER_LAYER + PER_MODULE else ""
        return {
            "metric": metric,
            "layer": layer,
            "note": layer_note,
            "points": [[s, _r(v)] for s, v in downsample(pts)],
        }

    def get_layer_breakdown(self, step: int | None = None) -> dict[str, Any]:
        step = self.store.current_step if step is None else int(step)
        rec = self.store.record_at(step)
        if rec is None:
            return {"error": f"no record at step {step}"}
        out: dict[str, Any] = {"step": step}
        for key in PER_LAYER:
            out[key] = [_r(v) for v in rec.get(key, [])]
        for key in PER_MODULE:
            out[key] = {k: _r(v) for k, v in rec.get(key, {}).items()}
        return out

    def compare_steps(self, step_a: int, step_b: int) -> dict[str, Any]:
        a, b = self.store.record_at(int(step_a)), self.store.record_at(int(step_b))
        if a is None or b is None:
            return {"error": "missing record for one of the steps"}
        out = {}
        for key in SCALAR_METRICS:
            if key in a and key in b:
                out[key] = [_r(a[key]), _r(b[key])]
        for key in PER_LAYER:
            if key in a and key in b:
                out[key] = [[_r(x) for x in a[key]], [_r(x) for x in b[key]]]
        return {"steps": [int(step_a), int(step_b)], "values": out}

    def propose_patch(
        self, changes: dict[str, Any], cause: str, rationale: str, rewind_to_step: int
    ) -> dict[str, Any]:
        problems = validate_patch(changes)
        rewind_to_step = int(rewind_to_step)
        if self.checkpoint_steps and not any(s <= rewind_to_step for s in self.checkpoint_steps):
            problems.append(f"no checkpoint at or before step {rewind_to_step}")
        if problems:
            return {"accepted": False, "problems": problems}
        self.decision = Decision(
            action="patch",
            cause=cause,
            rationale=rationale,
            changes=changes,
            rewind_to_step=rewind_to_step,
        )
        return {"accepted": True}

    def declare_no_action(self, rationale: str) -> dict[str, Any]:
        self.decision = Decision(action="no_action", rationale=rationale)
        return {"accepted": True}

    # dispatch

    def handlers(self) -> dict[str, Callable[..., dict[str, Any]]]:
        return {
            "get_run_overview": self.get_run_overview,
            "get_metric_window": self.get_metric_window,
            "get_layer_breakdown": self.get_layer_breakdown,
            "compare_steps": self.compare_steps,
            "propose_patch": self.propose_patch,
            "declare_no_action": self.declare_no_action,
        }

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        handler = self.handlers().get(name)
        if handler is None:
            return json.dumps({"error": f"unknown tool {name!r}"})
        try:
            return json.dumps(handler(**arguments))
        except TypeError as e:
            return json.dumps({"error": f"bad arguments for {name}: {e}"})


def tool_schemas() -> list[dict[str, Any]]:
    """function decls, same list for ollama + gemini"""
    patch_props = {
        k: {"description": f"allowed range {rng}"} for k, (_, _, rng) in PATCHABLE.items()
    }
    for k in ("lr", "z_loss", "eps", "weight_decay"):
        patch_props[k]["type"] = "number"
    patch_props["warmup_steps"]["type"] = "integer"
    patch_props["clip_norm"]["type"] = "number"
    patch_props["qk_norm"]["type"] = "boolean"
    return [
        {
            "name": "get_run_overview",
            "description": "Current step, losses, config, recent alerts and checkpoint steps.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "get_metric_window",
            "description": (
                "A downsampled [step, value] series for one metric. Per-layer metrics "
                "return the max over layers unless `layer` is given."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": list(ALL_METRICS)},
                    "start_step": {"type": "integer"},
                    "end_step": {"type": "integer"},
                    "layer": {"type": "integer"},
                },
                "required": ["metric"],
            },
        },
        {
            "name": "get_layer_breakdown",
            "description": "Every per-layer and per-module statistic at one step.",
            "parameters": {
                "type": "object",
                "properties": {"step": {"type": "integer"}},
            },
        },
        {
            "name": "compare_steps",
            "description": "Side-by-side scalar and per-layer values at two steps.",
            "parameters": {
                "type": "object",
                "properties": {"step_a": {"type": "integer"}, "step_b": {"type": "integer"}},
                "required": ["step_a", "step_b"],
            },
        },
        {
            "name": "propose_patch",
            "description": (
                "Final answer when the run needs a fix: config changes to apply, the root "
                "cause, and the step to rewind to (training restarts from the newest "
                "checkpoint at or before it)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "changes": {"type": "object", "properties": patch_props},
                    "cause": {
                        "type": "string",
                        "description": "Short failure-mode name, e.g. attn_logit_growth.",
                    },
                    "rationale": {"type": "string"},
                    "rewind_to_step": {"type": "integer"},
                },
                "required": ["changes", "cause", "rationale", "rewind_to_step"],
            },
        },
        {
            "name": "declare_no_action",
            "description": "Final answer if the alert is a false alarm and the run is fine.",
            "parameters": {
                "type": "object",
                "properties": {"rationale": {"type": "string"}},
                "required": ["rationale"],
            },
        },
    ]
