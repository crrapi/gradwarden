"""the warden. alert in, diagnosis + config patch out.

two policies, same interface: decide(alert, toolbox) -> Diagnosis

LLMWarden   builds an incident brief (alert, config, recent trajectories, per layer
            stats at the alert step) and lets a tool-calling model dig in. must
            finish on propose_patch or declare_no_action. the brief is usually
            enough to answer in 1 call; tools are there if it wants to dig.
RulePolicy  dumb alert->fix table. it's the baseline to compare against AND the
            fallback when the model wont decide (ollama timeout, garbage output, etc)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from gradwarden.agent.llm import ChatBackend, Message
from gradwarden.agent.tools import Decision, Toolbox, downsample, tool_schemas
from gradwarden.store import Alert

PLAYBOOK = """\
You are gradwarden, an on-call engineer for a Transformer pretraining run.
A cheap detector raised an alert. Decide whether the run needs a fix, and if so
which config change, and how far back to rewind.

Known failure modes and standard fixes:
- attn_logit_growth / attn_entropy_collapse: attention logits grow without bound
  and attention becomes one-hot (Dehghani et al. 2023, Zhai et al. 2023,
  Wortsman et al. 2023). Fix: qk_norm=true. Lowering lr also helps but costs speed.
- output_logit_divergence: the softmax normalizer log Z drifts far from 0
  (Wortsman et al. 2023). Fix: z_loss=1e-4.
- loss_spike / grad_norm_spike without the above: usually too high a learning rate
  or a bad batch. Fix: lower lr (about 0.5x) and/or set clip_norm=1.0; add warmup
  if warmup_steps is 0.
- non_finite: the run already diverged. Rewind well before the blow-up and lower lr.
- adam_epsilon: gradient RMS near Adam eps shrinks updates. Fix: lower eps (e.g. 1e-15).

Rules:
- Prefer the smallest change that addresses the root cause.
- Rewind to a step BEFORE the problem started (look at when the bad metric first
  left its normal range), using one of the listed checkpoint steps.
- If the alert is benign (a small blip that is already recovering), call
  declare_no_action.
- Be brief. Finish by calling propose_patch or declare_no_action.
"""

BRIEF_METRICS = ("loss", "grad_norm", "attn_logit_max", "attn_entropy", "log_z_mean")


def _fmt_series(points: list[tuple[int, float]]) -> str:
    return " ".join(f"{s}:{v:.3g}" for s, v in downsample(points, 12))


def incident_brief(alert: Alert, toolbox: Toolbox) -> str:
    store = toolbox.store
    lines = [
        f"ALERT at step {alert.step}: {alert.kind} [{alert.severity}] {alert.message}",
        "",
        "Config: " + json.dumps(store.config, separators=(",", ":")),
        f"Checkpoint steps available: {toolbox.checkpoint_steps}",
        "",
        "Trajectories (step:value; per-layer metrics are max over layers, "
        "attn_entropy is min over layers, normalized to [0,1]):",
    ]
    for metric in BRIEF_METRICS:
        if metric == "attn_entropy":
            pts = [(r["step"], min(r[metric])) for r in store.records if metric in r]
        else:
            pts = store.series(metric)
        if pts:
            lines.append(f"  {metric}: {_fmt_series(pts)}")
    rec = store.record_at(alert.step)
    if rec is not None:
        lines.append("")
        lines.append(f"Per-layer at step {alert.step}:")
        for key in ("attn_logit_max", "attn_entropy", "act_rms"):
            if key in rec:
                lines.append(f"  {key}: " + ", ".join(f"{v:.3g}" for v in rec[key]))
    return "\n".join(lines)


@dataclass
class Diagnosis:
    decision: Decision | None
    policy: str
    turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    fallback_used: bool = False
    transcript: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = self.decision
        return {
            "policy": self.policy,
            "action": d.action if d else None,
            "cause": d.cause if d else None,
            "changes": d.changes if d else None,
            "rewind_to_step": d.rewind_to_step if d else None,
            "rationale": d.rationale if d else None,
            "turns": self.turns,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "seconds": round(self.seconds, 1),
            "fallback_used": self.fallback_used,
        }


@dataclass
class RulePolicy:
    """alert -> patch lookup. baseline + fallback"""

    name: str = "rules"

    def _rewind_point(self, alert: Alert, toolbox: Toolbox, metric: str | None) -> int:
        target = alert.step
        if metric is not None:
            series = toolbox.store.series(metric)
            if metric == "attn_entropy":
                series = [(r["step"], min(r[metric])) for r in toolbox.store.records]
                healthy = [s for s, v in series if s < alert.step and v > 2 * alert.threshold]
            elif metric == "log_z_mean":
                healthy = [s for s, v in series if s < alert.step and abs(v) < alert.threshold / 2]
            else:
                healthy = [s for s, v in series if s < alert.step and v < alert.threshold / 2]
            if healthy:
                target = healthy[-1]
        eligible = [s for s in toolbox.checkpoint_steps if s <= target]
        return eligible[-1] if eligible else (toolbox.checkpoint_steps or [0])[0]

    def decide(self, alert: Alert, toolbox: Toolbox) -> Diagnosis:
        train_cfg = toolbox.store.config.get("train", {})
        lr = float(train_cfg.get("lr", 1e-3))
        kind = alert.kind
        if kind in ("attn_logit_growth", "attn_entropy_collapse"):
            metric = "attn_logit_max" if kind == "attn_logit_growth" else "attn_entropy"
            changes: dict[str, Any] = {"qk_norm": True}
            rewind = self._rewind_point(alert, toolbox, metric)
        elif kind == "output_logit_divergence":
            changes = {"z_loss": 1e-4}
            rewind = self._rewind_point(alert, toolbox, "log_z_mean")
        elif kind == "adam_epsilon":
            changes = {"eps": 1e-15}
            rewind = self._rewind_point(alert, toolbox, None)
        else:  # spikes and non-finite values
            changes = {"lr": lr * (0.3 if kind == "non_finite" else 0.5)}
            if train_cfg.get("clip_norm") is None:
                changes["clip_norm"] = 1.0
            rewind = self._rewind_point(alert, toolbox, None) - 1
            rewind = max(rewind, 0)
            eligible = [s for s in toolbox.checkpoint_steps if s <= rewind]
            rewind = eligible[-1] if eligible else 0
        result = toolbox.propose_patch(
            changes, cause=kind, rationale=f"rule table entry for {kind}", rewind_to_step=rewind
        )
        if not result["accepted"]:
            toolbox.declare_no_action(f"rule patch rejected: {result['problems']}")
        return Diagnosis(decision=toolbox.decision, policy=self.name)


@dataclass
class LLMWarden:
    backend: ChatBackend
    max_turns: int = 6
    max_nudges: int = 2
    fallback: RulePolicy | None = field(default_factory=RulePolicy)

    @property
    def name(self) -> str:
        return f"llm:{self.backend.name}"

    def decide(self, alert: Alert, toolbox: Toolbox) -> Diagnosis:
        toolbox.decision = None
        messages = [
            Message(role="system", content=PLAYBOOK),
            Message(role="user", content=incident_brief(alert, toolbox)),
        ]
        diag = Diagnosis(decision=None, policy=self.name)
        schemas = tool_schemas()
        nudges = 0
        t0 = time.perf_counter()
        error: str | None = None
        try:
            for _ in range(self.max_turns):
                resp = self.backend.chat(messages, schemas)
                diag.turns += 1
                diag.prompt_tokens += resp.prompt_tokens
                diag.completion_tokens += resp.completion_tokens
                msg = resp.message
                messages.append(msg)
                diag.transcript.append(
                    {
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [
                            {"name": c.name, "arguments": c.arguments} for c in msg.tool_calls
                        ],
                    }
                )
                if not msg.tool_calls:
                    if nudges >= self.max_nudges:
                        break
                    nudges += 1
                    messages.append(
                        Message(
                            role="user",
                            content="Finish now by calling propose_patch or declare_no_action.",
                        )
                    )
                    continue
                for call in msg.tool_calls:
                    result = toolbox.call(call.name, call.arguments)
                    messages.append(Message(role="tool", content=result, tool_name=call.name))
                    diag.transcript.append({"role": "tool", "name": call.name, "content": result})
                if toolbox.decision is not None:
                    break
        except Exception as e:  # ollama timeout, 500, garbage json, whatever
            error = f"{type(e).__name__}: {e}"
            diag.transcript.append({"role": "error", "content": error})
        diag.seconds = time.perf_counter() - t0
        diag.decision = toolbox.decision
        if diag.decision is None and self.fallback is not None:
            fb = self.fallback.decide(alert, toolbox)
            diag.decision = fb.decision
            diag.fallback_used = True
        return diag
