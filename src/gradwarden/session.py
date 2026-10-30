"""the watched run loop. train -> detect -> if critical, pause, ask the policy,
apply patch, rewind to a checkpoint, keep going.

after a patch the session keeps an eye on the next `verify_window` steps. if an alert
from the same family fires again in that window the intervention gets marked
`refired`, otherwise `held`. the outcome goes into followups.jsonl and the next
diagnosis sees it, so the agent (or the rule table) can escalate instead of
proposing the same fix twice.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from gradwarden import notify
from gradwarden.agent.tools import Decision, Toolbox
from gradwarden.agent.warden import Diagnosis
from gradwarden.config import RunConfig
from gradwarden.data import CharDataset
from gradwarden.detectors import DetectorSuite
from gradwarden.store import Alert, RunStore
from gradwarden.train import Trainer

log = logging.getLogger("gradwarden")

# alerts that are really the same problem. qk_norm fixes all of attention, so if
# entropy collapses right after a logit growth fix thats a refire, not a new thing
FAMILIES = {
    "attn_logit_growth": "attention",
    "attn_logit_forecast": "attention",
    "attn_entropy_collapse": "attention",
    "output_logit_divergence": "output",
    "loss_spike": "optimizer",
    "grad_norm_spike": "optimizer",
    "non_finite": "optimizer",
    "adam_epsilon": "adam_eps",
    "val_divergence": "generalization",
}


def family(kind: str) -> str:
    return FAMILIES.get(kind, kind)


class Policy(Protocol):
    name: str

    def decide(self, alert: Alert, toolbox: Toolbox) -> Diagnosis: ...


@dataclass
class _Watch:
    index: int  # into store.interventions
    family: str
    patched_at: int  # step we restarted from
    until: int


@dataclass
class Session:
    cfg: RunConfig
    data: CharDataset
    store: RunStore
    policy: Policy | None = None
    max_interventions: int = 3
    detectors: DetectorSuite | None = None
    verify_window: int = 75
    early: bool = False
    ckpt_dir: Any = None  # Path, where Trainer mirrors ckpts to disk. None = memory only
    trainer: Trainer | None = None  # built in post_init unless resume() hands one in
    notifier: notify.Notifier = field(default_factory=notify.NullNotifier)
    # human in the loop. gets (alert, decision) before a patch is applied, False = veto.
    # None = apply everything the policy says (the default, its an on call agent)
    approve: Callable[[Alert, Decision], bool] | None = None
    diagnoses: list[Diagnosis] = field(default_factory=list)
    _watch: _Watch | None = None
    _pending: Alert | None = None

    def __post_init__(self):
        if self.trainer is None:
            self.trainer = Trainer(self.cfg, self.data, ckpt_dir=self.ckpt_dir)
        self.cfg = self.trainer.cfg
        if self.detectors is None:
            self.detectors = DetectorSuite.default(eps=self.cfg.train.eps, early=self.early)
        self.store.set_config(self.cfg.to_dict())

    @classmethod
    def resume(
        cls, run_dir, ckpt_dir, data: CharDataset, policy: Policy | None = None, **kw
    ) -> Session:
        """pick a dead run back up from its newest disk ckpt.

        records/alerts after that ckpt get dropped (no state backs them), and the
        detectors get the kept records replayed thru them so the ema baselines etc
        come back the same as if it never stopped. an open follow up watch is NOT
        restored, if it died mid window that intervention just has no outcome
        """
        store = RunStore.load(run_dir, writable=True)
        trainer = Trainer.resume(ckpt_dir, data)
        store.truncate(trainer.step)
        sess = cls(
            trainer.cfg, data, store, policy=policy, trainer=trainer, ckpt_dir=ckpt_dir, **kw
        )
        for r in store.records:
            sess.detectors.update(r)
        log.info("resumed %s at step %d", run_dir, trainer.step)
        return sess

    # follow up on the last patch

    def _close_watch(self, outcome: str, step: int, alert: Alert | None = None) -> None:
        w = self._watch
        if w is None:
            return
        entry = {
            "intervention": w.index,
            "family": w.family,
            "outcome": outcome,
            "patched_at": w.patched_at,
            "checked_at": step,
            "refired_by": alert.to_dict() if alert is not None else None,
        }
        self.store.add_followup(entry)
        self.store.interventions[w.index]["outcome"] = outcome
        self.notifier.notify(
            "followup", f"intervention {w.index} {outcome} (checked at step {step})", entry
        )
        log.info("intervention %d %s (checked at step %d)", w.index, outcome, step)
        self._watch = None

    def _check_watch(self, record: dict[str, Any], alerts: list[Alert]) -> None:
        w = self._watch
        if w is None:
            return
        for a in alerts:
            if a.severity == "critical" and family(a.kind) == w.family:
                self._close_watch("refired", record["step"], a)
                return
        if record["step"] >= w.until:
            self._close_watch("held", record["step"])

    # loop

    def _on_record(self, record: dict[str, Any]) -> bool:
        self.store.add_record(record)
        alerts = self.detectors.update(record)
        for alert in alerts:
            self.store.add_alert(alert)
            log.info("step %d ALERT %s: %s", alert.step, alert.kind, alert.message)
            if alert.severity == "critical":
                self.notifier.notify("alert", notify.describe_alert(alert), alert.to_dict())
        self._check_watch(record, alerts)
        critical = [a for a in alerts if a.severity == "critical"]
        if not critical or self.policy is None:
            return False
        if len(self.store.interventions) >= self.max_interventions:
            return False
        self._pending = critical[0]
        return True  # pause the inner trainer loop so we can intervene

    def _toolbox(self) -> Toolbox:
        return Toolbox(
            store=self.store, checkpoint_steps=[c.step for c in self.trainer.checkpoints]
        )

    def _record_intervention(self, entry: dict[str, Any]) -> None:
        self.store.add_intervention(entry)
        slim = {k: v for k, v in entry.items() if k not in ("transcript", "config_before")}
        self.notifier.notify("intervention", notify.describe_intervention(entry), slim)

    def _intervene(self, alert: Alert) -> None:
        if self._watch is not None:
            # a new critical alert of a diff family while still watching. the
            # old patch didnt cause it but we cant say it held either
            self._close_watch("superseded", alert.step, alert)
        toolbox = self._toolbox()
        log.info("asking %s about %s at step %d", self.policy.name, alert.kind, alert.step)
        diag = self.policy.decide(alert, toolbox)
        self.diagnoses.append(diag)
        entry = {"alert": alert.to_dict(), **diag.to_dict(), "transcript": diag.transcript}
        decision = diag.decision
        if decision is None or decision.action != "patch":
            log.info("no action: %s", decision.rationale if decision else "no decision")
            entry["applied"] = False
            self._record_intervention(entry)
            return
        if self.approve is not None and not self.approve(alert, decision):
            log.info("patch %s vetoed", decision.changes)
            entry.update(applied=False, rejected=True)
            self._record_intervention(entry)
            return
        before = self.trainer.cfg
        self.trainer.apply_patch(decision.changes)
        restored = self.trainer.rewind(decision.rewind_to_step)
        self.store.rewind(restored)
        self.detectors.reset_after(restored)
        self.store.set_config(self.trainer.cfg.to_dict())
        entry.update(
            applied=True,
            restored_step=restored,
            redo_steps=alert.step + 1 - restored,
            config_before=before.to_dict(),
        )
        self._record_intervention(entry)
        self._watch = _Watch(
            index=len(self.store.interventions) - 1,
            family=family(alert.kind),
            patched_at=restored,
            until=restored + self.verify_window,
        )
        log.info("applied %s, rewound to step %d (%s)", decision.changes, restored, decision.cause)

    def run(self, total_steps: int | None = None) -> RunStore:
        total = total_steps or self.cfg.train.total_steps
        while self.trainer.step < total:
            self._pending = None
            self.trainer.run(total - self.trainer.step, callback=self._on_record)
            if self._pending is not None:
                self._intervene(self._pending)
        if self._watch is not None:
            # ran out of steps before the window closed. still usefull to know
            self._close_watch("held_partial", self.store.current_step)
        return self.store
