"""the watched run loop. train -> detect -> if critical, pause, ask the policy,
apply patch, rewind to a checkpoint, keep going.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from gradwarden.agent.tools import Toolbox
from gradwarden.agent.warden import Diagnosis
from gradwarden.config import RunConfig
from gradwarden.data import CharDataset
from gradwarden.detectors import DetectorSuite
from gradwarden.store import Alert, RunStore
from gradwarden.train import Trainer

log = logging.getLogger("gradwarden")


class Policy(Protocol):
    name: str

    def decide(self, alert: Alert, toolbox: Toolbox) -> Diagnosis: ...


@dataclass
class Session:
    cfg: RunConfig
    data: CharDataset
    store: RunStore
    policy: Policy | None = None
    max_interventions: int = 3
    detectors: DetectorSuite | None = None
    trainer: Trainer = field(init=False)
    diagnoses: list[Diagnosis] = field(default_factory=list)

    def __post_init__(self):
        self.trainer = Trainer(self.cfg, self.data)
        if self.detectors is None:
            self.detectors = DetectorSuite.default(eps=self.cfg.train.eps)
        self.store.set_config(self.cfg.to_dict())

    def _on_record(self, record: dict[str, Any]) -> bool:
        self.store.add_record(record)
        alerts = self.detectors.update(record)
        for alert in alerts:
            self.store.add_alert(alert)
            log.info("step %d ALERT %s: %s", alert.step, alert.kind, alert.message)
        critical = [a for a in alerts if a.severity == "critical"]
        if not critical or self.policy is None:
            return False
        if len(self.store.interventions) >= self.max_interventions:
            return False
        self._pending = critical[0]
        return True  # pause the inner trainer loop so we can intervene

    def _intervene(self, alert: Alert) -> None:
        toolbox = Toolbox(
            store=self.store, checkpoint_steps=[c.step for c in self.trainer.checkpoints]
        )
        log.info("asking %s about %s at step %d", self.policy.name, alert.kind, alert.step)
        diag = self.policy.decide(alert, toolbox)
        self.diagnoses.append(diag)
        entry = {"alert": alert.to_dict(), **diag.to_dict(), "transcript": diag.transcript}
        decision = diag.decision
        if decision is None or decision.action != "patch":
            log.info("no action: %s", decision.rationale if decision else "no decision")
            entry["applied"] = False
            self.store.add_intervention(entry)
            return
        self.trainer.apply_patch(decision.changes)
        restored = self.trainer.rewind(decision.rewind_to_step)
        self.store.rewind(restored)
        self.detectors.reset_after(restored)
        self.store.set_config(self.trainer.cfg.to_dict())
        entry.update(applied=True, restored_step=restored)
        self.store.add_intervention(entry)
        log.info("applied %s, rewound to step %d (%s)", decision.changes, restored, decision.cause)

    def run(self, total_steps: int | None = None) -> RunStore:
        total = total_steps or self.cfg.train.total_steps
        while self.trainer.step < total:
            self._pending = None
            self.trainer.run(total - self.trainer.step, callback=self._on_record)
            if self._pending is not None:
                self._intervene(self._pending)
        return self.store
