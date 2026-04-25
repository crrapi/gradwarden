"""run storage. metrics / alerts / interventions all as jsonl in the run dir.

rewinds dont delete anything — records past the rewind point get moved to an
`abandoned` segment, so the plots and the agent can still look at the bad branch.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PER_LAYER = ("attn_logit_max", "attn_entropy", "act_rms")
PER_MODULE = ("grad_norm_by_module", "update_ratio_by_module")


@dataclass
class Alert:
    step: int
    kind: str
    severity: str  # "warn" or "critical"
    value: float
    threshold: float
    message: str
    layer: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunStore:
    run_dir: Path | None = None
    records: list[dict[str, Any]] = field(default_factory=list)
    abandoned: list[list[dict[str, Any]]] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    interventions: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.run_dir is not None:
            self.run_dir = Path(self.run_dir)
            self.run_dir.mkdir(parents=True, exist_ok=True)

    # writing

    def _append(self, name: str, obj: dict[str, Any]) -> None:
        if self.run_dir is None:
            return
        with open(self.run_dir / name, "a") as f:
            f.write(json.dumps(obj) + "\n")

    def set_config(self, config: dict[str, Any]) -> None:
        self.config = config
        if self.run_dir is not None:
            (self.run_dir / "config.json").write_text(json.dumps(config, indent=2))

    def add_record(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        self._append("metrics.jsonl", record)

    def add_alert(self, alert: Alert) -> None:
        self.alerts.append(alert)
        self._append("alerts.jsonl", alert.to_dict())

    def add_intervention(self, intervention: dict[str, Any]) -> None:
        self.interventions.append(intervention)
        self._append("interventions.jsonl", intervention)

    def rewind(self, to_step: int) -> None:
        kept = [r for r in self.records if r["step"] < to_step]
        dropped = [r for r in self.records if r["step"] >= to_step]
        if dropped:
            self.abandoned.append(dropped)
        self.records = kept
        self._append("metrics.jsonl", {"rewind_to": to_step})

    # reading

    @property
    def current_step(self) -> int:
        return self.records[-1]["step"] if self.records else 0

    def series(self, name: str, layer: int | str | None = None, include_abandoned: bool = False):
        """[(step, value)]. per layer / per module metrics give the max unless layer is set"""
        rows = list(self.records)
        if include_abandoned:
            rows = [r for seg in self.abandoned for r in seg] + rows
            rows.sort(key=lambda r: r["step"])
        out = []
        for r in rows:
            if name not in r:
                continue
            value = r[name]
            if isinstance(value, list):
                value = max(value) if layer is None else value[int(layer)]
            elif isinstance(value, dict):
                value = max(value.values()) if layer is None else value[str(layer)]
            out.append((r["step"], value))
        return out

    def record_at(self, step: int) -> dict[str, Any] | None:
        for r in reversed(self.records):
            if r["step"] == step:
                return r
        for seg in self.abandoned:
            for r in seg:
                if r["step"] == step:
                    return r
        return None

    @classmethod
    def load(cls, run_dir: str | Path) -> RunStore:
        """load a run dir from disk (read only)"""
        run_dir = Path(run_dir)
        store = cls()
        store.run_dir = None  # IMPORTANT: dont append to files we're reading from
        cfg = run_dir / "config.json"
        if cfg.exists():
            store.config = json.loads(cfg.read_text())
        metrics = run_dir / "metrics.jsonl"
        if metrics.exists():
            for line in metrics.read_text().splitlines():
                obj = json.loads(line)
                if "rewind_to" in obj:
                    store.rewind(obj["rewind_to"])
                else:
                    store.records.append(obj)
        alerts = run_dir / "alerts.jsonl"
        if alerts.exists():
            store.alerts = [Alert(**json.loads(x)) for x in alerts.read_text().splitlines()]
        iv = run_dir / "interventions.jsonl"
        if iv.exists():
            store.interventions = [json.loads(x) for x in iv.read_text().splitlines()]
        return store
