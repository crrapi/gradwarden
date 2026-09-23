"""cheap per step detectors.

the llm is slow (qwen 9b runs ~5 tok/s on the backend) so it does NOT watch the
stream. these do, and only wake the agent when something trips.

  NonFinite                nan/inf loss or grads. instant critical
  EMAZScore                loss / grad_norm spikes vs an ema mean+var. basically
                           zclip (kumar 2025). important: outliers do NOT get folded
                           back into the stats so a spike cant raise its own baseline
  attn_logit_growth        attn logits past a bound (dehghani 2023, wortsman 2023)
  attn_entropy_collapse    normalized entropy -> 0 (zhai 2023)
  output_logit_divergence  log Z drifts far from 0, happens before output logits
                           blow up (wortsman 2023). z-loss fixes it
  AdamEpsilon              grad rms near adam eps so updates shrink (wortsman 3.4)

thresholds all came from probe runs on tinyshakespeare, numbers in docs/NOTES.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from gradwarden.store import Alert


class Detector:
    kind = "detector"

    def update(self, record: dict[str, Any]) -> list[Alert]:
        raise NotImplementedError

    def reset_after(self, step: int) -> None:
        """called after a rewind. stateless detectors can just ignore"""


@dataclass
class NonFinite(Detector):
    kind = "non_finite"

    def update(self, record):
        bad = [k for k in ("loss", "grad_norm") if not math.isfinite(record.get(k, 0.0))]
        if not bad:
            return []
        return [
            Alert(
                step=record["step"],
                kind=self.kind,
                severity="critical",
                value=float("nan"),
                threshold=0.0,
                message=f"non-finite {', '.join(bad)}",
            )
        ]


@dataclass
class EMAZScore(Detector):
    metric: str = "loss"
    alpha: float = 0.05
    z_threshold: float = 4.0
    warmup: int = 50
    severity: str = "critical"
    # log space for stuff that moves multiplicatively (grad norm)
    log_space: bool = False
    _mean: float | None = None
    _var: float = 0.0
    _n: int = 0
    _history: list[tuple[int, float, float, int]] = field(default_factory=list)

    @property
    def kind(self) -> str:  # type: ignore[override]
        return f"{self.metric}_spike"

    def _x(self, value: float) -> float:
        return math.log(max(value, 1e-12)) if self.log_space else value

    def update(self, record):
        value = record.get(self.metric)
        if value is None or not math.isfinite(value):
            return []
        x = self._x(value)
        step = record["step"]
        alerts = []
        if self._mean is None:
            self._mean = x
        elif self._n >= self.warmup:
            std = math.sqrt(max(self._var, 1e-12))
            z = (x - self._mean) / std
            if z > self.z_threshold:
                alerts.append(
                    Alert(
                        step=step,
                        kind=self.kind,
                        severity=self.severity,
                        value=value,
                        threshold=self.z_threshold,
                        message=f"{self.metric}={value:.4g} is {z:.1f} sigma above its EMA",
                    )
                )
        if not alerts:
            delta = x - self._mean
            self._mean += self.alpha * delta
            self._var = (1 - self.alpha) * (self._var + self.alpha * delta * delta)
            self._n += 1
        self._history.append((step, self._mean, self._var, self._n))
        return alerts

    def reset_after(self, step):
        kept = [h for h in self._history if h[0] < step]
        self._history = kept
        if kept:
            _, self._mean, self._var, self._n = kept[-1]
        else:
            self._mean, self._var, self._n = None, 0.0, 0


@dataclass
class PerLayerThreshold(Detector):
    metric: str = "attn_logit_max"
    threshold: float = 200.0
    below: bool = False  # flip it, alert when value drops under threshold
    severity: str = "critical"
    kind_name: str = "attn_logit_growth"

    @property
    def kind(self) -> str:  # type: ignore[override]
        return self.kind_name

    def update(self, record):
        values = record.get(self.metric)
        if values is None:
            return []
        if not isinstance(values, list):
            values = [values]
        tripped = [
            (i, v)
            for i, v in enumerate(values)
            if math.isfinite(v) and ((v < self.threshold) if self.below else (v > self.threshold))
        ]
        if not tripped:
            return []
        layer, value = (
            min(tripped, key=lambda t: t[1]) if self.below else max(tripped, key=lambda t: t[1])
        )
        word = "below" if self.below else "above"
        return [
            Alert(
                step=record["step"],
                kind=self.kind,
                severity=self.severity,
                value=value,
                threshold=self.threshold,
                layer=layer,
                message=f"{self.metric}={value:.4g} in layer {layer} is {word} {self.threshold:g}",
            )
        ]


@dataclass
class AbsThreshold(Detector):
    metric: str = "log_z_mean"
    threshold: float = 30.0
    severity: str = "critical"
    kind_name: str = "output_logit_divergence"

    @property
    def kind(self) -> str:  # type: ignore[override]
        return self.kind_name

    def update(self, record):
        value = record.get(self.metric)
        if value is None or not math.isfinite(value) or abs(value) <= self.threshold:
            return []
        return [
            Alert(
                step=record["step"],
                kind=self.kind,
                severity=self.severity,
                value=value,
                threshold=self.threshold,
                message=f"|{self.metric}|={abs(value):.4g} exceeds {self.threshold:g}",
            )
        ]


@dataclass
class AdamEpsilon(Detector):
    eps: float = 1e-8
    ratio: float = 10.0
    kind = "adam_epsilon"

    def update(self, record):
        rms = record.get("grad_rms")
        if rms is None or not math.isfinite(rms) or rms > self.ratio * self.eps:
            return []
        return [
            Alert(
                step=record["step"],
                kind=self.kind,
                severity="warn",
                value=rms,
                threshold=self.ratio * self.eps,
                message=f"grad RMS {rms:.3g} is within {self.ratio:g}x of Adam eps {self.eps:g}",
            )
        ]


@dataclass
class DetectorSuite:
    detectors: list[Detector]
    # after one alert of a given kind, mute that kind for this many steps.
    # otherwise logit growth fires every step and the alerts jsonl becomes a mess
    cooldown: int = 25
    _last_fired: dict[str, int] = field(default_factory=dict)

    @classmethod
    def default(cls, eps: float = 1e-8) -> DetectorSuite:
        return cls(
            detectors=[
                NonFinite(),
                EMAZScore(metric="loss", z_threshold=5.0, warmup=50),
                EMAZScore(metric="grad_norm", z_threshold=5.0, warmup=50, log_space=True),
                PerLayerThreshold(
                    metric="attn_logit_max", threshold=200.0, kind_name="attn_logit_growth"
                ),
                PerLayerThreshold(
                    metric="attn_entropy",
                    threshold=0.02,
                    below=True,
                    kind_name="attn_entropy_collapse",
                ),
                AbsThreshold(metric="log_z_mean", threshold=30.0),
                AdamEpsilon(eps=eps),
            ]
        )

    def update(self, record: dict[str, Any]) -> list[Alert]:
        alerts = []
        for det in self.detectors:
            for alert in det.update(record):
                last = self._last_fired.get(alert.kind)
                if last is not None and alert.step - last < self.cooldown:
                    continue
                self._last_fired[alert.kind] = alert.step
                alerts.append(alert)
        return alerts

    def reset_after(self, step: int) -> None:
        for det in self.detectors:
            det.reset_after(step)
        self._last_fired = {k: v for k, v in self._last_fired.items() if v < step}
