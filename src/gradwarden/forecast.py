"""early warning. the threshold detectors fire once a metric is already bad, by then
the rewind has to go back a ways and you redo a bunch of steps. but attn logit growth
isnt sudden, its ~exponential for 30-50 steps before it crosses 200 (see NOTES, demo
config goes 28 -> 45 -> 69 -> 96 -> 207 over steps 40..92).

so: fit log(value) ~ a + b*step over a trailing window, per layer. if the fit is good
and the line crosses the threshold within `horizon` steps, raise a forcast alert.

warn by default so it just shows up in alerts.jsonl. `--early` in the cli makes it
critical so the policy acts on it.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from gradwarden.detectors import Detector
from gradwarden.store import Alert


@dataclass
class Fit:
    slope: float
    intercept: float
    r2: float

    def predict(self, step: float) -> float:
        return self.intercept + self.slope * step

    def steps_until(self, level: float, now: float) -> float:
        """steps from `now` until the line reaches `level`. inf if it never does"""
        if self.slope <= 0:
            return math.inf
        return max((level - self.intercept) / self.slope - now, 0.0)


def linear_fit(xs: list[float], ys: list[float]) -> Fit | None:
    """plain least squares. numpy would be fine too but this runs every step on a
    ~20 point window and its not worth the array roundtrip"""
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys, strict=True))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return Fit(slope=slope, intercept=intercept, r2=r2)


@dataclass
class GrowthForecast(Detector):
    metric: str = "attn_logit_max"
    threshold: float = 200.0
    window: int = 20
    horizon: int = 40
    min_r2: float = 0.8
    # dont bother forecasting from tiny values. healthy runs sit at ~10 and a
    # 10 -> 14 drift over 100 steps would project a crossing eventually
    floor: float = 25.0
    severity: str = "warn"
    kind_name: str = "attn_logit_forecast"
    _buf: deque = field(default_factory=deque)

    @property
    def kind(self) -> str:  # type: ignore[override]
        return self.kind_name

    def update(self, record):
        values = record.get(self.metric)
        if values is None:
            return []
        if not isinstance(values, list):
            values = [values]
        step = record["step"]
        self._buf.append((step, list(values)))
        while len(self._buf) > self.window:
            self._buf.popleft()
        if len(self._buf) < self.window:
            return []

        worst: tuple[float, int, Fit, float] | None = None  # (eta, layer, fit, value)
        for layer in range(len(values)):
            pts = [(s, v[layer]) for s, v in self._buf if layer < len(v)]
            if any(not math.isfinite(v) or v <= 0 for _, v in pts):
                continue
            current = pts[-1][1]
            if current < self.floor or current >= self.threshold:
                # under the floor = not interesting, over threshold = the real
                # detecor already has it
                continue
            fit = linear_fit([s for s, _ in pts], [math.log(v) for _, v in pts])
            if fit is None or fit.r2 < self.min_r2:
                continue
            eta = fit.steps_until(math.log(self.threshold), step)
            if eta <= self.horizon and (worst is None or eta < worst[0]):
                worst = (eta, layer, fit, current)
        if worst is None:
            return []
        eta, layer, fit, current = worst
        doubling = math.log(2) / fit.slope
        return [
            Alert(
                step=step,
                kind=self.kind,
                severity=self.severity,
                value=current,
                threshold=self.threshold,
                layer=layer,
                message=(
                    f"{self.metric} in layer {layer} is {current:.3g} and doubling every "
                    f"{doubling:.0f} steps (r2={fit.r2:.2f}), projected to cross "
                    f"{self.threshold:g} in ~{eta:.0f} steps"
                ),
            )
        ]

    def reset_after(self, step):
        self._buf = deque((s, v) for s, v in self._buf if s < step)
