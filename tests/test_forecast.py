import math

from gradwarden.detectors import DetectorSuite, ValDivergence
from gradwarden.forecast import GrowthForecast, linear_fit


def rec(step, logits, **kw):
    return {"step": step, "loss": 2.0, "grad_norm": 0.3, "attn_logit_max": logits, **kw}


def test_linear_fit_recovers_line():
    fit = linear_fit([0, 1, 2, 3, 4], [1, 3, 5, 7, 9])
    assert math.isclose(fit.slope, 2.0) and math.isclose(fit.intercept, 1.0)
    assert math.isclose(fit.r2, 1.0)
    assert linear_fit([1, 1, 1], [1, 2, 3]) is None  # vertical, no slope
    assert linear_fit([0, 1], [0, 1]) is None  # too few points


def test_forecast_fires_on_exponential_growth():
    det = GrowthForecast(window=10, horizon=40)
    fired = []
    for s in range(60):
        v = 10.0 * 1.08**s  # doubles every ~9 steps, crosses 200 around step 39
        fired += det.update(rec(s, [5.0, v]))
    assert fired, "should warn before the threshold"
    first = fired[0]
    assert first.layer == 1 and first.value < 200 and first.step < 39
    assert first.kind == "attn_logit_forecast"


def test_forecast_quiet_on_flat_or_slow():
    det = GrowthForecast(window=10)
    for s in range(200):
        assert det.update(rec(s, [10.0 + 0.02 * s, 8.0])) == []  # healthy drift
    det = GrowthForecast(window=10, floor=0.0, horizon=20)
    for s in range(200):
        # above the floor but noisy + flat, r2 should kill it
        assert det.update(rec(s, [50.0 + (5 if s % 2 else -5), 8.0])) == []


def test_forecast_ignores_already_over_threshold():
    det = GrowthForecast(window=5)
    for s in range(20):
        assert det.update(rec(s, [300.0 * 1.1**s])) == []


def test_forecast_reset_after_rewind():
    det = GrowthForecast(window=10)
    for s in range(30):
        det.update(rec(s, [10.0]))
    det.reset_after(25)
    assert max(s for s, _ in det._buf) == 24


def test_val_divergence():
    det = ValDivergence(patience=2, margin=0.01)
    assert det.update({"step": 0, "val_loss": 2.0}) == []
    assert det.update({"step": 1, "loss": 1.0}) == []  # no val on this step
    assert det.update({"step": 50, "val_loss": 2.1}) == []
    a = det.update({"step": 100, "val_loss": 2.2})
    assert a and a[0].severity == "warn"
    det.reset_after(60)
    assert det._bad == 1


def test_default_suite_early_flag():
    kinds = {d.kind: d for d in DetectorSuite.default(early=True).detectors}
    assert kinds["attn_logit_forecast"].severity == "critical"
    kinds = {d.kind: d for d in DetectorSuite.default().detectors}
    assert kinds["attn_logit_forecast"].severity == "warn"
