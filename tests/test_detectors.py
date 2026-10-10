import math

import numpy as np

from gradwarden.detectors import (
    AbsThreshold,
    DetectorSuite,
    EMAZScore,
    NonFinite,
    PerLayerThreshold,
)


def rec(step, **kw):
    base = {
        "step": step,
        "loss": 2.0,
        "grad_norm": 0.3,
        "grad_rms": 1e-3,
        "attn_logit_max": [5.0, 5.0],
        "attn_entropy": [0.6, 0.6],
        "log_z_mean": 4.0,
    }
    base.update(kw)
    return base


def test_ema_catches_spike_and_not_smooth_decay():
    det = EMAZScore(metric="loss", warmup=20, z_threshold=5.0)
    rng = np.random.default_rng(0)
    for s in range(200):
        loss = 4.0 * math.exp(-s / 80) + 1.5 + rng.normal(0, 0.02)
        assert det.update(rec(s, loss=loss)) == []
    alerts = det.update(rec(200, loss=5.0))
    assert alerts and alerts[0].kind == "loss_spike"


def test_spike_doesnt_poison_baseline():
    det = EMAZScore(metric="loss", warmup=10, z_threshold=4.0)
    rng = np.random.default_rng(1)
    for s in range(50):
        det.update(rec(s, loss=2.0 + rng.normal(0, 0.01)))
    assert det.update(rec(50, loss=3.0))
    assert det.update(rec(51, loss=3.0))  # still anomalous, wasnt absorbed


def test_ema_reset_after_rewind():
    det = EMAZScore(metric="loss", warmup=5)
    for s in range(30):
        det.update(rec(s, loss=2.0 + 0.01 * (s % 3)))
    state_at_10 = [h for h in det._history if h[0] == 9][0]
    det.reset_after(10)
    assert det._mean == state_at_10[1]


def test_per_layer_threshold_reports_worst_layer():
    det = PerLayerThreshold(metric="attn_logit_max", threshold=200.0)
    assert det.update(rec(0, attn_logit_max=[10.0, 150.0])) == []
    a = det.update(rec(1, attn_logit_max=[250.0, 900.0]))[0]
    assert a.layer == 1 and a.value == 900.0


def test_entropy_collapse_below():
    det = PerLayerThreshold(metric="attn_entropy", threshold=0.02, below=True, kind_name="ec")
    assert det.update(rec(0)) == []
    a = det.update(rec(1, attn_entropy=[0.5, 0.004]))[0]
    assert a.kind == "ec" and a.layer == 1


def test_nonfinite_and_logz():
    assert NonFinite().update(rec(0, loss=float("nan")))[0].severity == "critical"
    assert AbsThreshold(threshold=30).update(rec(0, log_z_mean=-45.0))
    assert not AbsThreshold(threshold=30).update(rec(0, log_z_mean=12.0))


def test_suite_cooldown():
    suite = DetectorSuite.default()
    suite.cooldown = 10
    fired = []
    for s in range(30):
        fired += suite.update(rec(s, attn_logit_max=[500.0, 5.0]))
    steps = [a.step for a in fired if a.kind == "attn_logit_growth"]
    assert steps == [0, 10, 20]
