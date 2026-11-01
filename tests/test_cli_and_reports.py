import json

import pytest

from gradwarden import cli
from gradwarden.agent import RulePolicy
from gradwarden.config import RunConfig
from gradwarden.detectors import DetectorSuite, PerLayerThreshold
from gradwarden.postmortem import alert_timeline, render
from gradwarden.scenarios import SCENARIOS, get
from gradwarden.session import Session
from gradwarden.store import Alert, RunStore


def test_scenarios_apply_and_validate():
    base = RunConfig()
    for s in SCENARIOS.values():
        cfg = s.apply(base)
        for k, v in s.train.items():
            assert getattr(cfg.train, k) == v
        for k, v in s.model.items():
            assert getattr(cfg.model, k) == v
    assert get("qk_not_enough").apply(base).model.qk_norm
    with pytest.raises(ValueError):
        get("nope")


def test_alert_timeline():
    store = RunStore()
    for s, v in ((5, 210.0), (30, 900.0), (55, 400.0)):
        store.add_alert(Alert(s, "attn_logit_growth", "critical", v, 200.0, "x", layer=0))
    store.add_alert(Alert(40, "attn_entropy_collapse", "critical", 0.01, 0.02, "x"))
    store.add_alert(Alert(41, "attn_entropy_collapse", "critical", 0.002, 0.02, "x"))
    rows = {r["kind"]: r for r in alert_timeline(store)}
    g = rows["attn_logit_growth"]
    assert (g["first"], g["last"], g["count"], g["worst_step"]) == (5, 55, 3, 30)
    assert rows["attn_entropy_collapse"]["worst"] == 0.002  # lower is worse there


def test_postmortem_from_real_session(tiny_cfg, data):
    import dataclasses

    cfg = dataclasses.replace(
        tiny_cfg,
        train=dataclasses.replace(tiny_cfg.train, lr=0.1, warmup_steps=0, clip_norm=None),
    )
    suite = DetectorSuite(detectors=[PerLayerThreshold(metric="attn_logit_max", threshold=60.0)])
    store = Session(
        cfg, data, RunStore(), policy=RulePolicy(), detectors=suite, verify_window=10
    ).run(60)
    md = render(store, "t")
    assert md.startswith("# postmortem: t")
    assert "qk_norm=True" in md and "outcome: **held**" in md
    assert "| attn_logit_growth | critical |" in md


def test_postmortem_empty_run():
    md = render(RunStore(), "empty")
    assert "boring run" in md and "## interventions" in md


def test_cli_scenarios_lists_all(capsys):
    assert cli.main(["scenarios"]) == 0
    out = capsys.readouterr().out
    assert all(name in out for name in SCENARIOS)


def test_cli_postmortem_on_saved_run(tiny_cfg, data, tmp_path, capsys):
    run = tmp_path / "r"
    Session(tiny_cfg, data, RunStore(run)).run(20)
    assert cli.main(["postmortem", str(run)]) == 0
    assert (run / "POSTMORTEM.md").exists()
    assert cli.main(["postmortem", str(run), "--stdout"]) == 0
    assert "# postmortem: r" in capsys.readouterr().out
    json.loads((run / "config.json").read_text())


def test_cli_config_applies_scenario():
    p = cli.argparse.ArgumentParser()
    cli._add_run_args(p)
    args = p.parse_args(["--run-dir", "x", "--scenario", "hot_start", "--lr", "1e-3"])
    cfg = cli._config(args, vocab_size=65)
    # scenario wins over the cli lr, thats the point of it
    assert cfg.train.lr == 0.3 and cfg.train.clip_norm is None and cfg.model.vocab_size == 65


def test_plots_dont_crash(tiny_cfg, data, tmp_path):
    import dataclasses

    from gradwarden.report import plot_bench, plot_run

    cfg = dataclasses.replace(
        tiny_cfg,
        train=dataclasses.replace(tiny_cfg.train, lr=0.1, warmup_steps=0, clip_norm=None),
    )
    suite = DetectorSuite(detectors=[PerLayerThreshold(metric="attn_logit_max", threshold=60.0)])
    store = Session(cfg, data, RunStore(), policy=RulePolicy(), detectors=suite).run(40)
    store.add_alert(Alert(5, "attn_logit_forecast", "warn", 30.0, 200.0, "x", layer=0))
    assert plot_run(store, tmp_path / "d.png").stat().st_size > 0
    rows = [
        {"scenario": "a", "policy": p, "final_val_loss": v, "interventions": n}
        for p, v, n in (("none", 2.4, 0), ("rules", 1.9, 1))
    ] + [{"scenario": "b", "policy": "none", "final_val_loss": None, "interventions": 0}]
    assert plot_bench(rows, tmp_path / "b.png").stat().st_size > 0
