import dataclasses

from gradwarden.agent import RulePolicy, Toolbox, incident_brief
from gradwarden.detectors import DetectorSuite, PerLayerThreshold
from gradwarden.session import Session, family
from gradwarden.store import Alert, RunStore


def hot(cfg):
    # same trick as the e2e test, lr high + no warmup so logits actually blow up
    return dataclasses.replace(
        cfg, train=dataclasses.replace(cfg.train, lr=0.1, warmup_steps=0, clip_norm=None)
    )


def test_families():
    assert family("attn_logit_forecast") == family("attn_entropy_collapse") == "attention"
    assert family("loss_spike") == family("non_finite")
    assert family("made_up") == "made_up"


def test_patch_that_holds(tiny_cfg, data):
    suite = DetectorSuite(detectors=[PerLayerThreshold(metric="attn_logit_max", threshold=60.0)])
    sess = Session(
        hot(tiny_cfg), data, RunStore(), policy=RulePolicy(), detectors=suite, verify_window=15
    )
    store = sess.run(60)
    assert store.followups and store.followups[0]["outcome"] == "held"
    assert store.interventions[0]["outcome"] == "held"
    assert store.interventions[0]["redo_steps"] > 0
    assert store.interventions[0]["config_before"]["model"]["qk_norm"] is False


def test_patch_that_refires_then_escalates(tiny_cfg, data):
    # threshold so low that qk_norm cant keep logits under it -> refire, and the
    # second rule decision should go after lr instead of qk_norm again
    suite = DetectorSuite(
        detectors=[PerLayerThreshold(metric="attn_logit_max", threshold=3.0)], cooldown=1
    )
    sess = Session(
        hot(tiny_cfg),
        data,
        RunStore(),
        policy=RulePolicy(),
        detectors=suite,
        max_interventions=2,
        verify_window=50,
    )
    store = sess.run(40)
    assert store.interventions[0]["changes"] == {"qk_norm": True}
    assert store.interventions[0]["outcome"] == "refired"
    second = store.interventions[1]["changes"]
    assert "qk_norm" not in second and second["lr"] < 0.1


def test_brief_and_tool_show_history():
    store = RunStore()
    store.set_config({"model": {"qk_norm": True}, "train": {"lr": 1e-2}})
    store.add_record({"step": 0, "loss": 2.0, "attn_logit_max": [300.0]})
    store.add_intervention(
        {
            "alert": {"step": 0, "kind": "attn_logit_growth"},
            "policy": "rules",
            "changes": {"qk_norm": True},
            "applied": True,
            "restored_step": 0,
            "outcome": "refired",
        }
    )
    tb = Toolbox(store=store, checkpoint_steps=[0])
    hist = tb.get_intervention_history()["interventions"]
    assert hist[0]["outcome"] == "refired"
    alert = Alert(0, "attn_logit_growth", "critical", 300.0, 200.0, "x", layer=0)
    assert "outcome=refired" in incident_brief(alert, tb)


def test_rules_escalate_z_loss_and_wd():
    store = RunStore()
    store.set_config({"model": {}, "train": {"lr": 1e-2, "z_loss": 1e-4, "weight_decay": 0.1}})
    for s in range(5):
        store.add_record({"step": s, "loss": 2.0, "log_z_mean": 50.0})
    tb = Toolbox(store=store, checkpoint_steps=[0])
    a = Alert(4, "output_logit_divergence", "critical", 50.0, 30.0, "x")
    assert RulePolicy().decide(a, tb).decision.changes == {"z_loss": 1e-3}
    a = Alert(4, "val_divergence", "warn", 2.5, 2.1, "x")
    assert RulePolicy().decide(a, tb).decision.changes == {"weight_decay": 0.2}
