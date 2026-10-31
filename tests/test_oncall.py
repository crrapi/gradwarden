import dataclasses
import io
import json

import httpx

from gradwarden import cli
from gradwarden.agent import RulePolicy
from gradwarden.detectors import DetectorSuite, PerLayerThreshold
from gradwarden.live import _Follower, sparkline, tail
from gradwarden.notify import MemoryNotifier, NullNotifier, WebhookNotifier, from_env
from gradwarden.session import Session
from gradwarden.store import Alert, RunStore


def hot_session(cfg, data, store, **kw):
    cfg = dataclasses.replace(
        cfg, train=dataclasses.replace(cfg.train, lr=0.1, warmup_steps=0, clip_norm=None)
    )
    suite = DetectorSuite(detectors=[PerLayerThreshold(metric="attn_logit_max", threshold=60.0)])
    return Session(cfg, data, store, policy=RulePolicy(), detectors=suite, verify_window=10, **kw)


def test_session_pages_on_alert_patch_and_followup(tiny_cfg, data):
    pager = MemoryNotifier()
    hot_session(tiny_cfg, data, RunStore(), notifier=pager, max_interventions=1).run(60)
    events = [e for e, _, _ in pager.events]
    assert events[0] == "alert" and "intervention" in events and "followup" in events
    _, text, payload = next(e for e in pager.events if e[0] == "intervention")
    assert "qk_norm" in text and "transcript" not in payload


def test_veto_keeps_the_run_unpatched(tiny_cfg, data):
    seen = []

    def no(alert, decision):
        seen.append(decision.changes)
        return False

    store = hot_session(tiny_cfg, data, RunStore(), approve=no, max_interventions=2).run(60)
    assert seen and all(not iv["applied"] and iv["rejected"] for iv in store.interventions)
    assert store.config["model"]["qk_norm"] is False and not store.abandoned


def test_webhook_body_and_failures():
    got = []

    def ok(request):
        got.append(json.loads(request.content))
        return httpx.Response(200)

    wh = WebhookNotifier(url="http://hook", run_name="r", transport=httpx.MockTransport(ok))
    wh.notify("alert", "logits big", {"step": 3})
    wh.notify("debug", "ignored", {})
    assert len(got) == 1
    assert got[0]["text"] == got[0]["content"] == "[gradwarden:r] logits big"
    assert got[0]["gradwarden"] == {"event": "alert", "step": 3}

    dead = WebhookNotifier(
        url="http://hook", transport=httpx.MockTransport(lambda r: httpx.Response(500))
    )
    dead.notify("alert", "x", {})  # must not raise
    assert dead.failures == 1


def test_from_env(monkeypatch):
    monkeypatch.delenv("GRADWARDEN_WEBHOOK", raising=False)
    assert isinstance(from_env(), NullNotifier)
    monkeypatch.setenv("GRADWARDEN_WEBHOOK", "http://x")
    assert from_env("a").url == "http://x"


def test_sparkline():
    assert sparkline([]) == ""
    assert sparkline([5, 5, 5]) == "▁▁▁"
    s = sparkline([1, 10, 100, 1000])
    assert s[0] == "▁" and s[-1] == "█" and len(s) == 4


def test_follower_skips_half_written_line(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text('{"a": 1}\n{"a": ')
    f = _Follower(p)
    assert f.read() == [{"a": 1}]
    with open(p, "a") as fh:
        fh.write("2}\n")
    assert f.read() == [{"a": 2}]


def test_tail_once(tmp_path):
    store = RunStore(tmp_path)
    for s in range(25):
        store.add_record({"step": s, "loss": 2.0, "grad_norm": 0.3, "attn_logit_max": [s + 1.0]})
    store.add_alert(Alert(20, "attn_logit_growth", "critical", 21.0, 20.0, "too big", layer=0))
    store.rewind(10)
    buf = io.StringIO()
    n = tail(tmp_path, buf, every=10, follow=False)
    text = buf.getvalue()
    assert n == 3 and "rewound to step 10" in text and "!! critical attn_logit_growth" in text


def test_cli_compare(tiny_cfg, data, tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    Session(tiny_cfg, data, RunStore(a)).run(20)
    hot_session(tiny_cfg, data, RunStore(b), max_interventions=1).run(40)
    assert cli.main(["compare", str(a), str(b), "--out", str(tmp_path / "c.png")]) == 0
    out = capsys.readouterr().out
    assert "final config diff" in out and "train.lr" in out
    assert (tmp_path / "c.png").exists()
