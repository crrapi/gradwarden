import json

import httpx

from gradwarden.agent import (
    GeminiBackend,
    LLMWarden,
    OllamaBackend,
    RulePolicy,
    ScriptedBackend,
    Toolbox,
    incident_brief,
    tool_schemas,
)
from gradwarden.agent.llm import Message, ToolCall
from gradwarden.session import Session
from gradwarden.store import Alert, RunStore


def make_store(n=60, blowup_at=40):
    store = RunStore()
    store.set_config({"model": {"qk_norm": False}, "train": {"lr": 3e-2, "clip_norm": None}})
    for s in range(n):
        logit = 8.0 if s < blowup_at else 8.0 * 1.5 ** (s - blowup_at + 1)
        store.add_record(
            {
                "step": s,
                "loss": 3.0 - 0.01 * s,
                "grad_norm": 0.3,
                "attn_logit_max": [logit, 6.0],
                "attn_entropy": [0.5, 0.6],
                "act_rms": [1.0, 1.0],
                "log_z_mean": 4.0,
                "grad_norm_by_module": {"block_0": 0.2},
                "update_ratio_by_module": {"block_0": 1e-3},
            }
        )
    alert = Alert(
        step=n - 1, kind="attn_logit_growth", severity="critical", value=500.0,
        threshold=200.0, message="attn logits big", layer=0,
    )  # fmt: skip
    store.add_alert(alert)
    return store, alert


def test_tools_basics():
    store, alert = make_store()
    tb = Toolbox(store=store, checkpoint_steps=[0, 25, 50])
    over = tb.get_run_overview()
    assert over["current_step"] == 59 and over["n_layers"] == 2
    win = tb.get_metric_window("attn_logit_max", 30, 59, layer=0)
    assert len(win["points"]) <= 24 and win["points"][0][0] == 30
    assert "error" in tb.get_metric_window("vibes")
    assert json.loads(tb.call("nope", {}))["error"]
    assert "error" in json.loads(tb.call("compare_steps", {"step_a": 1}))


def test_propose_patch_validation():
    store, _ = make_store()
    tb = Toolbox(store=store, checkpoint_steps=[25, 50])
    bad = tb.propose_patch({"lr": 99}, "x", "y", 30)
    assert not bad["accepted"] and tb.decision is None
    bad = tb.propose_patch({"qk_norm": True}, "x", "y", 10)  # no ckpt before 10
    assert not bad["accepted"]
    ok = tb.propose_patch({"qk_norm": True}, "attn_logit_growth", "y", 30)
    assert ok["accepted"] and tb.decision.rewind_to_step == 30


def test_schemas_cover_handlers():
    names = {s["name"] for s in tool_schemas()}
    assert names == set(Toolbox(store=RunStore()).handlers())


def test_brief_has_the_important_bits():
    store, alert = make_store()
    brief = incident_brief(alert, Toolbox(store=store, checkpoint_steps=[0, 25, 50]))
    assert "attn_logit_growth" in brief and "Checkpoint steps available: [0, 25, 50]" in brief
    assert "attn_logit_max:" in brief and "Per-layer at step 59" in brief


def test_rule_policy_rewinds_before_growth():
    store, alert = make_store()
    diag = RulePolicy().decide(alert, Toolbox(store=store, checkpoint_steps=[0, 25, 50]))
    assert diag.decision.changes == {"qk_norm": True}
    assert diag.decision.rewind_to_step == 25  # logits went bad at ~40, 25 is the last ckpt before


def test_llm_warden_with_scripted_model():
    store, alert = make_store()
    backend = ScriptedBackend.from_calls(
        [
            ("get_metric_window", {"metric": "attn_logit_max", "start_step": 20, "layer": 0}),
            (
                "propose_patch",
                {
                    "changes": {"qk_norm": True},
                    "cause": "attn_logit_growth",
                    "rationale": "layer 0 logits grow exponentially from step 40",
                    "rewind_to_step": 25,
                },
            ),
        ]
    )
    diag = LLMWarden(backend=backend).decide(
        alert, Toolbox(store=store, checkpoint_steps=[0, 25, 50])
    )
    assert diag.turns == 2 and not diag.fallback_used
    assert diag.decision.changes == {"qk_norm": True}
    # second call should have seen the tool result
    assert backend.seen[1][-1].role == "tool"


def test_llm_warden_falls_back_when_model_wont_decide():
    store, alert = make_store()
    backend = ScriptedBackend(script=[Message(role="assistant", content="hmm idk")] * 5)
    diag = LLMWarden(backend=backend, max_nudges=1).decide(
        alert, Toolbox(store=store, checkpoint_steps=[0, 25, 50])
    )
    assert diag.fallback_used and diag.decision.changes == {"qk_norm": True}


def test_llm_warden_retries_after_rejected_patch():
    store, alert = make_store()
    backend = ScriptedBackend.from_calls(
        [
            (
                "propose_patch",
                {"changes": {"lr": 50.0}, "cause": "x", "rationale": "y", "rewind_to_step": 25},
            ),
            (
                "propose_patch",
                {"changes": {"lr": 1e-3}, "cause": "x", "rationale": "y", "rewind_to_step": 25},
            ),
        ]
    )
    diag = LLMWarden(backend=backend).decide(alert, Toolbox(store=store, checkpoint_steps=[0, 25]))
    assert diag.turns == 2 and diag.decision.changes == {"lr": 1e-3}


def test_ollama_payload_and_parse():
    sent = {}

    def handler(request):
        sent.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "get_run_overview", "arguments": {}}}],
                },
                "prompt_eval_count": 100,
                "eval_count": 7,
                "total_duration": 2_000_000_000,
            },
        )

    be = OllamaBackend(host="http://box:11434", model="m", transport=httpx.MockTransport(handler))
    msgs = [
        Message(role="system", content="s"),
        Message(role="assistant", tool_calls=[ToolCall("get_run_overview", {})]),
        Message(role="tool", content="{}", tool_name="get_run_overview"),
    ]
    resp = be.chat(msgs, tool_schemas())
    assert sent["think"] is False and sent["stream"] is False
    assert "num_ctx" not in sent["options"] and "keep_alive" not in sent
    assert sent["messages"][2]["tool_name"] == "get_run_overview"
    assert sent["tools"][0]["type"] == "function"
    assert resp.message.tool_calls[0].name == "get_run_overview"
    assert resp.prompt_tokens == 100 and resp.seconds == 2.0


def test_gemini_payload_and_parse():
    sent = {}

    def handler(request):
        sent["key"] = request.headers.get("x-goog-api-key")
        sent["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "declare_no_action",
                                        "args": {"rationale": "fine"},
                                    }
                                }
                            ]
                        }
                    }
                ],
                "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 9},
            },
        )

    be = GeminiBackend(api_key="k", model="g", transport=httpx.MockTransport(handler))
    msgs = [
        Message(role="system", content="sys"),
        Message(role="user", content="hi"),
        Message(role="assistant", tool_calls=[ToolCall("a", {}), ToolCall("b", {})]),
        Message(role="tool", content='{"x": 1}', tool_name="a"),
        Message(role="tool", content="[1, 2]", tool_name="b"),
    ]
    resp = be.chat(msgs, tool_schemas())
    body = sent["body"]
    assert sent["key"] == "k"
    assert body["systemInstruction"]["parts"][0]["text"] == "sys"
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
    assert len(body["contents"][2]["parts"]) == 2  # both tool results in one turn
    assert body["contents"][2]["parts"][1]["functionResponse"]["response"] == {"result": [1, 2]}
    assert resp.message.tool_calls[0].arguments == {"rationale": "fine"}
    assert resp.completion_tokens == 9


def test_session_end_to_end_with_rules(tiny_cfg, data):
    # crank lr w/ no warmup so attn logits actually blow up, then let the rules fix it
    import dataclasses

    from gradwarden.detectors import DetectorSuite, PerLayerThreshold

    cfg = dataclasses.replace(
        tiny_cfg, train=dataclasses.replace(tiny_cfg.train, lr=0.1, warmup_steps=0, clip_norm=None)
    )
    suite = DetectorSuite(detectors=[PerLayerThreshold(metric="attn_logit_max", threshold=60.0)])
    sess = Session(cfg, data, RunStore(), policy=RulePolicy(), detectors=suite, max_interventions=1)
    store = sess.run(60)
    assert store.current_step == 59
    assert len(store.interventions) == 1 and store.interventions[0]["applied"]
    assert store.config["model"]["qk_norm"] is True
    assert store.abandoned  # the bad branch is kept for plotting
