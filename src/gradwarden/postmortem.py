"""incident writeup for a finished run, as markdown. its an on-call tool so it should
leave a postmortem behind like an actual on call persn would (well, a short one).

  gradwarden postmortem runs/demo-rules        -> runs/demo-rules/POSTMORTEM.md

everything in here is read off the run dir, no recompute.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from gradwarden.store import RunStore


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, dict):
        return ", ".join(f"`{k}={_fmt(x)}`" for k, x in v.items()) or "-"
    return "-" if v is None else str(v)


def _redo(iv: dict[str, Any]) -> int:
    # runs from before redo_steps existed only have alert step + restored step
    if not iv.get("applied"):
        return 0
    if "redo_steps" in iv:
        return iv["redo_steps"]
    return iv["alert"]["step"] + 1 - iv["restored_step"]


def _last(store: RunStore, key: str):
    for r in reversed(store.records):
        if key in r:
            return r[key]
    return None


def alert_timeline(store: RunStore) -> list[dict[str, Any]]:
    """one row per alert kind: first/last step, count, worst value"""
    by_kind: dict[str, list] = defaultdict(list)
    for a in store.alerts:
        by_kind[a.kind].append(a)
    rows = []
    for kind, alerts in by_kind.items():
        below = kind == "attn_entropy_collapse"  # smaller = worse for that one
        worst = min(alerts, key=lambda a: a.value) if below else max(alerts, key=lambda a: a.value)
        rows.append(
            {
                "kind": kind,
                "severity": alerts[0].severity,
                "first": alerts[0].step,
                "last": alerts[-1].step,
                "count": len(alerts),
                "worst": worst.value,
                "worst_step": worst.step,
            }
        )
    return sorted(rows, key=lambda r: r["first"])


def render(store: RunStore, name: str = "run") -> str:
    out: list[str] = [f"# postmortem: {name}", ""]
    final_val = _last(store, "val_loss")
    redo = sum(_redo(iv) for iv in store.interventions)
    seconds = sum(iv.get("seconds", 0.0) or 0.0 for iv in store.interventions)
    tokens = sum(
        (iv.get("prompt_tokens") or 0) + (iv.get("completion_tokens") or 0)
        for iv in store.interventions
    )
    out += [
        "## summary",
        "",
        "| | |",
        "|---|---|",
        f"| final step | {store.current_step} |",
        f"| final train loss | {_fmt(_last(store, 'loss'))} |",
        f"| final val loss | {_fmt(final_val)} |",
        f"| alerts | {len(store.alerts)} |",
        f"| interventions | {len(store.interventions)} "
        f"({sum(1 for i in store.interventions if i.get('applied'))} applied) |",
        f"| steps redone after rewinds | {redo} |",
        f"| abandoned branches | {len(store.abandoned)} |",
        f"| time in diagnosis | {seconds:.1f}s |",
        f"| llm tokens | {tokens} |",
        "",
    ]

    rows = alert_timeline(store)
    out += ["## alerts", ""]
    if not rows:
        out += ["none. boring run, good run", ""]
    else:
        out += [
            "| kind | severity | first | last | count | worst (step) |",
            "|---|---|---|---|---|---|",
        ]
        for r in rows:
            out.append(
                f"| {r['kind']} | {r['severity']} | {r['first']} | {r['last']} | {r['count']} "
                f"| {_fmt(r['worst'])} ({r['worst_step']}) |"
            )
        out.append("")

    out += ["## interventions", ""]
    if not store.interventions:
        out += ["none", ""]
    for i, iv in enumerate(store.interventions):
        alert = iv.get("alert", {})
        out += [
            f"### {i + 1}. step {alert.get('step')}: {alert.get('kind')}",
            "",
            f"- alert: {alert.get('message', '-')}",
            f"- policy: `{iv.get('policy')}`"
            + (" (**fell back to rules**)" if iv.get("fallback_used") else ""),
            f"- action: {iv.get('action')}, changes: {_fmt(iv.get('changes'))}",
        ]
        if iv.get("applied"):
            out.append(
                f"- rewound to step {iv.get('restored_step')} (asked for "
                f"{iv.get('rewind_to_step')}), redid {_redo(iv)} steps"
            )
        out += [
            f"- outcome: **{iv.get('outcome', 'unknown')}**",
            f"- cost: {iv.get('turns', 0)} turns, {iv.get('seconds', 0)}s, "
            f"{iv.get('prompt_tokens', 0)}+{iv.get('completion_tokens', 0)} tokens",
            "",
            "> " + (iv.get("rationale") or "-").replace("\n", "\n> "),
            "",
        ]
        for f in store.followups:
            if f["intervention"] == i and f.get("refired_by"):
                rb = f["refired_by"]
                out += [f"came back at step {rb['step']}: {rb['message']}", ""]

    out += ["## final config", "", "```json", json.dumps(store.config, indent=2), "```", ""]
    return "\n".join(out)
