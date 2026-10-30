"""paging. its an on call agent, it should be able to wake someone up.

the session calls a Notifier on critical alerts, on every intervention and when a
followup closes. WebhookNotifier posts a small json body that slack + discord both
accept as is ({"text": ...} for slack, {"content": ...} for discord, we send both),
plus the structured event under "gradwarden" for anything custom.

failures to notify never kill the run, a dead webhook gets loged and thats it.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

log = logging.getLogger("gradwarden")


class Notifier(Protocol):
    def notify(self, event: str, text: str, payload: dict[str, Any]) -> None: ...


@dataclass
class NullNotifier:
    def notify(self, event, text, payload):
        pass


@dataclass
class MemoryNotifier:
    """keeps everything, for tests"""

    events: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)

    def notify(self, event, text, payload):
        self.events.append((event, text, payload))


@dataclass
class WebhookNotifier:
    url: str
    run_name: str = "run"
    timeout_s: float = 10.0
    # only page for these. "alert" is noisy on a bad run even w/ the cooldown
    events: tuple[str, ...] = ("alert", "intervention", "followup")
    transport: httpx.BaseTransport | None = None
    failures: int = 0

    def notify(self, event, text, payload):
        if event not in self.events:
            return
        msg = f"[gradwarden:{self.run_name}] {text}"
        body = {"text": msg, "content": msg, "gradwarden": {"event": event, **payload}}
        try:
            with httpx.Client(timeout=self.timeout_s, transport=self.transport) as client:
                client.post(self.url, json=body).raise_for_status()
        except httpx.HTTPError as e:
            self.failures += 1
            log.warning("webhook failed (%s): %s", event, e)


def from_env(run_name: str = "run") -> Notifier:
    url = os.environ.get("GRADWARDEN_WEBHOOK")
    return WebhookNotifier(url=url, run_name=run_name) if url else NullNotifier()


def describe_alert(alert) -> str:
    return f"step {alert.step} {alert.kind} [{alert.severity}]: {alert.message}"


def describe_intervention(entry: dict[str, Any]) -> str:
    a = entry["alert"]
    if not entry.get("applied"):
        why = "rejected by human" if entry.get("rejected") else "no action"
        return f"step {a['step']} {a['kind']}: {why} ({entry.get('rationale')})"
    return (
        f"step {a['step']} {a['kind']}: applied {entry.get('changes')} via "
        f"{entry.get('policy')}, rewound to {entry.get('restored_step')}"
    )
