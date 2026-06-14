"""chat backends w/ tool calling. ollama (local or lan), gemini, and a scripted
fake for tests.

one internal format (Message / ToolCall) that both backends get converted to/from,
and both return LLMResponse w/ token counts so each diagnosis has a cost.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str | None = None


@dataclass
class Message:
    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_name: str | None = None


@dataclass
class LLMResponse:
    message: Message
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0


class ChatBackend(Protocol):
    name: str

    def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> LLMResponse: ...


# Ollama


@dataclass
class OllamaBackend:
    """ollama /api/chat w/ native tool calls.

    host from OLLAMA_HOST (default localhost:11434), model from GRADWARDEN_OLLAMA_MODEL.
    do NOT send num_ctx or keep_alive — changing either makes ollama reload the model
    and stalls other requests on the host, learned that one the hard way.
    thinking off, these are short turns and it just wastes time.
    """

    model: str = field(
        default_factory=lambda: os.environ.get("GRADWARDEN_OLLAMA_MODEL", "qwen3.5:9b-text")
    )
    host: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    )
    timeout_s: float = 600.0
    temperature: float = 0.0
    think: bool = False
    transport: httpx.BaseTransport | None = None
    name: str = "ollama"

    def _base(self) -> str:
        host = self.host if "://" in self.host else f"http://{self.host}"
        return host.rstrip("/")

    def payload(self, messages: list[Message], tools: list[dict[str, Any]]) -> dict[str, Any]:
        out = []
        for m in messages:
            msg: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {"function": {"name": c.name, "arguments": c.arguments}} for c in m.tool_calls
                ]
            if m.role == "tool" and m.tool_name:
                msg["tool_name"] = m.tool_name
            out.append(msg)
        return {
            "model": self.model,
            "messages": out,
            "tools": [{"type": "function", "function": t} for t in tools],
            "stream": False,
            "think": self.think,
            "options": {"temperature": self.temperature},
        }

    def chat(self, messages, tools):
        with httpx.Client(timeout=self.timeout_s, transport=self.transport) as client:
            resp = client.post(f"{self._base()}/api/chat", json=self.payload(messages, tools))
            resp.raise_for_status()
            data = resp.json()
        raw = data.get("message", {})
        calls = []
        for c in raw.get("tool_calls") or []:
            fn = c.get("function", {})
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)
            calls.append(ToolCall(name=fn["name"], arguments=args, id=c.get("id")))
        return LLMResponse(
            message=Message(role="assistant", content=raw.get("content", ""), tool_calls=calls),
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            seconds=data.get("total_duration", 0) / 1e9,
        )


# Gemini


@dataclass
class GeminiBackend:
    """gemini generateContent rest api w/ function calling. needs GEMINI_API_KEY,
    model from GRADWARDEN_GEMINI_MODEL
    """

    model: str = field(
        default_factory=lambda: os.environ.get("GRADWARDEN_GEMINI_MODEL", "gemini-2.5-flash")
    )
    api_key: str | None = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY"))
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    timeout_s: float = 120.0
    temperature: float = 0.0
    transport: httpx.BaseTransport | None = None
    name: str = "gemini"

    def payload(self, messages: list[Message], tools: list[dict[str, Any]]) -> dict[str, Any]:
        # gemini wants system separate (systemInstruction), and tool results as
        # functionResponse parts under a user turn. not the openai shape.
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        contents: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                continue
            if m.role == "user":
                contents.append({"role": "user", "parts": [{"text": m.content}]})
            elif m.role == "assistant":
                parts: list[dict[str, Any]] = []
                if m.content:
                    parts.append({"text": m.content})
                for c in m.tool_calls:
                    parts.append({"functionCall": {"name": c.name, "args": c.arguments}})
                contents.append({"role": "model", "parts": parts or [{"text": ""}]})
            elif m.role == "tool":
                try:
                    response = json.loads(m.content)
                except json.JSONDecodeError:
                    response = {"result": m.content}
                if not isinstance(response, dict):
                    response = {"result": response}
                part = {"functionResponse": {"name": m.tool_name, "response": response}}
                # back to back tool results have to live in the same user turn or
                # gemini complains about role interleaving
                if (
                    contents
                    and contents[-1]["role"] == "user"
                    and "functionResponse" in (contents[-1]["parts"][0])
                ):
                    contents[-1]["parts"].append(part)
                else:
                    contents.append({"role": "user", "parts": [part]})
        body: dict[str, Any] = {
            "contents": contents,
            "tools": [{"functionDeclarations": tools}],
            "generationConfig": {"temperature": self.temperature},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    def chat(self, messages, tools):
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        url = f"{self.base_url}/models/{self.model}:generateContent"
        with httpx.Client(timeout=self.timeout_s, transport=self.transport) as client:
            resp = client.post(
                url, json=self.payload(messages, tools), headers={"x-goog-api-key": self.api_key}
            )
            resp.raise_for_status()
            data = resp.json()
        parts = data["candidates"][0]["content"].get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        calls = [
            ToolCall(name=p["functionCall"]["name"], arguments=p["functionCall"].get("args", {}))
            for p in parts
            if "functionCall" in p
        ]
        usage = data.get("usageMetadata", {})
        return LLMResponse(
            message=Message(role="assistant", content=text, tool_calls=calls),
            prompt_tokens=usage.get("promptTokenCount", 0),
            completion_tokens=usage.get("candidatesTokenCount", 0),
        )


# Scripted (tests and offline demos)


@dataclass
class ScriptedBackend:
    """replays canned assistant msgs, records what it got sent. for tests"""

    script: list[Message]
    name: str = "scripted"
    seen: list[list[Message]] = field(default_factory=list)

    @classmethod
    def from_calls(cls, calls: Iterable[tuple[str, dict[str, Any]]]) -> ScriptedBackend:
        return cls(
            script=[
                Message(role="assistant", tool_calls=[ToolCall(name=n, arguments=a)])
                for n, a in calls
            ]
        )

    def chat(self, messages, tools):
        self.seen.append(list(messages))
        if not self.script:
            return LLMResponse(message=Message(role="assistant", content="(script exhausted)"))
        return LLMResponse(message=self.script.pop(0))


def make_backend(name: str) -> ChatBackend:
    if name == "ollama":
        return OllamaBackend()
    if name == "gemini":
        return GeminiBackend()
    raise ValueError(f"unknown backend {name!r} (use ollama or gemini)")
