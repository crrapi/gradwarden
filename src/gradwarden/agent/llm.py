"""chat backends w/ tool calling. ollama (local or lan), gemini, anything that
speaks the openai chat completions api (llama.cpp server, vllm, lm studio, openrouter..),
and a scripted fake for tests.

one internal format (Message / ToolCall) that both backends get converted to/from,
and both return LLMResponse w/ token counts so each diagnosis has a cost.
"""

from __future__ import annotations

import json
import os
import time
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
    # only openai style apis care about this. ollama/gemini match results by name
    tool_call_id: str | None = None


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


_RETRY_CODES = {429, 500, 503}
_GEMINI_FALLBACK = ["gemini-3.7-flash", "gemini-3.8-flash"]


@dataclass
class GeminiBackend:
    """gemini generateContent rest api w/ function calling. free aistudio key works
    out of the box: set GEMINI_API_KEY, done. GRADWARDEN_GEMINI_MODEL picks the first
    model; GRADWARDEN_GEMINI_MODELS overrides the whole fallback list (comma separated).
    retries 429/500/503 with backoff, falls through to the next model when one is out
    of free-tier daily cap.
    """

    model: str = field(
        default_factory=lambda: os.environ.get("GRADWARDEN_GEMINI_MODEL", "gemini-3.5-flash")
    )
    models: list[str] | None = None
    api_key: str | None = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY"))
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    timeout_s: float = 120.0
    temperature: float = 0.0
    retries: int = 3
    backoff_s: float = 1.0
    transport: httpx.BaseTransport | None = None
    name: str = "gemini"
    sleep: Any = field(default=time.sleep, repr=False)

    def __post_init__(self) -> None:
        if self.models is None:
            raw = os.environ.get("GRADWARDEN_GEMINI_MODELS")
            if raw:
                self.models = [m.strip() for m in raw.split(",") if m.strip()]
            else:
                self.models = [self.model] + [m for m in _GEMINI_FALLBACK if m != self.model]

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
        body = self.payload(messages, tools)
        headers = {"x-goog-api-key": self.api_key}
        order = self.models or [self.model]
        last: httpx.HTTPStatusError | None = None
        with httpx.Client(timeout=self.timeout_s, transport=self.transport) as client:
            for model in order:
                url = f"{self.base_url}/models/{model}:generateContent"
                for attempt in range(self.retries):
                    resp = client.post(url, json=body, headers=headers)
                    if resp.status_code < 400:
                        return self._parse(resp.json())
                    err = httpx.HTTPStatusError(
                        f"{resp.status_code} on {model}", request=resp.request, response=resp
                    )
                    if resp.status_code not in _RETRY_CODES:
                        raise err
                    last = err
                    if resp.status_code == 429 and _is_daily_cap(resp):
                        # free tier burned for the day on this model, don't waste retries
                        break
                    if attempt + 1 < self.retries:
                        self.sleep(self.backoff_s * (2**attempt))
        assert last is not None
        raise last

    def _parse(self, data: dict[str, Any]) -> LLMResponse:
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


def _is_daily_cap(resp: httpx.Response) -> bool:
    try:
        body = resp.json()
    except ValueError:
        return False
    for detail in (body.get("error", {}) or {}).get("details", []) or []:
        for v in detail.get("violations", []) or []:
            if "PerDay" in (v.get("quotaId") or ""):
                return True
    return False


# OpenAI compatible


@dataclass
class OpenAIBackend:
    """/v1/chat/completions w/ tools. base url from GRADWARDEN_OPENAI_BASE_URL, key
    from GRADWARDEN_OPENAI_API_KEY (or OPENAI_API_KEY). local servers usually dont
    need a key so its optional.

    tool results have to reference the id of the call they answer. ollama doesnt
    always give ids, so calls w/o one get call_0, call_1.. in order and tool msgs w/o
    a tool_call_id take the next one off the queue. works bc the warden always
    answers calls in the order they came in
    """

    model: str = field(
        default_factory=lambda: os.environ.get("GRADWARDEN_OPENAI_MODEL", "gpt-4o-mini")
    )
    base_url: str = field(
        default_factory=lambda: os.environ.get(
            "GRADWARDEN_OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
    )
    api_key: str | None = field(
        default_factory=lambda: (
            os.environ.get("GRADWARDEN_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        )
    )
    timeout_s: float = 300.0
    temperature: float = 0.0
    transport: httpx.BaseTransport | None = None
    name: str = "openai"

    def payload(self, messages: list[Message], tools: list[dict[str, Any]]) -> dict[str, Any]:
        out = []
        pending: list[str] = []
        n = 0
        for m in messages:
            if m.role == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.content or None}
                if m.tool_calls:
                    calls = []
                    for c in m.tool_calls:
                        cid = c.id or f"call_{n}"
                        n += 1
                        pending.append(cid)
                        calls.append(
                            {
                                "id": cid,
                                "type": "function",
                                "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                            }
                        )
                    msg["tool_calls"] = calls
                out.append(msg)
            elif m.role == "tool":
                cid = m.tool_call_id
                if cid is None:
                    cid = pending.pop(0) if pending else "call_unknown"
                elif cid in pending:
                    pending.remove(cid)
                out.append({"role": "tool", "tool_call_id": cid, "content": m.content})
            else:
                out.append({"role": m.role, "content": m.content})
        return {
            "model": self.model,
            "messages": out,
            "tools": [{"type": "function", "function": t} for t in tools],
            "temperature": self.temperature,
        }

    def chat(self, messages, tools):
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        with httpx.Client(timeout=self.timeout_s, transport=self.transport) as client:
            resp = client.post(url, json=self.payload(messages, tools), headers=headers)
            resp.raise_for_status()
            data = resp.json()
        raw = data["choices"][0]["message"]
        calls = []
        for c in raw.get("tool_calls") or []:
            fn = c.get("function", {})
            args = fn.get("arguments") or "{}"
            # some servers send a dict here even tho the spec says string
            if isinstance(args, str):
                args = json.loads(args) if args.strip() else {}
            calls.append(ToolCall(name=fn["name"], arguments=args, id=c.get("id")))
        usage = data.get("usage") or {}
        return LLMResponse(
            message=Message(role="assistant", content=raw.get("content") or "", tool_calls=calls),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
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
    if name == "openai":
        return OpenAIBackend()
    raise ValueError(f"unknown backend {name!r} (use ollama, gemini or openai)")
