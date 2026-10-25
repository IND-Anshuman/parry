"""Test fixtures: an in-process mock LLM server speaking the OpenAI and
Anthropic wire formats (JSON and SSE), plus SDK clients pointed at it.

No test talks to a real provider.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

import parry
from parry import _state

PROMPT_TOKENS = 10
COMPLETION_TOKENS = 5


@dataclass
class MockState:
    requests: list[dict[str, Any]] = field(default_factory=list)
    # Status codes to return (in order) before succeeding, e.g. [429, 500].
    failures: deque = field(default_factory=deque)
    retry_after: str | None = None
    prompt_tokens: int = PROMPT_TOKENS
    completion_tokens: int = COMPLETION_TOKENS
    reply: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def reset(self) -> None:
        with self.lock:
            self.requests.clear()
            self.failures.clear()
            self.retry_after = None
            self.prompt_tokens = PROMPT_TOKENS
            self.completion_tokens = COMPLETION_TOKENS
            self.reply = None

    def count(self, path: str | None = None) -> int:
        with self.lock:
            return sum(1 for r in self.requests if path is None or r["path"].endswith(path))


def _last_user_text(body: dict[str, Any]) -> str:
    if "input" in body and isinstance(body["input"], str):
        return body["input"]
    for msg in reversed(body.get("messages") or body.get("input") or []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


class Handler(BaseHTTPRequestHandler):
    state: MockState
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:  # keep test output clean
        pass

    def _send_json(
        self, status: int, payload: dict[str, Any], headers: dict[str, str] | None = None
    ) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _send_sse(self, events: list[tuple[str | None, Any]]) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()
        for name, data in events:
            chunk = ""
            if name:
                chunk += f"event: {name}\n"
            chunk += f"data: {data if isinstance(data, str) else json.dumps(data)}\n\n"
            self.wfile.write(chunk.encode())
            self.wfile.flush()
        self.close_connection = True

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("content-length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        st = self.state
        with st.lock:
            st.requests.append({"path": self.path, "body": body, "headers": dict(self.headers)})
            failure = st.failures.popleft() if st.failures else None
            retry_after = st.retry_after
        if failure is not None:
            headers = {"retry-after": retry_after} if retry_after else {}
            if "messages" in self.path and "chat" not in self.path:
                err = {
                    "type": "error",
                    "error": {"type": "rate_limit_error", "message": "slow down"},
                }
            else:
                err = {"error": {"message": "slow down", "type": "rate_limit", "code": None}}
            self._send_json(failure, err, headers)
            return
        text = st.reply if st.reply is not None else f"echo: {_last_user_text(body)}"
        path = self.path.split("?", 1)[0]
        if path.endswith("/chat/completions"):
            self._chat(body, text)
        elif path.endswith("/responses"):
            self._responses(body, text)
        elif path.endswith("/messages"):
            self._messages(body, text)
        elif path.endswith("/embeddings"):
            inputs = body.get("input")
            n = len(inputs) if isinstance(inputs, list) else 1
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": i, "embedding": [0.1] * 8}
                        for i in range(n)
                    ],
                    "model": body.get("model"),
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            )
        else:
            self._send_json(404, {"error": {"message": f"no route {self.path}"}})

    # --- OpenAI chat --------------------------------------------------------

    def _chat(self, body: dict[str, Any], text: str) -> None:
        st = self.state
        model = body.get("model")
        usage = {
            "prompt_tokens": st.prompt_tokens,
            "completion_tokens": st.completion_tokens,
            "total_tokens": st.prompt_tokens + st.completion_tokens,
        }
        if not body.get("stream"):
            self._send_json(
                200,
                {
                    "id": f"chatcmpl-{len(st.requests)}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                            "logprobs": None,
                        }
                    ],
                    "usage": usage,
                },
            )
            return
        base = {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
        }
        events: list[tuple[str | None, Any]] = [
            (
                None,
                {
                    **base,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None,
                        }
                    ],
                },
            )
        ]
        for word in text.split(" "):
            events.append(
                (
                    None,
                    {
                        **base,
                        "choices": [
                            {"index": 0, "delta": {"content": word + " "}, "finish_reason": None}
                        ],
                    },
                )
            )
        events.append(
            (None, {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        )
        if (body.get("stream_options") or {}).get("include_usage"):
            events.append((None, {**base, "choices": [], "usage": usage}))
        events.append((None, "[DONE]"))
        self._send_sse(events)

    # --- OpenAI responses ---------------------------------------------------

    def _response_obj(self, body: dict[str, Any], text: str) -> dict[str, Any]:
        st = self.state
        return {
            "id": f"resp_{len(st.requests)}",
            "object": "response",
            "created_at": 1,
            "model": body.get("model"),
            "status": "completed",
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": {},
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "parallel_tool_calls": True,
            "temperature": 1.0,
            "tool_choice": "auto",
            "tools": [],
            "top_p": 1.0,
            "usage": {
                "input_tokens": st.prompt_tokens,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": st.completion_tokens,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": st.prompt_tokens + st.completion_tokens,
            },
        }

    def _responses(self, body: dict[str, Any], text: str) -> None:
        resp = self._response_obj(body, text)
        if not body.get("stream"):
            self._send_json(200, resp)
            return
        created = {**resp, "status": "in_progress", "output": [], "usage": None}
        events: list[tuple[str | None, Any]] = [
            (
                "response.created",
                {"type": "response.created", "response": created, "sequence_number": 0},
            )
        ]
        for i, word in enumerate(text.split(" ")):
            events.append(
                (
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "item_id": "msg_1",
                        "output_index": 0,
                        "content_index": 0,
                        "delta": word + " ",
                        "logprobs": [],
                        "sequence_number": i + 1,
                    },
                )
            )
        events.append(
            (
                "response.completed",
                {"type": "response.completed", "response": resp, "sequence_number": 99},
            )
        )
        self._send_sse(events)

    # --- Anthropic messages -------------------------------------------------

    def _messages(self, body: dict[str, Any], text: str) -> None:
        st = self.state
        message = {
            "id": f"msg_{len(st.requests)}",
            "type": "message",
            "role": "assistant",
            "model": body.get("model"),
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": st.prompt_tokens, "output_tokens": st.completion_tokens},
        }
        if not body.get("stream"):
            self._send_json(200, message)
            return
        start = {
            **message,
            "content": [],
            "stop_reason": None,
            "usage": {"input_tokens": st.prompt_tokens, "output_tokens": 1},
        }
        events: list[tuple[str | None, Any]] = [
            ("message_start", {"type": "message_start", "message": start}),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
        ]
        for word in text.split(" "):
            events.append(
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": word + " "},
                    },
                )
            )
        events += [
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": st.completion_tokens},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
        self._send_sse(events)


@pytest.fixture(scope="session")
def _server():
    state = MockState()
    handler = type("BoundHandler", (Handler,), {"state": state})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, state
    server.shutdown()


@dataclass
class Mock:
    url: str
    state: MockState

    def count(self, path: str | None = None) -> int:
        return self.state.count(path)

    @property
    def requests(self) -> list[dict[str, Any]]:
        return self.state.requests


@pytest.fixture
def mock(_server) -> Mock:
    server, state = _server
    state.reset()
    host, port = server.server_address[:2]
    return Mock(url=f"http://{host}:{port}", state=state)


@pytest.fixture(autouse=True)
def _clean_parry():
    parry.uninstall()
    yield
    parry.uninstall()
    assert _state.context_guard.get() is None


@pytest.fixture
def openai_client(mock):
    openai = pytest.importorskip("openai")
    return openai.OpenAI(base_url=f"{mock.url}/v1", api_key="test-key", max_retries=0)


@pytest.fixture
def async_openai_client(mock):
    openai = pytest.importorskip("openai")
    return openai.AsyncOpenAI(base_url=f"{mock.url}/v1", api_key="test-key", max_retries=0)


@pytest.fixture
def anthropic_client(mock):
    anthropic = pytest.importorskip("anthropic")
    return anthropic.Anthropic(base_url=mock.url, api_key="test-key", max_retries=0)


@pytest.fixture
def async_anthropic_client(mock):
    anthropic = pytest.importorskip("anthropic")
    return anthropic.AsyncAnthropic(base_url=mock.url, api_key="test-key", max_retries=0)


def chat(client, content: str = "hello", model: str = "gpt-4o", **kwargs):
    return client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": content}], **kwargs
    )


def message(client, content: str = "hello", model: str = "claude-sonnet-5-5", **kwargs):
    return client.messages.create(
        model=model, max_tokens=64, messages=[{"role": "user", "content": content}], **kwargs
    )
