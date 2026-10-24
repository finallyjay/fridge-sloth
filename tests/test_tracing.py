import io
import json
from collections.abc import Iterator
from typing import Any

import pytest
import sentry_sdk
from sentry_sdk.envelope import Envelope
from sentry_sdk.transport import Transport

import server
import tracing

SECRET_PHOTO = "U0VDUkVUX1BIT1RP"  # stands in for the base64 fridge photo


class CaptureTransport(Transport):
    def __init__(self, options: Any = None) -> None:
        super().__init__(options)
        self.sent: list[str] = []

    def capture_envelope(self, envelope: Envelope) -> None:
        self.sent.append(envelope.serialize().decode("utf-8", errors="replace"))


def ollama_reply(content: dict[str, Any]) -> io.BytesIO:
    return io.BytesIO(
        json.dumps(
            {
                "model": "gemma3:4b",
                "message": {"content": json.dumps(content)},
                "prompt_eval_count": 300,
                "eval_count": 50,
                "load_duration": 2_000_000,
                "prompt_eval_duration": 30_000_000,
                "eval_duration": 40_000_000,
            }
        ).encode()
    )


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch) -> Iterator[CaptureTransport]:
    capture = CaptureTransport()
    sentry_sdk.init(dsn="https://public@example.com/1", traces_sample_rate=1.0, transport=capture)
    monkeypatch.setattr(tracing, "enabled", True)
    yield capture
    sentry_sdk.get_client().close()
    sentry_sdk.init()  # back to a disabled client for the other tests


def test_disabled_by_default_is_a_no_op() -> None:
    assert tracing.enabled is False
    with tracing.agent("scan", "m") as agent, tracing.chat("vision", "m", 0) as chat:
        assert agent is None and chat is None
        tracing.record_response(chat, {"eval_count": 1})
        tracing.set_result(agent, items_seen=3)
    tracing.capture(RuntimeError("not sent"))


def test_init_needs_a_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    assert tracing.init() is False

    calls: list[dict[str, Any]] = []
    monkeypatch.setenv("SENTRY_DSN", "https://public@example.com/1")
    monkeypatch.setattr(sentry_sdk, "init", lambda **kw: calls.append(kw))
    monkeypatch.setattr(tracing, "enabled", False)
    assert tracing.init() is True
    assert tracing.enabled is True
    assert calls[0]["send_default_pii"] is False


def test_scan_sends_metadata_but_never_the_photo_or_prompts(
    transport: CaptureTransport, monkeypatch: pytest.MonkeyPatch
) -> None:
    replies = iter(
        [
            ollama_reply({"seen": [{"item": "eggs", "amount": "some"}]}),
            ollama_reply({"staples": [{"name": "eggs", "match": "yes", "seen_item": "eggs"}]}),
        ]
    )
    monkeypatch.setattr(server.urllib.request, "urlopen", lambda req, timeout: next(replies))

    result = server.scan_fridge(SECRET_PHOTO, ["eggs"])
    sentry_sdk.flush()
    sent = "\n".join(transport.sent)

    assert result["grocery_list"] == []
    assert "gen_ai.invoke_agent" in sent
    assert sent.count('"gen_ai.chat"') >= 2
    for expected in ('"vision"', '"matching"', "gen_ai.usage.input_tokens", "items_seen"):
        assert expected in sent
    # The privacy promise: no photo, no prompts, no answers.
    assert SECRET_PHOTO not in sent
    assert "inside of a fridge" not in sent
    assert '"seen_item"' not in sent
