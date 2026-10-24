import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

import server


def fake_gemma(*responses: Any) -> Any:
    """Return a stand-in for ask_gemma that replays responses and records each call."""
    calls: list[dict[str, Any]] = []
    queue = list(responses)

    def ask(message: dict[str, Any], schema: dict[str, Any], temperature: float = 0) -> Any:
        calls.append({"message": message, "schema": schema, "temperature": temperature})
        return queue.pop(0)

    ask.calls = calls  # type: ignore[attr-defined]
    return ask


SEEN = {
    "seen": [
        {"item": "Milk", "amount": "plenty"},
        {"item": "cheddar cheese", "amount": "some"},
        {"item": "eggs", "amount": "low"},
    ]
}


def test_scan_fridge_statuses_and_grocery_list(monkeypatch: pytest.MonkeyPatch) -> None:
    ask = fake_gemma(
        SEEN,
        {
            "staples": [
                {"name": "milk", "match": "yes", "seen_item": "milk"},
                {"name": "cheese", "match": "yes", "seen_item": "cheddar cheese"},
                {"name": "eggs", "match": "yes", "seen_item": "eggs"},
                {"name": "butter", "match": "no", "seen_item": ""},
            ]
        },
    )
    monkeypatch.setattr(server, "ask_gemma", ask)

    result = server.scan_fridge("img", ["milk", "cheese", "eggs", "butter"])

    assert result["staples"] == [
        {"name": "milk", "status": "stocked"},
        {"name": "cheese", "status": "stocked"},
        {"name": "eggs", "status": "low"},
        {"name": "butter", "status": "missing"},
    ]
    assert result["grocery_list"] == ["eggs", "butter"]
    assert result["model"] == server.MODEL


def test_vision_step_never_sees_the_staples_list(monkeypatch: pytest.MonkeyPatch) -> None:
    ask = fake_gemma(SEEN, {"staples": []})
    monkeypatch.setattr(server, "ask_gemma", ask)

    server.scan_fridge("img", ["unicorn steaks"])

    vision, matching = ask.calls
    assert vision["message"]["images"] == ["img"]
    assert "unicorn" not in vision["message"]["content"]
    assert "images" not in matching["message"]
    assert "unicorn steaks" in matching["message"]["content"]


def test_match_to_an_item_that_was_never_seen_is_unsure(monkeypatch: pytest.MonkeyPatch) -> None:
    ask = fake_gemma(
        SEEN,
        {"staples": [{"name": "tomatoes", "match": "yes", "seen_item": "tomatoes"}]},
    )
    monkeypatch.setattr(server, "ask_gemma", ask)

    result = server.scan_fridge("img", ["tomatoes", "apples"])

    # "tomatoes" was claimed but never seen; "apples" was dropped by the model entirely.
    assert result["staples"] == [
        {"name": "tomatoes", "status": "unsure"},
        {"name": "apples", "status": "unsure"},
    ]
    assert result["grocery_list"] == []


def test_staples_the_model_dropped_fall_back_to_text_matching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Seen with gemma3:12b: the matching step occasionally comes back without the staples.
    ask = fake_gemma(SEEN, {"staples": []})
    monkeypatch.setattr(server, "ask_gemma", ask)

    result = server.scan_fridge("img", ["milk", "cheese", "eggs", "butter"])

    assert result["staples"] == [
        {"name": "milk", "status": "stocked"},
        {"name": "cheese", "status": "stocked"},
        {"name": "eggs", "status": "low"},
        {"name": "butter", "status": "unsure"},
    ]


def test_suggest_recipes_lists_items_and_adds_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    ask = fake_gemma({"recipes": []})
    monkeypatch.setattr(server, "ask_gemma", ask)

    result = server.suggest_recipes(["eggs", "spinach"])

    assert result["recipes"] == []
    assert "seconds" in result
    assert "- eggs\n- spinach" in ask.calls[0]["message"]["content"]
    assert ask.calls[0]["temperature"] > 0


def test_load_staples_defaults_then_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "STAPLES_FILE", tmp_path / "staples.json")
    assert server.load_staples() == server.DEFAULT_STAPLES

    (tmp_path / "staples.json").write_text('["kale"]')
    assert server.load_staples() == ["kale"]


# ---- HTTP endpoints ----


@pytest.fixture
def base_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.setattr(server, "STAPLES_FILE", tmp_path / "staples.json")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def request(url: str, payload: Any = None) -> tuple[int, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data)) as resp:
            body = resp.read()
            status = resp.status
    except urllib.error.HTTPError as e:
        body, status = e.read(), e.code
    try:
        return status, json.loads(body)
    except ValueError:
        return status, body.decode()


def test_index_page(base_url: str) -> None:
    status, body = request(base_url + "/")
    assert status == 200
    assert "Fridge Sloth" in body


def test_staples_roundtrip(base_url: str) -> None:
    status, saved = request(base_url + "/api/staples", ["  kale ", "", "tofu"])
    assert (status, saved) == (200, ["kale", "tofu"])
    assert request(base_url + "/api/staples") == (200, ["kale", "tofu"])


def test_scan_accepts_data_urls(base_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    received: dict[str, Any] = {}

    def fake_scan(image: str, staples: list[str]) -> dict[str, Any]:
        received.update(image=image, staples=staples)
        return {"ok": True}

    monkeypatch.setattr(server, "scan_fridge", fake_scan)
    status, _ = request(
        base_url + "/api/scan", {"image": "data:image/jpeg;base64,QUJD", "staples": ["milk"]}
    )
    assert status == 200
    assert received == {"image": "QUJD", "staples": ["milk"]}


def test_recipes_need_items(base_url: str) -> None:
    status, body = request(base_url + "/api/recipes", {"items": []})
    assert status == 400
    assert "error" in body


def test_errors_are_returned_as_json(base_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(items: list[str]) -> dict[str, Any]:
        raise RuntimeError("ollama is not running")

    monkeypatch.setattr(server, "suggest_recipes", broken)
    assert request(base_url + "/api/recipes", {"items": ["eggs"]}) == (
        500,
        {"error": "ollama is not running"},
    )


def test_unknown_paths_404(base_url: str) -> None:
    assert request(base_url + "/nope")[0] == 404
    assert request(base_url + "/nope", {})[0] == 404
