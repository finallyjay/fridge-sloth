import io
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

    def ask(
        message: dict[str, Any], schema: dict[str, Any], temperature: float = 0, step: str = ""
    ) -> Any:
        calls.append(
            {"message": message, "schema": schema, "temperature": temperature, "step": step}
        )
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
    ask = fake_gemma(SEEN)
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


def test_gemma_only_looks_at_the_photo(monkeypatch: pytest.MonkeyPatch) -> None:
    ask = fake_gemma(SEEN)  # a second call would fail: staples are matched without Gemma
    monkeypatch.setattr(server, "ask_gemma", ask)

    server.scan_fridge("img", ["unicorn steaks", "milk"])

    (vision,) = ask.calls
    assert vision["message"]["images"] == ["img"]
    assert "unicorn" not in vision["message"]["content"]


def test_a_flavour_is_not_the_food(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {
        "seen": [
            {"item": "tomato sauce", "amount": "plenty"},
            {"item": "orange", "amount": "plenty"},
            {"item": "egg", "amount": "some"},
        ]
    }
    monkeypatch.setattr(server, "ask_gemma", fake_gemma(seen))

    result = server.scan_fridge("img", ["tomatoes", "orange juice", "eggs"])

    assert result["staples"] == [
        {"name": "tomatoes", "status": "missing"},
        {"name": "orange juice", "status": "missing"},
        {"name": "eggs", "status": "stocked"},
    ]


def test_suggest_recipes_lists_items_and_adds_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    ask = fake_gemma({"recipes": []})
    monkeypatch.setattr(server, "ask_gemma", ask)

    result = server.suggest_recipes(["eggs", "spinach"])

    assert result["recipes"] == []
    assert "seconds" in result
    assert "- eggs\n- spinach" in ask.calls[0]["message"]["content"]
    assert ask.calls[0]["temperature"] > 0


def test_recipe_ingredients_not_in_the_fridge_move_to_need_to_buy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = {
        "name": "Yogurt bowl",
        "uses": ["yogurt", "banana", "egg"],
        "need_to_buy": ["Granola", "Banana"],
    }
    monkeypatch.setattr(server, "ask_gemma", fake_gemma({"recipes": [recipe]}))

    result = server.suggest_recipes(["greek yogurt", "eggs", "orange"])

    assert result["recipes"][0]["uses"] == ["yogurt", "egg"]
    assert result["recipes"][0]["need_to_buy"] == ["Granola", "Banana"]


def test_seeing_an_orange_does_not_mean_you_have_orange_juice() -> None:
    recipes: list[dict[str, Any]] = [{"uses": ["orange juice", "orange"]}]

    assert server.check_recipe_ingredients(recipes, ["orange"]) == 1
    assert recipes[0] == {"uses": ["orange"], "need_to_buy": ["orange juice"]}


@pytest.mark.parametrize(
    ("item", "name", "expected"),
    [
        ("cheddar cheese", "cheese", True),
        ("eggs", "egg", True),
        ("egg", "eggs", True),
        ("tomato", "tomatoes", True),
        ("cherry tomatoes", "tomato", True),
        ("orange", "orange juice", False),
        ("eggplant", "egg", False),
        ("eggplant", "eggs", False),
        ("", "milk", False),
        ("free-range eggs", "eggs", True),
        ("greek yogurt", "yogurt", True),
        ("tomato sauce", "tomatoes", False),
        ("milk chocolate", "milk", False),
        ("cheese balls", "cheese", False),
    ],
)
def test_mentions(item: str, name: str, expected: bool) -> None:
    assert server.mentions(item, name) is expected


def test_an_eggplant_is_not_an_egg() -> None:
    recipes: list[dict[str, Any]] = [{"uses": ["egg", "eggplant"], "need_to_buy": ["Egg "]}]

    assert server.check_recipe_ingredients(recipes, ["eggplant"]) == 1
    # "egg" moves to need_to_buy, without duplicating the existing "Egg ".
    assert recipes[0] == {"uses": ["eggplant"], "need_to_buy": ["Egg "]}


def test_repeated_items_are_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    looping = {"seen": [{"item": "Beer", "amount": "plenty"}, {"item": "beer ", "amount": "low"}]}
    monkeypatch.setattr(server, "ask_gemma", fake_gemma(looping))

    assert server.scan_fridge("img", [])["seen"] == [{"item": "Beer", "amount": "plenty"}]


def test_cut_off_answers_fail_with_a_clear_message(monkeypatch: pytest.MonkeyPatch) -> None:
    reply = {"message": {"content": '{"seen": [{"item": "be'}, "done_reason": "length"}
    monkeypatch.setattr(
        server.urllib.request,
        "urlopen",
        lambda req, timeout: io.BytesIO(json.dumps(reply).encode()),
    )
    with pytest.raises(RuntimeError, match="cut off"):
        server.ask_gemma({"role": "user", "content": "hi"}, {})


def test_vision_schema_caps_the_number_of_items() -> None:
    properties: Any = server.SEEN_SCHEMA["properties"]
    assert properties["seen"]["maxItems"] == 30


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
