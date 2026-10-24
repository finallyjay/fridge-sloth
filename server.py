"""Fridge Sloth: snap your fridge, get a grocery list and healthy meals.

Runs fully offline on Gemma via Ollama.
"""

import json
import os
import re
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import tracing

ROOT = Path(__file__).parent
STAPLES_FILE = ROOT / "staples.json"
INDEX_FILE = ROOT / "static" / "index.html"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("FRIDGE_MODEL", "gemma3:4b")
PORT = int(os.environ.get("PORT", "8000"))
MAX_ANSWER_TOKENS = 2048  # a full answer is ~300–600 tokens

DEFAULT_STAPLES = [
    "milk", "eggs", "butter", "cheese", "yogurt",
    "tomatoes", "lettuce", "carrots", "apples", "orange juice",
]  # fmt: skip

# Step 1 looks at the photo WITHOUT the staples list.
# Showing the list made Gemma "see" staples that weren't there.
SEEN_SCHEMA = {
    "type": "object",
    "properties": {
        "seen": {
            "type": "array",
            # On messy photos the 4B model can loop, repeating items until its JSON is cut off.
            "maxItems": 30,
            "items": {
                "type": "object",
                "properties": {
                    "item": {"type": "string"},
                    "amount": {"type": "string", "enum": ["plenty", "some", "low"]},
                },
                "required": ["item", "amount"],
            },
        },
    },
    "required": ["seen"],
}

SEEN_PROMPT = """You are looking at a photo of the inside of a fridge.

List every food or drink item you can actually see, shelf by shelf,
including fruit, vegetables, bread and packaged food.
Use plain grocery names (e.g. "cheddar cheese", "eggs", "pineapple", "milk").
For each, estimate the amount left: "plenty", "some" or "low".
If you see a closed, opaque container and can't tell what's inside, skip it.
Only list things you can see. Never guess items that might be hidden."""

# Step 2 is text only: match the household's staples against what was actually seen.
RECIPE_SCHEMA = {
    "type": "object",
    "properties": {
        "recipes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "why_healthy": {"type": "string"},
                    "minutes": {"type": "integer"},
                    "uses": {"type": "array", "items": {"type": "string"}},
                    "need_to_buy": {"type": "array", "items": {"type": "string"}},
                    "steps": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "why_healthy", "minutes", "uses", "need_to_buy", "steps"],
            },
        },
    },
    "required": ["recipes"],
}

RECIPE_PROMPT = """Everything in my fridge right now:
{items}

Suggest 3 different HEALTHY meals (breakfast, lunch or dinner) I can make mostly from these.
Sauces, condiments and drinks are only small additions, never the base of a meal.

Healthy means:
- a real dish such as an omelette, a big salad, a grain bowl, a stir-fry, a soup or a wrap
- built around vegetables, fruit, whole grains, legumes, eggs, fish or lean protein
- little added sugar, no deep frying, go easy on butter, cream, cheese and salty sauces
- never a drink, smoothie or snack presented as a meal

Rules:
- each meal uses at least 2 foods from my fridge (not counting sauces or drinks)
- each meal is clearly different from the others
- "uses" lists ONLY items from my fridge list above, never anything else
- "need_to_buy" lists every other ingredient needed (3 items or fewer);
  assume I have salt, pepper, oil and water
- I'm feeling lazy: 30 minutes or less, at most 5 short steps
- "why_healthy" is one short, specific sentence (mention the vegetables, protein or fibre)"""


def load_staples() -> list[str]:
    if STAPLES_FILE.exists():
        staples: list[str] = json.loads(STAPLES_FILE.read_text())
        return staples
    return DEFAULT_STAPLES


def ask_gemma(
    message: dict[str, Any], schema: dict[str, Any], temperature: float = 0, step: str = ""
) -> Any:
    body = {
        "model": MODEL,
        "stream": False,
        "format": schema,
        # num_predict caps the answer length so a looping model fails fast instead of rambling.
        "options": {"temperature": temperature, "num_predict": MAX_ANSWER_TOKENS},
        "messages": [message],
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with tracing.chat(step, MODEL, temperature) as span:
        with urllib.request.urlopen(req, timeout=300) as resp:
            response = json.loads(resp.read())
        tracing.record_response(span, response)
    if response.get("done_reason") == "length":
        raise RuntimeError(
            "Gemma's answer got too long and was cut off. Try again, or use a clearer photo."
        )
    return json.loads(response["message"]["content"])


def mentions(item: str, name: str) -> bool:
    """True if `item` is a `name`: the name is the food the item ends with, singular or plural.

    "cheddar cheese" is cheese and "free-range eggs" are eggs (or "cherry tomatoes" tomatoes).
    But the name has to be what the item *is*, not a flavour or an ingredient of it:
    "tomato sauce" is not tomatoes, "milk chocolate" is not milk, "orange" is not orange juice,
    and an "eggplant" is not an egg.
    """
    name, item = name.strip().lower(), item.strip().lower()
    stems = {name, name.removesuffix("s"), name.removesuffix("es")}
    return any(s and re.search(rf"\b{re.escape(s)}(?:e?s)?$", item) for s in stems)


def dedupe(seen: list[dict[str, str]]) -> list[dict[str, str]]:
    """Drop repeated items (same name, any case), keeping the first."""
    unique: dict[str, dict[str, str]] = {}
    for s in seen:
        unique.setdefault(s["item"].strip().lower(), s)
    return list(unique.values())


def local_match(staple: str, seen_items: list[str]) -> str | None:
    """The first seen item that is this staple by plain word matching, without the model."""
    return next((item for item in seen_items if mentions(item, staple)), None)


def scan_fridge(image_b64: str, staples: list[str]) -> dict[str, Any]:
    with tracing.agent("scan", MODEL) as span:
        result = _scan_fridge(image_b64, staples)
        tracing.set_result(
            span,
            items_seen=len(result["seen"]),
            staples=len(staples),
            grocery_list=len(result["grocery_list"]),
        )
        return result


def _scan_fridge(image_b64: str, staples: list[str]) -> dict[str, Any]:
    started = time.time()
    vision = {"role": "user", "content": SEEN_PROMPT, "images": [image_b64]}
    raw_seen = ask_gemma(vision, SEEN_SCHEMA, step="vision")["seen"]
    with tracing.tool("dedupe_seen_items") as span:
        seen = dedupe(raw_seen)
        tracing.set_result(span, items_in=len(raw_seen), items_out=len(seen))
    amounts = {s["item"].strip().lower(): s["amount"] for s in seen}
    seen_items = list(amounts)

    # Staples are matched by name, not by Gemma. Asked to match staples under "other names",
    # the 4B model over-claimed (spinach -> cheddar cheese), and its yes/no answer per staple
    # was the slowest part of a scan (~25 s in Sentry traces). Missing a synonym only means
    # buying something you have; a false match means not buying something you need.
    with tracing.tool("match_staples_by_name") as span:
        found = {s: item for s in staples if (item := local_match(s, seen_items))}
        tracing.set_result(span, staples=len(staples), matched=len(found))

    result_staples = [
        {
            "name": name,
            "status": "missing"
            if name not in found
            else ("low" if amounts[found[name]] == "low" else "stocked"),
        }
        for name in staples
    ]

    result: dict[str, Any] = {
        "seen": seen,
        "staples": result_staples,
        "grocery_list": [s["name"] for s in result_staples if s["status"] in ("missing", "low")],
    }
    result["seconds"] = round(time.time() - started, 1)
    result["model"] = MODEL
    return result


def check_recipe_ingredients(recipes: list[dict[str, Any]], items: list[str]) -> int:
    """Move "from your fridge" ingredients that weren't actually seen into "need to buy".

    Gemma sometimes ignores the rule and "uses" food that isn't there (a banana that was
    never in the photo). Returns how many ingredients were moved.
    """
    moved = 0
    for recipe in recipes:
        need = recipe.setdefault("need_to_buy", [])
        uses = []
        for ingredient in recipe.get("uses", []):
            if any(mentions(item, ingredient) for item in items):
                uses.append(ingredient)
            else:
                moved += 1
                if ingredient.strip().lower() not in (n.strip().lower() for n in need):
                    need.append(ingredient.strip())
        recipe["uses"] = uses
    return moved


def suggest_recipes(items: list[str]) -> dict[str, Any]:
    with tracing.agent("recipes", MODEL) as span:
        started = time.time()
        # A little temperature so "suggest again" gives different meals.
        result: dict[str, Any] = ask_gemma(
            {
                "role": "user",
                "content": RECIPE_PROMPT.format(items="\n".join(f"- {i}" for i in items)),
            },
            RECIPE_SCHEMA,
            temperature=0.7,
            step="recipes",
        )
        with tracing.tool("check_recipe_ingredients") as tool_span:
            moved = check_recipe_ingredients(result.get("recipes", []), items)
            tracing.set_result(tool_span, moved_to_grocery_list=moved)
        result["seconds"] = round(time.time() - started, 1)
        tracing.set_result(
            span,
            ingredients=len(items),
            recipes=len(result.get("recipes", [])),
            ingredients_not_in_fridge=moved,
        )
        return result


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: Any, content_type: str = "application/json") -> None:
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:  # noqa: N802 (name required by BaseHTTPRequestHandler)
        if self.path == "/":
            self._send(200, INDEX_FILE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/staples":
            self._send(200, load_staples())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path == "/api/scan":
                body = self._read_json()
                image = body["image"].split(",", 1)[-1]  # accept data URLs
                self._send(200, scan_fridge(image, body.get("staples") or load_staples()))
            elif self.path == "/api/recipes":
                items = self._read_json().get("items") or []
                if not items:
                    self._send(400, {"error": "no food found in the scan"})
                else:
                    self._send(200, suggest_recipes(items))
            elif self.path == "/api/staples":
                staples = [s.strip() for s in self._read_json() if s.strip()]
                STAPLES_FILE.write_text(json.dumps(staples, indent=2))
                self._send(200, staples)
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:  # surface errors to the page instead of a dropped connection
            tracing.capture(e)
            self._send(500, {"error": str(e)})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} {fmt % args}")


if __name__ == "__main__":
    print(f"Fridge Sloth on http://localhost:{PORT} using {MODEL}")
    if tracing.init():
        print("Sentry Agent Tracing is on (timings and token counts only)")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
