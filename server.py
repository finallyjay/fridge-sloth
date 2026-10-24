"""Fridge Sloth: snap your fridge, get a grocery list and healthy meals.

Runs fully offline on Gemma via Ollama.
"""

import json
import os
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
MATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "staples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "match": {"type": "string", "enum": ["yes", "no"]},
                    "seen_item": {"type": "string"},
                },
                "required": ["name", "match", "seen_item"],
            },
        },
    },
    "required": ["staples"],
}

MATCH_PROMPT = """Items seen in a fridge:
{seen}

For EACH staple below, say whether it is among the seen items:
{staples}

- "yes": a seen item is this staple (e.g. "cheddar cheese" counts as "cheese",
  "2% milk" counts as "milk"). Put that item in seen_item.
- "no": nothing seen matches. Leave seen_item empty.

Be strict: a different food is not a match (yogurt is not milk, pesto is not lettuce).
Return the staples in the same order and spelling as given."""


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
        "options": {"temperature": temperature},
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
    return json.loads(response["message"]["content"])


def fallback_match(staple: str, amounts: dict[str, str]) -> dict[str, str]:
    """Plain text match for a staple the model left out of its answer ("cheese" ~ "cheddar cheese").

    No text match means we can't tell, not that it's missing: the model never ruled on it.
    """
    # Only the staple inside the seen item, never the reverse: seeing "orange" must not
    # count as "orange juice". The singular form lets "egg" count for "eggs".
    names = {staple.lower(), staple.lower().removesuffix("s")}
    for item in amounts:
        if item and any(name in item for name in names):
            return {"match": "yes", "seen_item": item}
    return {"match": "", "seen_item": ""}


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
    seen = ask_gemma(vision, SEEN_SCHEMA, step="vision")["seen"]
    amounts = {s["item"].strip().lower(): s["amount"] for s in seen}

    matched = ask_gemma(
        {
            "role": "user",
            "content": MATCH_PROMPT.format(
                seen="\n".join(f"- {s['item']}" for s in seen) or "- nothing",
                staples="\n".join(f"- {s}" for s in staples),
            ),
        },
        MATCH_SCHEMA,
        step="matching",
    )["staples"]

    # Small models sometimes drop or rename staples, or "match" to an item that was never seen.
    by_name = {m["name"].strip().lower(): m for m in matched}
    result_staples: list[dict[str, str]] = []
    for name in staples:
        m = by_name.get(name.lower()) or fallback_match(name, amounts)
        seen_item = m.get("seen_item", "").strip().lower()
        if m["match"] == "yes" and seen_item in amounts:
            status = "low" if amounts[seen_item] == "low" else "stocked"
        elif m["match"] == "no":
            status = "missing"
        else:
            status = "unsure"
        result_staples.append({"name": name, "status": status})

    result: dict[str, Any] = {
        "seen": seen,
        "staples": result_staples,
        "grocery_list": [s["name"] for s in result_staples if s["status"] in ("missing", "low")],
    }
    result["seconds"] = round(time.time() - started, 1)
    result["model"] = MODEL
    return result


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
        result["seconds"] = round(time.time() - started, 1)
        tracing.set_result(span, ingredients=len(items), recipes=len(result.get("recipes", [])))
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
