# Fridge Sloth 🦥

[![CI](https://github.com/finallyjay/fridge-sloth/actions/workflows/ci.yml/badge.svg)](https://github.com/finallyjay/fridge-sloth/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![Gemma 3](https://img.shields.io/badge/model-Gemma%203-4285F4.svg)](https://ai.google.dev/gemma)

Snap a photo of your fridge, get a **grocery list** and **three healthy meals** you can cook with what's already there.
Runs fully offline on [Gemma 3](https://ai.google.dev/gemma) through [Ollama](https://ollama.com). Your fridge photo never leaves your machine.

Built for the [Hacktoberfest Open-Source AI Challenge, Week 3: Slow and Steady](https://dev.to/challenges/hacktoberfest-week3-2026-10-19).

## How it works

```
┌──────────────┐   1. vision (no hints)   ┌──────────────────────┐
│ fridge photo │ ───────────────────────> │ items actually seen  │
└──────────────┘        Gemma 3           │ + amount left        │
                                          └──────────┬───────────┘
┌──────────────┐   2. text-only matching             │
│ your staples │ ────────────────────────────────────┤
└──────────────┘        Gemma 3                      │
                                          ┌──────────▼───────────┐
                                          │ grocery list         │
                                          │ (missing / low)      │
                                          └──────────┬───────────┘
                     3. healthy recipes              │
                          Gemma 3         ┌──────────▼───────────┐
                                          │ 3 meals, ≤ 30 min    │
                                          └──────────────────────┘
```

1. **Vision:** Gemma lists the food it can actually see in the photo, with a rough amount (`plenty`, `some`, `low`).
   It is *not* shown your staples list. When it was, it "found" staples that weren't there.
2. **Matching:** a text-only call matches your staples (the things you always want at home) against what was seen.
   A match only counts if the matched item really is in step 1's list. Anything missing or low goes on the grocery list.
3. **Recipes:** Gemma suggests three healthy meals built mostly from what's in the fridge: real dishes based on
   vegetables, whole grains, legumes, eggs, fish or lean protein, 30 minutes or less, at most 5 steps.
   Missing ingredients can be added to the grocery list with one tap.

Every call uses a JSON schema in Ollama's `format` field, so the server always gets structured data back.

## Setup

### 1. Install Ollama and pull Gemma

```shell
brew install ollama          # or see https://ollama.com/download
brew services start ollama
ollama pull gemma3:4b        # ~3 GB
```

### 2. Run Fridge Sloth

There are no Python dependencies; only the standard library is used.

```shell
python3 server.py
```

Open http://localhost:8000, take or upload a photo of your fridge and wait about a minute.
To use your phone's camera, open `http://<your-computer's-local-IP>:8000` on the same Wi-Fi.

### Configuration

| Variable       | Default                  | Description                                   |
|----------------|--------------------------|-----------------------------------------------|
| `FRIDGE_MODEL` | `gemma3:4b`              | Any Ollama model with vision, e.g. `gemma3:12b` |
| `OLLAMA_URL`   | `http://localhost:11434` | Where Ollama is running                       |
| `PORT`         | `8000`                   | Port for the web app                          |

Your staples list is saved to `staples.json` next to `server.py` (ignored by git).

### Optional: Sentry Agent Tracing

Want to see where the time goes? Fridge Sloth can send traces to [Sentry Agent Tracing](https://docs.sentry.io/product/agents/getting-started/).
It's **off unless you set `SENTRY_DSN`**:

```shell
pip install -r requirements-sentry.txt
SENTRY_DSN=https://...ingest.sentry.io/... python3 server.py
```

Each scan becomes an `invoke_agent` trace with one `chat` span per Gemma call (`vision`, `matching`, `recipes`),
with the model, token counts and Ollama's own timings (model load, prompt processing, generation).

**Only metadata is sent.** The photo, the prompts and Gemma's answers never leave your machine;
[a test](tests/test_tracing.py) checks exactly that.

## Try it with the example photos

`examples/photos/` has a few real fridge photos from Wikimedia Commons (see [credits](examples/CREDITS.md)).
With the server running:

```shell
python3 scripts/try_photo.py examples/photos/usda-0017.jpg
```

On an M1 MacBook (16 GB), a scan takes about 45 s and the recipes about 30 s with `gemma3:4b`.

## Limitations

- Works well on clear, well-lit photos. On messy or dim ones, the 4B model can miss items or misread them
  (in testing it called a pack of beer "juice bottles").
- It can only see what's in the photo: items hidden behind others or in opaque containers are skipped,
  so a staple can show up as missing when it's just out of view.
- Recipes are only as good as the ingredient list, and are suggestions, not nutritional advice.

## Development

```shell
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pre-commit install

.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/mypy server.py tracing.py scripts tests
.venv/bin/pytest
```

Tests mock Gemma, so they run without Ollama.

## License

[MIT](LICENSE). The example photos keep their original licenses, listed in [examples/CREDITS.md](examples/CREDITS.md).
