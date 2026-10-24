"""Optional Sentry Agent Tracing for Fridge Sloth.

Off unless SENTRY_DSN is set and sentry-sdk is installed (pip install -r requirements-sentry.txt).
Only metadata leaves the machine: which step ran, the model, timings and token counts.
Never the photo, the prompts or Gemma's answers.
"""

import os
from contextlib import AbstractContextManager, nullcontext
from typing import Any

try:
    import sentry_sdk
except ImportError:  # tracing is optional; the app itself has no dependencies
    sentry_sdk = None  # type: ignore[assignment]

AGENT_NAME = "fridge-sloth"
PROVIDER = "ollama"
enabled = False


def init() -> bool:
    global enabled
    dsn = os.environ.get("SENTRY_DSN")
    if not dsn:
        return False
    if sentry_sdk is None:
        print("SENTRY_DSN is set but sentry-sdk is missing: pip install -r requirements-sentry.txt")
        return False
    sentry_sdk.init(
        dsn=dsn,
        traces_sample_rate=1.0,
        send_default_pii=False,
        environment=os.environ.get("SENTRY_ENVIRONMENT", "local"),
    )
    enabled = True
    return True


def agent(task: str, model: str) -> AbstractContextManager[Any]:
    """Root span for one user action: "scan" (vision + matching) or "recipes"."""
    if not enabled:
        return nullcontext()
    span = sentry_sdk.start_transaction(
        op="gen_ai.invoke_agent", name=f"invoke_agent {AGENT_NAME} {task}"
    )
    span.set_data("gen_ai.operation.name", "invoke_agent")
    span.set_data("gen_ai.agent.name", AGENT_NAME)
    span.set_data("gen_ai.request.model", model)
    span.set_data("fridge_sloth.task", task)
    return span


def chat(step: str, model: str, temperature: float) -> AbstractContextManager[Any]:
    """One call to Gemma. `step` is "vision", "matching" or "recipes"."""
    if not enabled:
        return nullcontext()
    span = sentry_sdk.start_span(op="gen_ai.chat", name=f"chat {model}")
    span.set_data("gen_ai.operation.name", "chat")
    span.set_data("gen_ai.system", PROVIDER)
    span.set_data("gen_ai.provider.name", PROVIDER)
    span.set_data("gen_ai.request.model", model)
    span.set_data("gen_ai.request.temperature", temperature)
    span.set_data("fridge_sloth.step", step)
    return span


def record_response(span: Any, response: dict[str, Any]) -> None:
    """Add the model, token counts and Ollama's own timings from an /api/chat response."""
    if span is None:
        return
    span.set_data("gen_ai.response.model", response.get("model"))
    input_tokens = response.get("prompt_eval_count") or 0
    output_tokens = response.get("eval_count") or 0
    span.set_data("gen_ai.usage.input_tokens", input_tokens)
    span.set_data("gen_ai.usage.output_tokens", output_tokens)
    span.set_data("gen_ai.usage.total_tokens", input_tokens + output_tokens)
    # Ollama reports nanoseconds. Load time shows when the model wasn't in memory yet.
    for key in ("load_duration", "prompt_eval_duration", "eval_duration"):
        if key in response:
            span.set_data(f"fridge_sloth.ollama.{key}_ms", round(response[key] / 1e6))


def set_result(span: Any, **counts: int) -> None:
    """Record result sizes on the agent span (e.g. items seen), never the items themselves."""
    if span is None:
        return
    for key, value in counts.items():
        span.set_data(f"fridge_sloth.{key}", value)


def capture(error: Exception) -> None:
    if enabled:
        sentry_sdk.capture_exception(error)
