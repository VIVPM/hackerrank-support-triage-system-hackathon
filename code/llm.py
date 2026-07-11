"""Thin wrapper around HuggingFace Inference Providers (OpenAI-compatible).

All LLM calls in the project go through this module — no scattered SDK
clients elsewhere. That makes it easy to swap providers, retry, log, or
add a fake for tests.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO_ROOT = Path(__file__).resolve().parent.parent

# HuggingFace Inference Providers OpenAI-compatible router.
DEFAULT_BASE_URL = "https://router.huggingface.co/v1"

# Model defaults (architecture.md §3 / Stage 3b + Stage 5).
DEFAULT_STAGE3B_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_STAGE5_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

# Pin a HuggingFace Inference Provider for determinism. When unset, the HF
# router auto-routes across backends (Together, Fireworks, Cerebras,
# Hyperbolic, Novita, Nebius) and subtle tokenizer / sampling differences
# between them cause occasional verdict flips on borderline tickets even at
# temperature=0. The interview brief documents two known cold-vs-cached
# flips caused by this. Pinning "together" by default removes that
# non-determinism. Override or unset via HF_PROVIDER.
DEFAULT_HF_PROVIDER = "together"

DEFAULT_TEMPERATURE = 0.0
LLM_RETRIES = 4
LLM_RETRY_BASE_DELAY = 4

# Throttle: after every N successful LLM calls, sleep S seconds.
# Override via env vars LLM_THROTTLE_EVERY and LLM_THROTTLE_SLEEP.
DEFAULT_THROTTLE_EVERY = 4
DEFAULT_THROTTLE_SLEEP = 30

_client = None
_call_count = 0  # increments on every successful chat completion


def _get_client():
    """Build a process-cached OpenAI client pointed at HF Providers."""
    global _client
    if _client is None:
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv(REPO_ROOT / ".env")
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "HF_TOKEN not set — see .env.example. Get a token at "
                "https://huggingface.co/settings/tokens"
            )
        _client = OpenAI(base_url=DEFAULT_BASE_URL, api_key=token)
    return _client


def stage3b_model() -> str:
    return os.environ.get("STAGE3B_MODEL", DEFAULT_STAGE3B_MODEL)


def stage5_model() -> str:
    return os.environ.get("STAGE5_MODEL", DEFAULT_STAGE5_MODEL)


def hf_provider() -> str | None:
    """Return the pinned HF Inference Provider, or None for auto-routing.

    Reads HF_PROVIDER env var; falls back to DEFAULT_HF_PROVIDER. To
    explicitly opt back into auto-routing, set HF_PROVIDER="auto" or "".
    """
    val = os.environ.get("HF_PROVIDER", DEFAULT_HF_PROVIDER)
    if not val or val.strip().lower() in {"auto", "none"}:
        return None
    return val.strip()


def _throttle_config() -> tuple[int, int]:
    every = int(os.environ.get("LLM_THROTTLE_EVERY", DEFAULT_THROTTLE_EVERY))
    sleep_s = int(os.environ.get("LLM_THROTTLE_SLEEP", DEFAULT_THROTTLE_SLEEP))
    return max(0, every), max(0, sleep_s)


def _maybe_throttle() -> None:
    """Increment the global call counter and sleep if we've hit the threshold.

    Cheap rate-limit avoidance for HF Providers free tier. Sleep happens AFTER
    the call returns, so the just-completed response is unaffected; the next
    call begins after the cooldown.
    """
    global _call_count
    _call_count += 1
    every, sleep_s = _throttle_config()
    if every <= 0 or sleep_s <= 0:
        return
    if _call_count % every == 0:
        print(
            f"  [throttle] {_call_count} LLM calls done — sleeping {sleep_s}s "
            f"to respect provider rate limits…",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(sleep_s)


def reset_call_count() -> None:
    """For tests / multi-run sessions where you want a fresh throttle window."""
    global _call_count
    _call_count = 0


def call_chat(
    model: str,
    messages: list[dict],
    *,
    json_object: bool = False,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = 1024,
) -> str:
    """Single chat-completion call. Returns the assistant message text.

    `json_object=True` asks the provider to constrain output to a JSON
    object (OpenAI's `response_format={"type":"json_object"}`). Not every
    HF Providers backend honors this, so callers should still defensively
    parse the result.
    """
    client = _get_client()
    kwargs: dict[str, Any] = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if json_object:
        kwargs["response_format"] = {"type": "json_object"}
    provider = hf_provider()
    if provider:
        # HF Inference Providers OpenAI-compatible endpoint reads the
        # provider routing hint from extra_body. Pinning here removes the
        # cross-backend non-determinism documented in the interview brief.
        kwargs["extra_body"] = {"provider": provider}

    last_exc: Exception | None = None
    for attempt in range(LLM_RETRIES):
        try:
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            _maybe_throttle()
            return content
        except Exception as e:
            last_exc = e
            msg = str(e).lower()
            transient = (
                "rate" in msg
                or "timeout" in msg
                or "503" in msg
                or "502" in msg
                or "504" in msg
                or "connection" in msg
                or "temporarily" in msg
            )
            if transient and attempt < LLM_RETRIES - 1:
                wait = LLM_RETRY_BASE_DELAY * (2**attempt)
                print(f"  LLM transient error — retry in {wait}s ({e!r})", file=sys.stderr)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"LLM call failed after {LLM_RETRIES} retries: {last_exc}")


def parse_json_lenient(text: str) -> dict:
    """Best-effort JSON parser tolerating common LLM quirks (code fences,
    leading/trailing prose). Returns {} on failure rather than raising —
    callers decide whether to escalate."""
    if not text:
        return {}
    s = text.strip()
    # Strip markdown code fences.
    if s.startswith("```"):
        # remove first fence line and last fence
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    # If model added prose around a JSON object, try to grab the first {...} block.
    if not s.startswith("{"):
        i = s.find("{")
        j = s.rfind("}")
        if i != -1 and j != -1 and j > i:
            s = s[i : j + 1]
    try:
        obj = json.loads(s)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _selftest() -> None:
    """Tiny smoke test — one cheap call to confirm credentials work."""
    print(f"Stage 3b model: {stage3b_model()}")
    print(f"Stage 5 model:  {stage5_model()}")
    print()
    print("Sending a 1-token ping to Stage 3b model...")
    t0 = time.time()
    out = call_chat(
        stage3b_model(),
        [{"role": "user", "content": "Reply with exactly the word: pong"}],
        max_tokens=10,
    )
    print(f"  response: {out!r}")
    print(f"  elapsed:  {time.time() - t0:.1f}s")


if __name__ == "__main__":
    _selftest()
