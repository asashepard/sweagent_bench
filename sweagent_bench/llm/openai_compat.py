"""OpenAI-compatible HTTP client for model inference.

ADAPTED: adds ContextLengthError for immediate fail-fast on context overflow.
"""
from __future__ import annotations

import os
import time

import requests


class ContextLengthError(RuntimeError):
    """Raised when the request exceeds the model's context window.

    This is NOT retried — the caller must reduce input size or abort.
    """
    pass


def get_base_url() -> str:
    """Get the OpenAI-compatible API base URL from environment."""
    return os.environ.get("OPENAI_BASE_URL", "http://localhost:8001/v1")


def get_api_key() -> str:
    """Get the API key from environment."""
    return os.environ.get("OPENAI_API_KEY", "EMPTY")


def chat_completion(
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    timeout_s: int = 120,
    api_base: str | None = None,
) -> str:
    """Call OpenAI-compatible chat completion endpoint.

    Args:
        model: Model name/path.
        messages: List of message dicts with 'role' and 'content'.
        temperature: Sampling temperature (default 0.0 for deterministic).
        top_p: Top-p sampling parameter (default 1.0).
        max_tokens: Maximum tokens to generate (default 512).
        timeout_s: Request timeout in seconds.
        api_base: Optional API base URL override. If provided, used instead
            of the OPENAI_BASE_URL environment variable. This avoids
            thread-unsafe os.environ mutation.

    Returns:
        The assistant's response content string.

    Raises:
        ContextLengthError: If the request exceeds the model's context window.
            NOT retried — caller must handle.
        RuntimeError: If request fails after retries.
    """
    data = chat_completion_with_metadata(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
        api_base=api_base,
    )
    return data["content"]


def chat_completion_with_metadata(
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    timeout_s: int = 120,
    api_base: str | None = None,
) -> dict:
    """Call chat completion endpoint and return content plus optional usage.

    Args:
        model: Model name/path.
        messages: List of message dicts.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        max_tokens: Maximum tokens to generate.
        timeout_s: Request timeout in seconds.
        api_base: Optional API base URL override. Takes precedence over
            the OPENAI_BASE_URL environment variable and avoids
            thread-unsafe os.environ mutation.

    Returns a dict with keys:
    - content: assistant text
    - usage: token usage dict (may be empty)
    """
    base_url = (api_base or get_base_url()).rstrip("/")
    url = f"{base_url}/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {get_api_key()}",
    }

    request_model = model
    if "api.openai.com" in base_url and "/" in request_model:
        provider, bare = request_model.split("/", 1)
        if provider.strip().lower() == "openai" and bare.strip():
            request_model = bare.strip()

    payload = {
        "model": request_model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    max_retries = 4
    last_error = None

    for attempt in range(max_retries):
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout_s,
            )
            response.raise_for_status()
            data = response.json()
            usage_raw = data.get("usage", {}) if isinstance(data, dict) else {}
            usage = usage_raw if isinstance(usage_raw, dict) else {}
            return {
                "content": data["choices"][0]["message"]["content"],
                "usage": {
                    "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                    "total_tokens": int(usage.get("total_tokens", 0) or 0),
                },
            }
        except requests.RequestException as e:
            last_error = e
            if hasattr(e, "response") and e.response is not None:
                status = getattr(e.response, "status_code", None)
                if status and 400 <= status < 500:
                    body = e.response.text[:1200]

                    if any(marker in body.lower() for marker in (
                        "context_length",
                        "maximum context length",
                        "token limit",
                        "context window",
                        "prompt is too long",
                    )):
                        raise ContextLengthError(
                            f"Context length exceeded (HTTP {status}): {body[:300]}"
                        ) from e

                    if (
                        "unsupported_parameter" in body
                        and "max_tokens" in body
                        and "max_completion_tokens" in body
                        and "max_tokens" in payload
                    ):
                        payload.pop("max_tokens", None)
                        payload["max_completion_tokens"] = max_tokens

                    last_error = RuntimeError(f"HTTP {status}: {body}")
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                if hasattr(e, "response") and getattr(e.response, "status_code", 0) == 429:
                    wait = max(wait, 10)
                time.sleep(wait)

    raise RuntimeError(f"Chat completion failed after {max_retries} attempts: {last_error}")
