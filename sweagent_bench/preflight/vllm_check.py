"""Check that the vLLM server is reachable and responding."""
from __future__ import annotations

import os

import requests


DEFAULT_API_BASE = "http://localhost:8001/v1"


def check_vllm(api_base: str | None = None) -> bool:
    """Ping the vLLM /models endpoint.

    Args:
        api_base: OpenAI-compatible base URL. Falls back to
            OPENAI_BASE_URL env var, then localhost:8001/v1.

    Returns:
        True if the server lists at least one model.
    """
    base = api_base or os.environ.get("OPENAI_BASE_URL", DEFAULT_API_BASE)
    base = base.rstrip("/")
    url = f"{base}/models"

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data", [])
        if models:
            names = [m.get("id", "?") for m in models]
            print(f"    vLLM models: {', '.join(names)}")
            return True
        print("    vLLM responded but no models loaded")
        return False
    except requests.ConnectionError:
        print(f"    Cannot connect to vLLM at {url}")
        return False
    except Exception as exc:
        print(f"    vLLM check error: {exc}")
        return False
