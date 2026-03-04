"""Run ID generation utilities."""
from __future__ import annotations

import secrets
from datetime import datetime


def make_run_id(prefix: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(2)
    return f"{prefix}_{timestamp}_{suffix}"
