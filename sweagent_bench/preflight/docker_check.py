"""Check that Docker is available and running."""
from __future__ import annotations

import subprocess


def check_docker() -> bool:
    """Verify Docker daemon is running and accessible.

    Returns:
        True if ``docker info`` succeeds.
    """
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            return True
        print(f"    docker info failed: {result.stderr[:200]}")
        return False
    except FileNotFoundError:
        print("    docker command not found")
        return False
    except subprocess.TimeoutExpired:
        print("    docker info timed out")
        return False
    except Exception as exc:
        print(f"    Docker check error: {exc}")
        return False
