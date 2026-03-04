"""Pre-flight checks for experiment readiness."""
from __future__ import annotations

from sweagent_bench.preflight.vllm_check import check_vllm
from sweagent_bench.preflight.docker_check import check_docker
from sweagent_bench.preflight.dataset_check import check_dataset


def run_preflight(
    api_base: str | None = None,
    dataset_name: str = "princeton-nlp/SWE-bench_Verified",
    split: str = "test",
) -> bool:
    """Run all pre-flight checks. Returns True if all pass."""
    checks = [
        ("vLLM endpoint", lambda: check_vllm(api_base)),
        ("Docker availability", check_docker),
        ("Dataset load", lambda: check_dataset(dataset_name, split)),
    ]
    all_ok = True
    for name, fn in checks:
        try:
            ok = fn()
            status = "OK" if ok else "FAIL"
            if not ok:
                all_ok = False
        except Exception as exc:
            status = f"ERROR: {exc}"
            all_ok = False
        print(f"  [preflight] {name}: {status}")
    return all_ok
