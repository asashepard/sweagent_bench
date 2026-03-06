"""Continuous oracle tuning loop for AGENTS.md."""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from sweagent_bench.git.checkout import checkout_repo
from sweagent_bench.guidance.schema import RepoGuidance
from sweagent_bench.kb.agents_md import AGENTS_MD_CHAR_BUDGET, render_agents_md
from sweagent_bench.kb.builder import build_kb
from sweagent_bench.kb.schema import RepoKB
from sweagent_bench.oracle.apply import apply_edits
from sweagent_bench.oracle.diagnose import diagnose_failures
from sweagent_bench.oracle.judge import evaluate_probe
from sweagent_bench.oracle.probes import generate_probes
from sweagent_bench.oracle.schema import Edit, OracleConfig, OracleState, Probe, ProbeResult
from sweagent_bench.probes import run_all_probes


MAX_EDITS_PER_ITERATION = 8
RESERVED_CHAR_BUFFER = 640
SOFT_CHAR_BUDGET = AGENTS_MD_CHAR_BUDGET - RESERVED_CHAR_BUFFER
MIN_EDIT_SUPPORT_DEFAULT = 2


def _olog(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [oracle] {msg}", flush=True)


def _summarize_edit(edit: Edit, max_len: int = 180) -> str:
    """Compact one-line summary of an edit for logging."""
    content = " ".join(edit.content.split())
    if len(content) > max_len:
        content = content[: max_len - 3] + "..."
    return f"section={edit.section!r} action={edit.action!r} content={content!r}"


def run_oracle_loop(config: OracleConfig) -> tuple[RepoKB, RepoGuidance]:
    """Run the continuous LLM-driven oracle loop for one repository."""
    out = (
        Path(config.output_dir) if config.output_dir
        else Path("artifacts/guidance") / config.repo.replace("/", "__")
    )
    out.mkdir(parents=True, exist_ok=True)

    state_path = out / "tuning_state.json"
    kb_dir = out / "kb"
    kb_dir.mkdir(parents=True, exist_ok=True)
    guidance_dir = out / "versions"
    guidance_dir.mkdir(parents=True, exist_ok=True)

    _olog(f"Starting oracle loop for repo={config.repo} commit={config.commit}")
    _olog(f"Model: {config.model} | Iterations: {config.iterations} | Timeout: {config.timeout_s}s")

    t_checkout = time.perf_counter()
    repo_dir = checkout_repo(config.repo, config.commit)
    _olog(f"Checkout complete in {time.perf_counter() - t_checkout:.2f}s at {repo_dir}")

    t_probes = time.perf_counter()
    probe_results = run_all_probes(repo_dir)
    _olog(f"Structural probes complete in {time.perf_counter() - t_probes:.2f}s")

    _save_probe_results_summary(probe_results, kb_dir / "probes_summary.json")

    t_kb = time.perf_counter()
    kb = build_kb(config.repo, config.commit, probe_results)
    kb.save(kb_dir / "kb.json")
    _olog(f"KB built in {time.perf_counter() - t_kb:.2f}s")

    agents_md = render_agents_md(kb)
    v0_guidance = RepoGuidance(
        repo=config.repo, commit=config.commit,
        lines=agents_md.splitlines(), version=0,
    )
    v0_guidance.save(guidance_dir / "v0.json")
    (kb_dir / "agents_md_v0.md").write_text(agents_md, encoding="utf-8")

    if config.iterations <= 0:
        final_path = out / "best_guidance.json"
        v0_guidance.save(final_path)
        return kb, v0_guidance

    if state_path.exists():
        state = OracleState.load(state_path)
        if state.completed_iterations > 0:
            current_path = guidance_dir / f"v{state.current_version}.json"
            if current_path.exists():
                current = RepoGuidance.load(current_path)
                agents_md = current.render()
            else:
                state = OracleState(repo=config.repo)
        else:
            state = OracleState(repo=config.repo)
    else:
        state = OracleState(repo=config.repo)

    current = RepoGuidance(
        repo=config.repo, commit=config.commit,
        lines=agents_md.splitlines(), version=state.current_version,
    )

    start_iter = state.completed_iterations + 1
    for t in range(start_iter, config.iterations + 1):
        iter_start = time.perf_counter()
        _olog(f"Iteration {t}/{config.iterations} started for {config.repo}")

        t_probe_gen = time.perf_counter()
        new_probes = generate_probes(
            kb, config.model, agents_md,
            prior_probes=[], timeout_s=config.timeout_s,
        )
        _olog(
            f"Iteration {t}: generated {len(new_probes)} probes in "
            f"{time.perf_counter() - t_probe_gen:.2f}s (pool={len(new_probes)})"
        )
        if new_probes:
            probe_ids = ", ".join(p.id for p in new_probes)
            _olog(f"Iteration {t}: probe ids this round: {probe_ids}")
        else:
            _olog(f"Iteration {t}: no probes generated; continuing with empty result set")

        t_eval = time.perf_counter()
        results = _evaluate_all_probes_detailed(agents_md, new_probes, config)
        _olog(
            f"Iteration {t}: evaluated {len(results)} probes in "
            f"{time.perf_counter() - t_eval:.2f}s"
        )

        t_diag = time.perf_counter()
        llm_edits = diagnose_failures(agents_md, results, config.model, timeout_s=config.timeout_s)
        _olog(
            f"Iteration {t}: diagnose_failures proposed {len(llm_edits)} edits in "
            f"{time.perf_counter() - t_diag:.2f}s"
        )
        direct_edits = _collect_edits_from_results(results)
        _olog(f"Iteration {t}: direct edits from probes={len(direct_edits)}")
        raw_edits = [*direct_edits, *llm_edits]
        deduped_edits = _dedupe_edits(raw_edits)
        _olog(f"Iteration {t}: deduped edits={len(deduped_edits)}")
        edits = _prioritize_edits_for_iteration(
            raw_edits=raw_edits,
            deduped_edits=deduped_edits,
            current_agents_md=agents_md,
            max_edits=MAX_EDITS_PER_ITERATION,
        )
        _olog(
            f"Iteration {t}: prioritized edits={len(edits)} "
            f"(cap={MAX_EDITS_PER_ITERATION}, soft_budget={SOFT_CHAR_BUDGET})"
        )
        if edits:
            for idx, edit in enumerate(edits, start=1):
                _olog(f"Iteration {t}: edit {idx}/{len(edits)} -> {_summarize_edit(edit)}")
        else:
            _olog(f"Iteration {t}: no deduped edits to apply")

        if edits:
            t_apply = time.perf_counter()
            before_agents_md = agents_md
            agents_md = apply_edits(agents_md, edits, config.model, timeout_s=config.timeout_s)
            _olog(
                f"Iteration {t}: apply_edits complete in "
                f"{time.perf_counter() - t_apply:.2f}s"
            )
            _olog(
                f"Iteration {t}: AGENTS.md length {len(before_agents_md)} -> {len(agents_md)} "
                f"(delta={len(agents_md) - len(before_agents_md)})"
            )
        else:
            _olog(f"Iteration {t}: no edits to apply")

        new_version = current.version + 1
        current = RepoGuidance(
            repo=config.repo, commit=config.commit,
            lines=agents_md.splitlines(), version=new_version,
        )
        current.save(guidance_dir / f"v{new_version}.json")

        state.current_version = new_version
        state.completed_iterations = t
        state.history.append({
            "version": new_version, "iteration": t,
            "probe_pool_size": len(new_probes), "new_probes": len(new_probes),
            "edits_count": len(edits),
        })
        state.save(state_path)
        _olog(
            f"Iteration {t}: state saved current_version={state.current_version} "
            f"completed_iterations={state.completed_iterations}"
        )

        _olog(f"Iteration {t} complete in {time.perf_counter() - iter_start:.2f}s; saved v{new_version}")

    final_path = out / "best_guidance.json"
    current.save(final_path)
    _olog(f"Done. Final for {config.repo}: v{current.version}")

    config_path = out / "oracle_config.json"
    config_path.write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")

    return kb, current


def _evaluate_all_probes_detailed(
    agents_md: str, probes: list[Probe], config: OracleConfig,
) -> list[ProbeResult]:
    results: list[ProbeResult] = []
    for i, probe in enumerate(probes):
        try:
            t_probe = time.perf_counter()
            _olog(f"Probe {i+1}/{len(probes)} start id={probe.id}")
            result = evaluate_probe(agents_md, probe, config.model, timeout_s=config.timeout_s)
            results.append(result)
            _olog(
                f"Probe {i+1}/{len(probes)} complete in {time.perf_counter() - t_probe:.2f}s "
                f"reviews={len(result.behavior_reviews)} edits={len(result.proposed_edits)}"
            )
        except Exception as exc:
            _olog(f"Probe {i+1}/{len(probes)} ERROR: {exc}")
            results.append(ProbeResult(
                probe_id=probe.id, task=probe.task, response="",
                behavior_reviews=[], proposed_edits=[],
                overall_notes=f"evaluation_error: {exc}",
            ))
    return results


def _collect_edits_from_results(results: list[ProbeResult]) -> list[Edit]:
    edits: list[Edit] = []
    for result in results:
        edits.extend(result.proposed_edits)
    return edits


def _dedupe_edits(edits: list[Edit]) -> list[Edit]:
    seen: set[tuple[str, str, str]] = set()
    out: list[Edit] = []
    for edit in edits:
        key = (edit.section.strip(), edit.action.strip().lower(), edit.content.strip())
        if not key[2] or key in seen:
            continue
        seen.add(key)
        out.append(Edit(section=key[0] or "General", action=key[1] or "add", content=key[2]))
    return out


def _prioritize_edits_for_iteration(
    *,
    raw_edits: list[Edit],
    deduped_edits: list[Edit],
    current_agents_md: str,
    max_edits: int,
) -> list[Edit]:
    if not deduped_edits:
        return []

    action_weight = {
        "modify": 4,
        "strengthen": 3,
        "remove": 2,
        "add": 1,
    }

    freq_by_key: dict[tuple[str, str, str], int] = {}
    for edit in raw_edits:
        key = _edit_key(edit)
        freq_by_key[key] = freq_by_key.get(key, 0) + 1

    min_support = MIN_EDIT_SUPPORT_DEFAULT
    if len(deduped_edits) <= 3:
        min_support = 1

    at_or_over_soft_budget = len(current_agents_md) >= SOFT_CHAR_BUDGET
    selected: list[Edit] = []
    seen_keys: set[tuple[str, str, str]] = set()

    ranked = sorted(
        deduped_edits,
        key=lambda e: (
            freq_by_key.get(_edit_key(e), 1),
            action_weight.get(e.action.strip().lower(), 0),
            -len(e.content),
        ),
        reverse=True,
    )

    for edit in ranked:
        key = _edit_key(edit)
        support = freq_by_key.get(key, 1)
        action = edit.action.strip().lower()

        if key in seen_keys:
            continue

        if _looks_overly_instance_specific(edit):
            continue

        if support < min_support:
            continue

        if at_or_over_soft_budget and action == "add":
            continue

        selected.append(edit)
        seen_keys.add(key)
        if len(selected) >= max_edits:
            break

    return selected


def _edit_key(edit: Edit) -> tuple[str, str, str]:
    return (
        edit.section.strip().lower(),
        edit.action.strip().lower(),
        " ".join(edit.content.strip().split()).lower(),
    )


def _looks_overly_instance_specific(edit: Edit) -> bool:
    content = edit.content.strip()
    lower = content.lower()

    path_like = len(re.findall(r"\b[\w./-]+\.(?:py|txt|md|json|yaml|yml|ini|cfg)\b", content))
    command_like = bool(re.search(r"\b(pytest|python manage\.py|grep|rg|tox|head\s+-\d+)\b", lower))
    code_block_like = "```" in content or "def " in content or "class " in content

    if code_block_like:
        return True
    if command_like and edit.action.strip().lower() == "add":
        return True
    if path_like >= 2 and edit.action.strip().lower() == "add":
        return True
    return False


def _save_probe_results_summary(probe_results, path: Path) -> None:
    summary = {
        "repo_dir": probe_results.repo_dir,
        "hub_count": len(probe_results.imports.hubs),
        "hub_files": [h.file for h in probe_results.imports.hubs],
        "symbol_count": len(probe_results.symbols.entries),
        "entry_point_count": len(probe_results.entry_points.entries),
        "cluster_count": len(probe_results.clusters.clusters),
        "chain_count": len(probe_results.clusters.chains),
        "integration_count": len(probe_results.clusters.integrations),
        "test_command": probe_results.tests.test_command,
        "test_dir_count": len(probe_results.tests.test_dirs),
        "conventions": probe_results.conventions.detected_patterns,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
