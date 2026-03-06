"""Continuous oracle tuning loop for AGENTS.md."""
from __future__ import annotations

from dataclasses import asdict
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
from sweagent_bench.llm.openai_compat import chat_completion
from sweagent_bench.oracle.apply import apply_edits
from sweagent_bench.oracle.diagnose import diagnose_failures
from sweagent_bench.oracle.judge import evaluate_probe
from sweagent_bench.oracle.probes import generate_probes
from sweagent_bench.oracle.schema import Edit, OracleConfig, OracleState, Probe, ProbeResult
from sweagent_bench.probes import run_all_probes


MAX_EDITS_PER_ITERATION = 5
RESERVED_CHAR_BUFFER = 640
SOFT_CHAR_BUDGET = AGENTS_MD_CHAR_BUDGET - RESERVED_CHAR_BUFFER
ORACLE_PROBE_EXECUTION_MODE = "single_shot"
TARGET_PROBES_PER_ITERATION = 10
MAX_PROBE_TOPUP_ATTEMPTS = 3

_EDIT_PRIORITIZER_SYSTEM = """\
You are selecting the best AGENTS.md edits from candidate proposals.

Return ONLY valid JSON with this shape:
{{
    "selected_indices": [0, 4, 2],
    "notes": "optional short rationale"
}}

Rules:
- Select up to {max_edits} candidates by index.
- Prefer reusable repo-level guidance, minimal scope, and high-signal improvements.
- Avoid duplicates/near-duplicates and avoid overly instance-specific suggestions.
- Prefer edits that improve evidence-first localization, dependency tracing, and targeted validation.
- Do not invent new indices.
"""

_EDIT_PRIORITIZER_USER = """\
CURRENT AGENTS.MD:
---
{agents_md}
---

CANDIDATE EDITS (JSON array):
{candidates_json}

Select the top edits now.
"""


def _olog(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [oracle] {msg}", flush=True)


def _summarize_edit(edit: Edit, max_len: int = 180) -> str:
    """Compact one-line summary of an edit for logging."""
    content = " ".join(edit.content.split())
    if len(content) > max_len:
        content = content[: max_len - 3] + "..."
    return f"section={edit.section!r} action={edit.action!r} content={content!r}"


def _runner_note_value(overall_notes: str, key: str) -> str:
    match = re.search(rf"{re.escape(key)}=([^;|]+)", overall_notes or "")
    return match.group(1).strip() if match else ""


def _normalize_probe_text(text: str) -> str:
    return " ".join(text.lower().split())


def _probe_signature(task: str) -> str:
    return _normalize_probe_text(task)


def _load_prior_probe_signatures(out_dir: Path, completed_iterations: int) -> set[str]:
    signatures: set[str] = set()
    for idx in range(1, completed_iterations + 1):
        probe_path = out_dir / f"iteration_{idx}_probes.json"
        if not probe_path.exists():
            continue
        try:
            data = json.loads(probe_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        probes = data if isinstance(data, list) else data.get("probes_kept", [])
        if not isinstance(probes, list):
            continue
        for item in probes:
            if not isinstance(item, dict):
                continue
            task = str(item.get("task", "")).strip()
            if task:
                signatures.add(_probe_signature(task))
    return signatures


def _load_prior_probe_tasks(out_dir: Path, completed_iterations: int) -> list[str]:
    tasks: list[str] = []
    seen: set[str] = set()
    for idx in range(1, completed_iterations + 1):
        probe_path = out_dir / f"iteration_{idx}_probes.json"
        if not probe_path.exists():
            continue
        try:
            data = json.loads(probe_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        probes = data if isinstance(data, list) else data.get("probes_kept", [])
        if not isinstance(probes, list):
            continue
        for item in probes:
            if not isinstance(item, dict):
                continue
            task = str(item.get("task", "")).strip()
            if not task:
                continue
            sig = _probe_signature(task)
            if sig in seen:
                continue
            seen.add(sig)
            tasks.append(task)
    return tasks


def _write_iteration_artifacts(
    out_dir: Path,
    iteration: int,
    probes_payload: dict,
    edits_selected: list[Edit],
    guidance_before: str,
    guidance_after: str,
    summary: dict,
) -> None:
    (out_dir / f"iteration_{iteration}_probes.json").write_text(
        json.dumps(probes_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / f"iteration_{iteration}_edits_selected.json").write_text(
        json.dumps([asdict(e) for e in edits_selected], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / f"iteration_{iteration}_guidance_before.md").write_text(
        guidance_before,
        encoding="utf-8",
    )
    (out_dir / f"iteration_{iteration}_guidance_after.md").write_text(
        guidance_after,
        encoding="utf-8",
    )
    (out_dir / f"iteration_{iteration}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


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

    prior_probe_signatures = _load_prior_probe_signatures(out, state.completed_iterations)
    prior_probe_tasks = _load_prior_probe_tasks(out, state.completed_iterations)
    no_change_streak = 0

    start_iter = state.completed_iterations + 1
    for t in range(start_iter, config.iterations + 1):
        iter_start = time.perf_counter()
        _olog(f"Iteration {t}/{config.iterations} started for {config.repo}")
        effective_probe_timeout_s = max(30, int(config.probe_timeout_s))
        _olog(
            f"Iteration {t}: probe execution mode={ORACLE_PROBE_EXECUTION_MODE} "
            f"timeout_s={effective_probe_timeout_s}"
        )
        guidance_before = agents_md
        iter_stop_reason = ""

        t_probe_gen = time.perf_counter()
        generation_prior = [Probe(id=f"prior_{i}", task=task) for i, task in enumerate(prior_probe_tasks)]
        generated_probes = generate_probes(
            kb,
            config.model,
            agents_md,
            prior_probes=generation_prior,
            timeout_s=config.timeout_s,
            max_probes=TARGET_PROBES_PER_ITERATION,
        )
        deduped_probes: list[Probe] = []
        duplicate_probe_count = 0
        for probe in generated_probes:
            sig = _probe_signature(probe.task)
            if sig in prior_probe_signatures:
                duplicate_probe_count += 1
                continue
            prior_probe_signatures.add(sig)
            prior_probe_tasks.append(probe.task)
            deduped_probes.append(probe)

        topup_attempts = 0
        while len(deduped_probes) < TARGET_PROBES_PER_ITERATION and topup_attempts < MAX_PROBE_TOPUP_ATTEMPTS:
            remaining = TARGET_PROBES_PER_ITERATION - len(deduped_probes)
            generation_prior = [Probe(id=f"prior_{i}", task=task) for i, task in enumerate(prior_probe_tasks)]
            topup_batch = generate_probes(
                kb,
                config.model,
                agents_md,
                prior_probes=generation_prior,
                timeout_s=config.timeout_s,
                max_probes=remaining,
            )
            if not topup_batch:
                break
            topup_attempts += 1
            generated_probes.extend(topup_batch)
            for probe in topup_batch:
                sig = _probe_signature(probe.task)
                if sig in prior_probe_signatures:
                    duplicate_probe_count += 1
                    continue
                prior_probe_signatures.add(sig)
                prior_probe_tasks.append(probe.task)
                deduped_probes.append(probe)

        if len(deduped_probes) < TARGET_PROBES_PER_ITERATION:
            raise RuntimeError(
                f"Failed to enforce {TARGET_PROBES_PER_ITERATION} probes "
                f"(kept={len(deduped_probes)}, dropped={duplicate_probe_count})"
            )
        _olog(
            f"Iteration {t}: generated {len(generated_probes)} probes in "
            f"{time.perf_counter() - t_probe_gen:.2f}s (pool={len(generated_probes)})"
        )
        _olog(
            f"Iteration {t}: probe dedup dropped={duplicate_probe_count} "
            f"kept={len(deduped_probes)} target={TARGET_PROBES_PER_ITERATION} "
            f"topup_attempts={topup_attempts}"
        )
        if deduped_probes:
            probe_ids = ", ".join(p.id for p in deduped_probes)
            _olog(f"Iteration {t}: probe ids this round: {probe_ids}")
        else:
            iter_stop_reason = "no_probes_after_dedup"
            _olog(f"Iteration {t}: no probes kept after dedup; continuing with empty result set")

        t_eval = time.perf_counter()
        results = _evaluate_all_probes_detailed(agents_md, deduped_probes, config)
        probe_count_evaluated = len(results)
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
            model=config.model,
            timeout_s=config.timeout_s,
        )
        _olog(
            f"Iteration {t}: prioritized edits={len(edits)} "
            f"(cap={MAX_EDITS_PER_ITERATION}, soft_budget={SOFT_CHAR_BUDGET})"
        )
        if edits:
            for idx, edit in enumerate(edits, start=1):
                _olog(f"Iteration {t}: edit {idx}/{len(edits)} -> {_summarize_edit(edit)}")
        else:
            if not iter_stop_reason:
                iter_stop_reason = "no_selected_edits"
            _olog(
                f"Iteration {t}: prioritization selected zero edits "
                f"(reason={iter_stop_reason})"
            )

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
            _olog(f"Iteration {t}: no edits applied (reason=no_selected_edits)")

        guidance_changed = agents_md != guidance_before
        if not guidance_changed and not iter_stop_reason:
            iter_stop_reason = "no_guidance_change"

        no_change_counted = False
        if probe_count_evaluated == 0:
            no_change_streak = 0
        elif not edits or not guidance_changed:
            no_change_counted = True
            no_change_streak += 1
        else:
            no_change_streak = 0

        summary: dict = {
            "probe_count_generated": len(generated_probes),
            "probe_count_after_dedup": len(deduped_probes),
            "probe_count_evaluated": probe_count_evaluated,
            "selected_edit_count": len(edits),
            "guidance_changed": guidance_changed,
            "no_change_counted": no_change_counted,
            "probe_execution_mode": ORACLE_PROBE_EXECUTION_MODE,
            "effective_probe_timeout_s": effective_probe_timeout_s,
        }
        if iter_stop_reason:
            summary["stop_reason"] = iter_stop_reason

        probes_payload = {
            "probe_count_generated": len(generated_probes),
            "duplicate_count_dropped": duplicate_probe_count,
            "probe_count_kept": len(deduped_probes),
            "probes_kept": [asdict(p) for p in deduped_probes],
        }
        _write_iteration_artifacts(
            out_dir=out,
            iteration=t,
            probes_payload=probes_payload,
            edits_selected=edits,
            guidance_before=guidance_before,
            guidance_after=agents_md,
            summary=summary,
        )

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
            "probe_pool_size": len(deduped_probes), "new_probes": len(deduped_probes),
            "edits_count": len(edits),
        })
        state.save(state_path)
        _olog(
            f"Iteration {t}: state saved current_version={state.current_version} "
            f"completed_iterations={state.completed_iterations}"
        )

        _olog(f"Iteration {t} complete in {time.perf_counter() - iter_start:.2f}s; saved v{new_version}")

        if no_change_streak >= 2:
            stop_reason = "2 consecutive no-change iterations with evaluated probes"
            _olog(f"Iteration {t}: stopping oracle early ({stop_reason})")
            final_summary_path = out / f"iteration_{t}_summary.json"
            try:
                final_summary = json.loads(final_summary_path.read_text(encoding="utf-8"))
            except Exception:
                final_summary = summary
            final_summary["stop_reason"] = stop_reason
            final_summary_path.write_text(
                json.dumps(final_summary, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            break

    final_path = out / "best_guidance.json"
    current.save(final_path)
    _olog(f"Done. Final for {config.repo}: v{current.version}")

    config_path = out / "oracle_config.json"
    config_path.write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")

    return kb, current


def _evaluate_all_probes_detailed(
    agents_md: str, probes: list[Probe], config: OracleConfig,
) -> list[ProbeResult]:
    effective_probe_timeout_s = max(30, int(config.probe_timeout_s))
    results: list[ProbeResult] = []
    for i, probe in enumerate(probes):
        try:
            t_probe = time.perf_counter()
            _olog(f"Probe {i+1}/{len(probes)} start id={probe.id}")
            result = evaluate_probe(
                agents_md,
                probe,
                config.model,
                repo=config.repo,
                commit=config.commit,
                timeout_s=config.timeout_s,
                probe_timeout_s=effective_probe_timeout_s,
                api_base=config.api_base,
            )
            results.append(result)
            runner_status = ""
            runner_error = ""
            token_total = 0
            if result.overall_notes:
                runner_status = _runner_note_value(result.overall_notes, "runner_status")
                runner_error = _runner_note_value(result.overall_notes, "runner_error")
                token_total_raw = _runner_note_value(result.overall_notes, "runner_token_total")
                token_total = int(token_total_raw) if token_total_raw.isdigit() else 0
            status_suffix = f" status={runner_status or 'unknown'}"
            if runner_status.lower() == "error":
                status_suffix += f" error={runner_error or 'unknown'}"
            else:
                status_suffix += f" tokens={token_total}"
            _olog(
                f"Probe {i+1}/{len(probes)} complete in {time.perf_counter() - t_probe:.2f}s "
                f"reviews={len(result.behavior_reviews)} edits={len(result.proposed_edits)}"
                f"{status_suffix}"
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
        run_status = _runner_note_value(result.overall_notes, "runner_status").lower()
        if run_status == "error":
            continue
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
    model: str,
    timeout_s: int,
) -> list[Edit]:
    if not raw_edits:
        return []

    candidates: list[dict] = []
    for idx, edit in enumerate(raw_edits):
        candidates.append(
            {
                "index": idx,
                "section": edit.section,
                "action": edit.action,
                "content": edit.content,
            }
        )

    messages = [
        {"role": "system", "content": _EDIT_PRIORITIZER_SYSTEM.format(max_edits=max_edits)},
        {
            "role": "user",
            "content": _EDIT_PRIORITIZER_USER.format(
                agents_md=current_agents_md,
                candidates_json=json.dumps(candidates, ensure_ascii=False, indent=2),
            ),
        },
    ]

    selected_indices: list[int] = []
    try:
        raw = chat_completion(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
            timeout_s=timeout_s,
        )
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        obj = json.loads(text)
        idxs = obj.get("selected_indices", []) if isinstance(obj, dict) else []
        if isinstance(idxs, list):
            for v in idxs:
                if isinstance(v, int):
                    selected_indices.append(v)
    except Exception:
        selected_indices = []

    selected: list[Edit] = []
    seen_keys: set[tuple[str, str, str]] = set()

    for idx in selected_indices:
        if idx < 0 or idx >= len(raw_edits):
            continue
        edit = raw_edits[idx]
        key = (
            edit.section.strip().lower(),
            edit.action.strip().lower(),
            " ".join(edit.content.strip().split()).lower(),
        )
        if not key[2] or key in seen_keys:
            continue
        selected.append(edit)
        seen_keys.add(key)
        if len(selected) >= max_edits:
            break

    if not selected:
        for edit in deduped_edits[:max_edits]:
            key = (
                edit.section.strip().lower(),
                edit.action.strip().lower(),
                " ".join(edit.content.strip().split()).lower(),
            )
            if not key[2] or key in seen_keys:
                continue
            selected.append(edit)
            seen_keys.add(key)
            if len(selected) >= max_edits:
                break

    return selected


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
