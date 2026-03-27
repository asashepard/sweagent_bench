# Probe-and-Refine Tuning of Repository Guidance for AI Coding Agents

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

This repository contains the code for the paper:

> **Iterative Probe-and-Refine Tuning of Repository Guidance for AI Coding Agents**
> Asa Shepard (Williams College)
> *arXiv preprint, 2026*

## Overview

Probe-and-refine tuning is a lightweight context-engineering procedure that iteratively improves repository-level guidance (AGENTS.md) for AI coding agents. Each iteration uses four single-shot LLM calls:

1. **Generate a probe** -- a synthetic bug-fix task from the repository's codebase
2. **Generate an attempted solution** given the current guidance
3. **Diagnose** what went wrong in the attempt
4. **Edit the guidance** to address the diagnosed failure

After 2--5 iterations per repository, generic instructions like *"reproduce the failure before editing"* become repo-specific strategies like *"trace through subclasses.py to map parent-child relationships."* No multi-step agent loop, no reinforcement learning, and no tool use during the tuning process.

### Key Results

SWE-bench Verified (500 instances, [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B), 200 agent steps):

| Condition | Resolve Rate | Unique Solves |
|-----------|:---:|:---:|
| No Context (baseline) | 22.8% | 18 |
| Static KB | 27.4% | 9 |
| **Probe-and-Refine** | **34.2%** | **41** |

The improvement comes from **evaluation coverage**: the guided agent produces well-formed, evaluable patches for 57% of instances vs. 37% for the baseline, while per-patch precision is constant at ~59% across all conditions.

## Setup

```bash
# Python 3.11+ required
pip install -e .
```

**Requirements:** [SWE-bench](https://github.com/princeton-nlp/SWE-bench) >= 2.0.0, [tree-sitter](https://tree-sitter.github.io/tree-sitter/) >= 0.23, Docker, and an OpenAI-compatible inference endpoint (e.g., [SGLang](https://github.com/sgl-project/sglang)).

## Reproducing Paper Results

### 1. Serve the model

The paper's experiments used SGLang to serve Qwen3.5-35B-A3B on a single GPU:

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-35B-A3B \
  --host 0.0.0.0 --port 8000 \
  --tp-size 1 \
  --context-length 16384 \
  --mem-fraction-static 0.90
```

A SLURM batch script is provided at `slurm/serve_sglang.sh`.

### 2. Run experiments

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY

# Full 500-instance run, all 3 conditions, 200 steps (Table 4 in the paper)
MODEL=Qwen/Qwen3.5-35B-A3B \
IDS_FILE=ids/verified_full_ids.txt \
CONDITIONS="no_context static_kb oracle_tuned" \
STEP_LIMIT=200 \
ORACLE_ITERS=5 \
TIMEOUT=1800 \
  bash scripts/run_experiment.sh
```

Results are written to `results/<experiment_id>/`. SLURM scripts for cluster environments are in `slurm/`.

### Experimental conditions

The three conditions correspond to the paper's terminology:

| Code | Paper | Description |
|------|-------|-------------|
| `no_context` | No Context | Issue + file tree only (baseline) |
| `static_kb` | Static KB | Issue + tree + deterministic AGENTS.md from tree-sitter analysis |
| `oracle_tuned` | Probe-and-Refine | Issue + tree + iteratively refined AGENTS.md |

### Key parameters

| Parameter | Paper value | Notes |
|-----------|:-----------:|-------|
| Model | Qwen/Qwen3.5-35B-A3B | 3B active params via MoE |
| Instances | 500 | SWE-bench Verified (`ids/verified_full_ids.txt`) |
| Step budget | 200 | Primary; also 25, 50, 100 for budget analysis |
| Oracle iterations | 5 | Per repository |
| Temperature | 0.0 / 0.7 | Inference / probe generation |
| Context window | 16k | Hard truncation |
| Repository commits | Pinned | See `configs/repos_12.json` |

### Quick smoke test

```bash
# 4 instances, fast iteration
IDS_FILE=ids/verified_smoke_4_ids.txt bash scripts/run_experiment.sh
```

## Repository Structure

```
sweagent_bench/       # Main Python package
  orchestrator.py     #   Experiment orchestration (tuning + evaluation)
  generation/         #   Patch generation via agent loop + single-shot fallback
  oracle/             #   Probe-and-refine tuning loop (Section 3.3)
  kb/                 #   Knowledge base building and AGENTS.md rendering
  probes/             #   Tree-sitter static analysis
configs/              # Pinned repository commits (repos_12.json)
ids/                  # Instance ID lists (smoke, mini, stratified, full 500)
scripts/              # Experiment runner shell scripts
slurm/                # SLURM batch job scripts
tests/                # Unit tests
```

## Citation

```bibtex
@article{shepard2026proberefine,
  title={Iterative Probe-and-Refine Tuning of Repository Guidance for AI Coding Agents},
  author={Shepard, Asa},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
