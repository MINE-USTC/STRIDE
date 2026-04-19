"""Path helpers for STRIDE outputs (repository / code root)."""

from __future__ import annotations

import os
import re
from pathlib import Path

_STRIDE_ROOT = Path(__file__).resolve().parent


def stride_root() -> Path:
    return _STRIDE_ROOT


def repo_root() -> Path:
    """Same as code root when the repo is the flat package layout."""
    return _STRIDE_ROOT


def default_run_name(input_jsonl: str) -> str:
    """Default output run id: basename of the jsonl file without extension."""
    stem = Path(input_jsonl).resolve().stem
    safe = re.sub(r"[^\w.\-]+", "_", stem)
    return safe if safe else "run"


def meta_plan_jsonl_name(write_base: str) -> str:
    return f"{write_base}.jsonl"


def meta_plan_relative_for_supervisor(write_base: str) -> str:
    """Relative path segment under meta_plans/<run_name>/ passed to Supervisor."""
    return meta_plan_jsonl_name(write_base)


def meta_plan_version_from_plan_arg(plan_file_name: str) -> str:
    base = os.path.basename(plan_file_name)
    return base.split("_")[-1].replace(".jsonl", "")


def supervisor_output_basename(
    write_base: str,
    top_k_docs: int,
    max_iteration: int = 5,
    failed_threshold: int = 2,
) -> str:
    name = f"{write_base}_top{top_k_docs}"
    if max_iteration != 5:
        name += f"-iter{max_iteration}"
    if failed_threshold != 2:
        name += f"-f{failed_threshold}"
    return name


def supervisor_result_relpath(
    plan_file_name: str,
    write_base: str,
    top_k_docs: int,
    max_iteration: int = 5,
    failed_threshold: int = 2,
) -> str:
    ver = meta_plan_version_from_plan_arg(plan_file_name)
    fname = supervisor_output_basename(
        write_base,
        top_k_docs,
        max_iteration,
        failed_threshold,
    )
    return f"{ver}/{fname}.jsonl"


def resolve_supervisor_jsonl(
    run_name: str,
    plan_file_name: str,
    write_base: str,
    *,
    top_k_docs: int = 5,
    max_iteration: int = 5,
    failed_threshold: int = 2,
) -> str:
    """Absolute path to supervisor output jsonl for a run."""
    root = stride_root() / "output"
    ver = meta_plan_version_from_plan_arg(plan_file_name)
    fname = supervisor_output_basename(
        write_base,
        top_k_docs,
        max_iteration,
        failed_threshold,
    )
    return str(root / run_name / ver / f"{fname}.jsonl")


def default_ft_reasoner_output() -> str:
    """Default directory for supervised LoRA checkpoints."""
    return str(_STRIDE_ROOT / "ft_models" / "reasoner")


def default_ft_dpo_output() -> str:
    """Default directory for DPO checkpoints."""
    return str(_STRIDE_ROOT / "ft_models" / "dpo")
