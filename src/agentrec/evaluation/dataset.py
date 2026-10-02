"""Read-only loaders for versioned AgentRec evaluation datasets.

This module depends only on the evaluation contracts. It deliberately does not
import or initialize the Agent, recommendation, retrieval, or workflow runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .contracts import EvaluationCase


class EvaluationDatasetError(ValueError):
    """Base error for malformed or unreadable evaluation dataset files."""


class EvaluationManifestError(EvaluationDatasetError):
    """Raised when an evaluation manifest cannot be loaded as a JSON object."""


class EvaluationCaseLoadError(EvaluationDatasetError):
    """Raised when a JSONL case line is invalid or case IDs are ambiguous."""


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load one benchmark manifest as an independent JSON object.

    Args:
        path: Path to the UTF-8 JSON manifest.

    Returns:
        A newly decoded manifest dictionary.

    Raises:
        EvaluationManifestError: If the file is unreadable, invalid JSON, or
            the root JSON value is not an object.
    """

    manifest_path = Path(path)
    try:
        raw = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvaluationManifestError(
            f"Could not read evaluation manifest: {manifest_path}."
        ) from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EvaluationManifestError(
            f"Invalid JSON in evaluation manifest {manifest_path}: "
            f"line {exc.lineno}, column {exc.colno}."
        ) from exc
    if not isinstance(value, dict):
        raise EvaluationManifestError(
            f"Evaluation manifest root must be a JSON object: {manifest_path}."
        )
    return value


def load_cases(path: str | Path) -> tuple[EvaluationCase, ...]:
    """Load and validate cases from UTF-8 JSONL while preserving line order.

    Blank lines are ignored. Every non-blank line is decoded independently and
    passed to :meth:`EvaluationCase.model_validate`. Duplicate case IDs are
    rejected because they make benchmark result attribution ambiguous.

    Args:
        path: Path to the benchmark ``cases.jsonl`` file.

    Returns:
        An immutable tuple in the original JSONL record order.

    Raises:
        EvaluationCaseLoadError: If the file is unreadable, a line contains
            invalid JSON or schema data, or a case ID occurs more than once.
    """

    cases_path = Path(path)
    try:
        lines = cases_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvaluationCaseLoadError(
            f"Could not read evaluation cases: {cases_path}."
        ) from exc

    cases: list[EvaluationCase] = []
    first_line_by_id: dict[str, int] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise EvaluationCaseLoadError(
                f"Invalid JSON in evaluation cases {cases_path} at line "
                f"{line_number}, column {exc.colno}."
            ) from exc
        try:
            case = EvaluationCase.model_validate(payload)
        except ValidationError as exc:
            raise EvaluationCaseLoadError(
                f"Invalid EvaluationCase schema in {cases_path} at line "
                f"{line_number}: {exc.error_count()} validation error(s)."
            ) from exc

        previous_line = first_line_by_id.get(case.case_id)
        if previous_line is not None:
            raise EvaluationCaseLoadError(
                f"Duplicate case_id {case.case_id!r} in {cases_path} at line "
                f"{line_number}; first defined at line {previous_line}."
            )
        first_line_by_id[case.case_id] = line_number
        cases.append(case)
    return tuple(cases)


__all__ = [
    "EvaluationCaseLoadError",
    "EvaluationDatasetError",
    "EvaluationManifestError",
    "load_cases",
    "load_manifest",
]
