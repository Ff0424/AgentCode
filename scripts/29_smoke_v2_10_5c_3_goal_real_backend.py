"""Read-only V2-10.5c.3 real Recommendation feasibility probe.

This server-only smoke loads the published serving-v2 bundle and Product
Catalog through the production ``RecommendationService``.  It checks whether
one deterministic model-known user has recommendation candidates for the
three frozen Golden Goal requirements.  It does not initialize retrieval,
BGE-M3, verification, LangGraph, or Agent integration and writes no files.

Run from the repository root on the Ubuntu GPU server::

    /home/server/anaconda3/envs/agentrec/bin/python \
      scripts/29_smoke_v2_10_5c_3_goal_real_backend.py \
      --project-root /home/server/AgentCode \
      --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(DEFAULT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_PROJECT_ROOT))

from src.agentrec.recommendation import RecommendationService  # noqa: E402
from src.agentrec.tools import (  # noqa: E402
    RecommendationToolAdapter,
    RecommendationToolArgs,
    RecommendationToolResult,
)


@dataclass(frozen=True)
class ProbeSpec:
    """One frozen Golden Goal recommendation constraint set."""

    category: str
    max_price: float
    required_features: tuple[str, ...]


PROBES = (
    ProbeSpec("Docking Stations", 166.67, ("HDMI",)),
    ProbeSpec("Mouse", 83.33, ()),
    ProbeSpec("Headphones", 250.00, ()),
)


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="AgentRec repository root.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device passed unchanged to RecommendationService.",
    )
    return parser.parse_args()


def _run_probe(
    *,
    adapter: RecommendationToolAdapter,
    user_id: str,
    spec: ProbeSpec,
) -> RecommendationToolResult:
    result = adapter.recommend(
        user_id=user_id,
        args=RecommendationToolArgs(
            top_k=5,
            category=spec.category,
            max_price=spec.max_price,
            required_features=spec.required_features,
        ),
        excluded_parent_asins=(),
    )

    # The production Catalog filter owns price eligibility.  This assertion
    # verifies its externally visible result without duplicating its logic.
    for item in result.items:
        if item.price is None or item.price > spec.max_price:
            raise AssertionError(
                f"{spec.category} returned an item outside max_price={spec.max_price:.2f}."
            )
    return result


def _print_probe(spec: ProbeSpec, result: RecommendationToolResult) -> None:
    _section(f"Probe: {spec.category}")
    print(f"max_price={spec.max_price:.2f}")
    print(f"required_features={spec.required_features}")
    print(f"personalization_status={result.personalization_status}")
    print(f"returned_count={result.returned_count}")
    if result.items:
        top = result.items[0]
        print(f"top_parent_asin={top.parent_asin}")
        print(f"top_title={top.title}")
        print(f"top_price={top.price:.2f}")
        print(f"top_score_source={top.score_source}")


def main() -> int:
    args = _parse_args()
    project_root = args.project_root.resolve()
    serving_dir = project_root / "artifacts/recommendation/serving_v2"
    catalog_path = (
        project_root / "data/processed/recommendation/product_catalog.jsonl"
    )

    started = time.perf_counter()
    service = RecommendationService(
        bundle_dir=serving_dir,
        catalog_path=catalog_path,
        device=args.device,
    )
    service_init_seconds = time.perf_counter() - started
    adapter = RecommendationToolAdapter(service)

    artifacts = service.artifacts
    if not artifacts.model_user_ids:
        raise RuntimeError("Serving bundle contains no model-known users.")
    user_id = artifacts.model_user_ids[0]
    canonical_user_index = artifacts.resolve_canonical_user(user_id)
    if canonical_user_index is None:
        raise AssertionError("First model user is absent from canonical user lookup.")
    model_user_row = artifacts.resolve_model_user_row(canonical_user_index)
    if model_user_row != 0:
        raise AssertionError(
            "model_user_ids[0] did not resolve to model_user_row=0."
        )

    _section("Environment")
    print(f"device={args.device}")
    print(f"service_init_seconds={service_init_seconds:.6f}")

    _section("User")
    print("selection_rule=model_user_ids[0]")
    print(f"user_id={user_id}")
    print(f"canonical_user_index={canonical_user_index}")
    print(f"model_user_row={model_user_row}")

    probe_started = time.perf_counter()
    results: list[tuple[ProbeSpec, RecommendationToolResult]] = []
    for spec in PROBES:
        result = _run_probe(adapter=adapter, user_id=user_id, spec=spec)
        results.append((spec, result))
        _print_probe(spec, result)
    probe_total_seconds = time.perf_counter() - probe_started

    statuses = {
        spec.category: (
            "PASS"
            if result.returned_count > 0
            and result.personalization_status == "personalized"
            else "ZERO_CANDIDATES"
            if result.returned_count == 0
            and result.personalization_status == "personalized"
            else "UNEXPECTED_PERSONALIZATION_STATUS"
        )
        for spec, result in results
    }
    all_pass = all(status == "PASS" for status in statuses.values())

    _section("Feasibility Summary")
    for spec in PROBES:
        print(f"{spec.category}: {statuses[spec.category]}")
    print(f"probe_total_seconds={probe_total_seconds:.6f}")
    print(f"ALL_RECOMMENDATION_PROBES_PASS={all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
