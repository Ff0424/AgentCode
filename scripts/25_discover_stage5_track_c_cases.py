"""Discover real-artifact candidates for pending Stage 5 Track C ground truth.

This is a read-only discovery harness. It loads the published Recommendation,
Catalog, Knowledge, and Retrieval artifacts, then asks the production evidence
verifier to classify candidate-restricted evidence. It never runs the Agent
workflow and never writes benchmark annotations.

Ubuntu server example::

    python scripts/25_discover_stage5_track_c_cases.py \
      --project-root /home/server/AgentCode --device cuda:0
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_PROJECT_ROOT = SCRIPT_PATH.parents[1]
for import_root in (DEFAULT_PROJECT_ROOT, DEFAULT_PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from agentrec.domain import ShoppingRequirement  # noqa: E402
from agentrec.evidence import (  # noqa: E402
    EvidenceCandidate,
    EvidenceEmptyError,
    GroundedEvidenceService,
)
from agentrec.feature_vocabulary import DEFAULT_FEATURE_ALIASES  # noqa: E402
from agentrec.recommendation import RecommendationRequest, RecommendationService  # noqa: E402
from agentrec.retrieval import (  # noqa: E402
    BGEM3QueryEncoder,
    ChunkRetrievalArtifacts,
    ChunkRetriever,
    NumPyExactBackend,
    ValidationMode,
)
from agentrec.tools import RecommendationToolAdapter, RecommendationToolArgs  # noqa: E402
from agentrec.verification import (  # noqa: E402
    ConstraintVerificationStatus,
    EvidenceConstraintVerifier,
)


OUTPUT_JSON = "stage5_track_c_case_discovery.json"
OUTPUT_TEXT = "stage5_track_c_case_discovery.txt"
DEFAULT_BUDGETS = (50.0, 100.0, 250.0, 500.0)
MAX_CATEGORIES_PER_FEATURE = 20
MIN_STRUCTURED_ELIGIBLE = 10
MAX_RECOMMENDATION_CASES = 160
MAX_EVIDENCE_RETRIEVALS = 320
_MAX_TEXT = 4000
_SECRET_PATTERN = re.compile(r"\b(?:sk-[A-Za-z0-9_-]+|AKIA[A-Z0-9]+)\b")
_WINDOWS_PATH_PATTERN = re.compile(r"(?i)\b[a-z]:[\\/][^\s]+")
_LINUX_PATH_PATTERN = re.compile(r"(?<!\w)/(?:home|root|opt|srv|mnt)/[^\s]+")


def _safe_text(value: object, *, max_length: int = _MAX_TEXT) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    text = _SECRET_PATTERN.sub("[REDACTED_SECRET]", text)
    text = _WINDOWS_PATH_PATTERN.sub("[REDACTED_PATH]", text)
    text = _LINUX_PATH_PATTERN.sub("[REDACTED_PATH]", text)
    return text[:max_length] or None


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())


@dataclass(frozen=True)
class Candidate:
    rank: int
    item_index: int
    parent_asin: str
    title: str | None
    price: float | None
    source: str

    @property
    def identity(self) -> tuple[int, str]:
        return self.item_index, self.parent_asin


class CatalogRecords:
    """Compact read-only offsets plus category names from the canonical Catalog."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offsets: dict[tuple[int, str], int] = {}
        categories: set[str] = set()
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                record = json.loads(line)
                item_index = record.get("item_index")
                parent_asin = record.get("parent_asin")
                if isinstance(item_index, bool) or not isinstance(item_index, int):
                    raise ValueError("Catalog contains an invalid item_index.")
                if not isinstance(parent_asin, str) or not parent_asin:
                    raise ValueError("Catalog contains an invalid parent_asin.")
                identity = (item_index, parent_asin)
                if identity in self.offsets:
                    raise ValueError("Catalog contains duplicate canonical identity.")
                self.offsets[identity] = offset
                categories.update(_strings(record.get("categories")))
        self.categories = tuple(sorted(categories, key=lambda value: (value.casefold(), value)))

    def get(self, identity: tuple[int, str]) -> dict[str, Any]:
        offset = self.offsets.get(identity)
        if offset is None:
            raise ValueError("Recommendation identity is absent from Product Catalog.")
        with self.path.open("rb") as handle:
            handle.seek(offset)
            record = json.loads(handle.readline())
        if (record.get("item_index"), record.get("parent_asin")) != identity:
            raise ValueError("Catalog byte-offset identity mismatch.")
        price = record.get("price")
        return {
            "item_index": identity[0],
            "parent_asin": identity[1],
            "title": _safe_text(record.get("title")),
            "price": (
                float(price)
                if isinstance(price, (int, float))
                and not isinstance(price, bool)
                and math.isfinite(float(price))
                else None
            ),
            "categories": _strings(record.get("categories")),
            "features": _strings(record.get("features")),
            "description": _safe_text(record.get("description")),
        }


class DiscoveryRuntime:
    def __init__(self, *, project_root: Path, device: str) -> None:
        catalog_path = project_root / "data/processed/recommendation/product_catalog.jsonl"
        self.service = RecommendationService(
            bundle_dir=project_root / "artifacts/recommendation/serving_v2",
            catalog_path=catalog_path,
            device=device,
        )
        self.adapter = RecommendationToolAdapter(self.service)
        artifacts = ChunkRetrievalArtifacts(
            retrieval_dir=project_root / "artifacts/recommendation/retrieval",
            knowledge_dir=project_root / "artifacts/recommendation/knowledge",
            validation_mode=ValidationMode.STRICT,
            expected_dimension=1024,
        )
        encoder = BGEM3QueryEncoder(
            model_path=project_root / "models/bge-m3",
            device=device,
            use_fp16=True,
            batch_size=8,
            max_length=2048,
            embedding_dimension=1024,
        )
        retriever = ChunkRetriever(
            encoder=encoder,
            backend=NumPyExactBackend(artifacts.embeddings),
            artifacts=artifacts,
        )
        self.evidence = GroundedEvidenceService(retriever=retriever)
        self.verifier = EvidenceConstraintVerifier()
        self.catalog = CatalogRecords(catalog_path)
        self.evidence_retrievals = 0

    def recommend(
        self, *, user_id: str, category: str, feature: str, budget: float, top_k: int
    ) -> tuple[Candidate, ...]:
        result = self.adapter.recommend(
            user_id=user_id,
            args=RecommendationToolArgs(
                top_k=top_k,
                category=category,
                max_price=budget,
                required_features=(feature,),
            ),
            excluded_parent_asins=(),
        )
        if result.personalization_status != "personalized":
            raise RuntimeError("Discovery user unexpectedly left personalized mode.")
        return tuple(
            Candidate(
                rank=value.rank,
                item_index=value.item_index,
                parent_asin=value.parent_asin,
                title=_safe_text(value.title),
                price=value.price,
                source=value.score_source,
            )
            for value in result.items
        )

    def eligible_count(self, *, category: str, feature: str, budget: float) -> int:
        request = RecommendationRequest(
            user_id="discovery-count-only",
            top_k=10,
            category=category,
            max_price=budget,
            required_features=(feature,),
            excluded_parent_asins=(),
        )
        return int(self.service.catalog.build_eligible_mask(request).sum())

    def inspect_evidence(
        self, *, category: str, feature: str, budget: float, candidate: Candidate
    ) -> dict[str, Any]:
        if self.evidence_retrievals >= MAX_EVIDENCE_RETRIEVALS:
            raise RuntimeError("Evidence retrieval bound reached.")
        self.evidence_retrievals += 1
        requirement = ShoppingRequirement(
            requirement_id="discovery-requirement",
            category=category,
            quantity=1,
            max_budget=budget,
            required_features=(feature,),
            soft_preferences=(),
            priority=3,
        )
        try:
            evidence = self.evidence.retrieve(
                plan_id="stage5-track-c-discovery",
                plan_version=0,
                requirement=requirement,
                candidates=(
                    EvidenceCandidate(
                        item_index=candidate.item_index,
                        parent_asin=candidate.parent_asin,
                    ),
                ),
            )
        except EvidenceEmptyError:
            return {"status": "unknown", "reason": "no_evidence_chunks", "snippets": []}
        verification = self.verifier.verify(
            requirement=requirement,
            evidence=evidence,
            current_plan_id="stage5-track-c-discovery",
            current_plan_version=0,
        )
        constraint = verification.candidates[0].constraints[0]
        return {
            "status": constraint.status.value,
            "reason": constraint.reason.value,
            "canonical_constraint": constraint.canonical_constraint,
            "supporting_chunk_ids": list(constraint.supporting_chunk_ids),
            "contradicting_chunk_ids": list(constraint.contradicting_chunk_ids),
            "query": evidence.query,
            "snippets": [
                {
                    "rank": value.rank,
                    "chunk_id": value.chunk_id,
                    "chunk_type": value.chunk_type.value,
                    "text": _safe_text(value.text),
                    "similarity_score": value.similarity_score,
                }
                for value in evidence.products[0].snippets
            ],
        }


def _candidate_dict(value: Candidate) -> dict[str, Any]:
    return {
        "rank": value.rank,
        "item_index": value.item_index,
        "parent_asin": value.parent_asin,
        "title": value.title,
        "price": value.price,
        "source": value.source,
    }


def _shortlist(
    runtime: DiscoveryRuntime, *, feature: str, budgets: tuple[float, ...]
) -> tuple[tuple[str, int], ...]:
    maximum = max(budgets)
    values = []
    for category in runtime.catalog.categories:
        count = runtime.eligible_count(category=category, feature=feature, budget=maximum)
        if count >= MIN_STRUCTURED_ELIGIBLE:
            values.append((category, count))
    values.sort(key=lambda value: (-value[1], value[0].casefold(), value[0]))
    return tuple(values[:MAX_CATEGORIES_PER_FEATURE])


def _case_record(
    *,
    target: str,
    category: str,
    feature: str,
    budget: float,
    top5: tuple[Candidate, ...],
    top10: tuple[Candidate, ...],
    catalog: CatalogRecords,
    evidence: dict[tuple[int, str], dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    top5_ids = tuple(value.identity for value in top5)
    new = tuple(value for value in top10 if value.identity not in set(top5_ids))
    relevant = tuple(dict.fromkeys((*top5, *new)))
    return {
        "target": target,
        "category": category,
        "required_feature": feature,
        "evaluation_user_identity_policy": "serving_v2.model_user_ids[0] (model-known)",
        "budget": budget,
        "top5_candidate_identities": [list(value.identity) for value in top5],
        "top10_candidate_identities": [list(value.identity) for value in top10],
        "NEW_TOP10": [list(value.identity) for value in new],
        "top5_candidates": [_candidate_dict(value) for value in top5],
        "top10_candidates": [_candidate_dict(value) for value in top10],
        "relevant_catalog_facts": [catalog.get(value.identity) for value in relevant],
        "retrieved_evidence": [
            {
                "item_index": identity[0],
                "parent_asin": identity[1],
                **value,
            }
            for identity, value in evidence.items()
        ],
        "why_case_qualifies": reason,
        "deterministic_from_current_artifacts": True,
    }


def _discover(
    runtime: DiscoveryRuntime, *, budgets: tuple[float, ...]
) -> dict[str, Any]:
    if not runtime.service.artifacts.model_user_ids:
        raise RuntimeError("Serving bundle contains no model-known users.")
    user_id = runtime.service.artifacts.model_user_ids[0]
    features = tuple(DEFAULT_FEATURE_ALIASES.keys())
    shortlists = {feature: _shortlist(runtime, feature=feature, budgets=budgets) for feature in features}
    contradicted: dict[str, Any] | None = None
    recoverable: dict[str, Any] | None = None
    recommendation_cases = 0
    evaluated: list[dict[str, Any]] = []

    # Interleave canonical features and category ranks. This prevents one feature
    # from consuming the evidence budget before the other is inspected. Higher
    # budgets run first because they are most likely to expose a real Top5/Top10
    # boundary; the complete caller-supplied budget set remains deterministic.
    search_specs: list[tuple[str, str, int, float]] = []
    for category_rank in range(MAX_CATEGORIES_PER_FEATURE):
        for feature in features:
            values = shortlists[feature]
            if category_rank >= len(values):
                continue
            category, eligible_at_max = values[category_rank]
            search_specs.extend(
                (feature, category, eligible_at_max, budget)
                for budget in reversed(budgets)
            )

    for feature, category, eligible_at_max, budget in search_specs:
                if recommendation_cases >= MAX_RECOMMENDATION_CASES:
                    break
                recommendation_cases += 1
                top5 = runtime.recommend(
                    user_id=user_id, category=category, feature=feature, budget=budget, top_k=5
                )
                top10 = runtime.recommend(
                    user_id=user_id, category=category, feature=feature, budget=budget, top_k=10
                )
                top5_ids = tuple(value.identity for value in top5)
                top10_ids = tuple(value.identity for value in top10)
                prefix_aligned = top10_ids[: len(top5_ids)] == top5_ids
                new = tuple(value for value in top10 if value.identity not in set(top5_ids))
                evaluated.append(
                    {
                        "category": category,
                        "required_feature": feature,
                        "budget": budget,
                        "structured_eligible_at_max_budget": eligible_at_max,
                        "top5_count": len(top5),
                        "top10_count": len(top10),
                        "new_top10_count": len(new),
                        "prefix_aligned": prefix_aligned,
                    }
                )
                if not top10 or not prefix_aligned:
                    continue

                evidence_by_id: dict[tuple[int, str], dict[str, Any]] = {}
                # One evidence read can contribute to either qualification target.
                for candidate in top10:
                    if runtime.evidence_retrievals >= MAX_EVIDENCE_RETRIEVALS:
                        break
                    evidence_by_id[candidate.identity] = runtime.inspect_evidence(
                        category=category,
                        feature=feature,
                        budget=budget,
                        candidate=candidate,
                    )

                if contradicted is None:
                    contradicted_candidates = tuple(
                        value
                        for value in top10
                        if evidence_by_id.get(value.identity, {}).get("status")
                        == ConstraintVerificationStatus.CONTRADICTED.value
                    )
                    if contradicted_candidates:
                        matched = contradicted_candidates[0]
                        contradicted = _case_record(
                            target="CONTRADICTED",
                            category=category,
                            feature=feature,
                            budget=budget,
                            top5=top5,
                            top10=top10,
                            catalog=runtime.catalog,
                            evidence={matched.identity: evidence_by_id[matched.identity]},
                            reason=(
                                "Production verification found explicit contradicting evidence "
                                "and no supporting evidence for the canonical hard feature."
                            ),
                        )

                if recoverable is None and new:
                    top5_all_unknown = bool(top5) and all(
                        evidence_by_id.get(value.identity, {}).get("status")
                        == ConstraintVerificationStatus.UNKNOWN.value
                        for value in top5
                    )
                    supported_new = tuple(
                        value
                        for value in new
                        if evidence_by_id.get(value.identity, {}).get("status")
                        == ConstraintVerificationStatus.SUPPORTED.value
                    )
                    if top5_all_unknown and supported_new:
                        relevant = (*top5, supported_new[0])
                        recoverable = _case_record(
                            target="UNKNOWN_TO_REPLAN_TO_SUPPORTED",
                            category=category,
                            feature=feature,
                            budget=budget,
                            top5=top5,
                            top10=top10,
                            catalog=runtime.catalog,
                            evidence={
                                value.identity: evidence_by_id[value.identity]
                                for value in relevant
                            },
                            reason=(
                                "Every Top5 candidate is UNKNOWN from retrieved evidence; "
                                "Top10 adds a new candidate whose retrieved evidence is "
                                "explicitly SUPPORTED for the same canonical hard feature."
                            ),
                        )
                if contradicted is not None and recoverable is not None:
                    break

    return {
        "report_version": "1.0",
        "purpose": "real_artifact_track_c_case_discovery_only",
        "ground_truth_mutated": False,
        "application_state_mutated": False,
        "canonical_features": list(features),
        "budgets": list(budgets),
        "bounds": {
            "min_structured_eligible": MIN_STRUCTURED_ELIGIBLE,
            "max_categories_per_feature": MAX_CATEGORIES_PER_FEATURE,
            "max_recommendation_cases": MAX_RECOMMENDATION_CASES,
            "max_evidence_retrievals": MAX_EVIDENCE_RETRIEVALS,
        },
        "category_shortlists": {
            feature: [
                {"category": category, "eligible_at_max_budget": count}
                for category, count in values
            ]
            for feature, values in shortlists.items()
        },
        "recommendation_cases_evaluated": recommendation_cases,
        "evidence_retrievals_performed": runtime.evidence_retrievals,
        "search_trace": evaluated,
        "discovered_cases": {
            "contradicted": contradicted,
            "unknown_to_replan_to_supported": recoverable,
        },
        "no_valid_contradicted_case_found": contradicted is None,
        "no_valid_unknown_to_replan_to_supported_case_found": recoverable is None,
    }


def _render_text(report: dict[str, Any]) -> str:
    lines = [
        "=== Stage 5 Track C Real Case Discovery ===",
        "ground_truth_mutated=false",
        "application_state_mutated=false",
        f"canonical_features={tuple(report['canonical_features'])}",
        f"budgets={tuple(report['budgets'])}",
        f"recommendation_cases_evaluated={report['recommendation_cases_evaluated']}",
        f"evidence_retrievals_performed={report['evidence_retrievals_performed']}",
    ]
    for key in ("contradicted", "unknown_to_replan_to_supported"):
        value = report["discovered_cases"][key]
        lines.extend(("", f"=== {key.upper()} ==="))
        if value is None:
            lines.append("NO VALID REAL CASE FOUND WITHIN BOUNDED SEARCH")
            continue
        lines.extend(
            (
                f"category={value['category']}",
                f"required_feature={value['required_feature']}",
                f"evaluation_user_identity_policy={value['evaluation_user_identity_policy']}",
                f"budget={value['budget']:.2f}",
                f"top5_candidate_identities={value['top5_candidate_identities']}",
                f"top10_candidate_identities={value['top10_candidate_identities']}",
                f"NEW_TOP10={value['NEW_TOP10']}",
                f"why_case_qualifies={value['why_case_qualifies']}",
                "deterministic_from_current_artifacts=true",
                "relevant_catalog_facts=",
                json.dumps(value["relevant_catalog_facts"], ensure_ascii=False, indent=2),
                "retrieved_evidence=",
                json.dumps(value["retrieved_evidence"], ensure_ascii=False, indent=2),
            )
        )
    return "\n".join(lines) + "\n"


def _parse_budgets(raw: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise argparse.ArgumentTypeError("budgets must be positive finite comma-separated values")
    return tuple(sorted(set(values)))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--budgets",
        type=_parse_budgets,
        default=DEFAULT_BUDGETS,
        help="Positive comma-separated per-item budgets (default: 50,100,250,500).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    project_root = args.project_root.resolve()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    runtime = DiscoveryRuntime(project_root=project_root, device=args.device)
    report = _discover(runtime, budgets=args.budgets)

    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / OUTPUT_JSON
    text_path = output_dir / OUTPUT_TEXT
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    text_path.write_text(_render_text(report), encoding="utf-8")

    print("STAGE 5 TRACK C REAL CASE DISCOVERY: COMPLETE")
    print(f"contradicted_found={report['discovered_cases']['contradicted'] is not None}")
    print(
        "unknown_to_replan_to_supported_found="
        f"{report['discovered_cases']['unknown_to_replan_to_supported'] is not None}"
    )
    print(f"recommendation_cases={report['recommendation_cases_evaluated']}")
    print(f"evidence_retrievals={report['evidence_retrievals_performed']}")
    print(f"json_report=output/{OUTPUT_JSON}")
    print(f"text_report=output/{OUTPUT_TEXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
