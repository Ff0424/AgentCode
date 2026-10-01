"""Lightweight tests for the Track C artifact inspection orchestration."""

from __future__ import annotations

from pathlib import Path
import ast
import unittest

from agentrec.evaluation import (
    CandidateEvidenceInspection,
    CandidateInspection,
    CatalogFact,
    EvaluationCase,
    EvidenceSnippetInspection,
    IdentityInspection,
    InspectionRequirement,
    RecommendationInspection,
    TRACK_C_CASE_IDS,
    TrackCArtifactInspectionReport,
    TrackCArtifactInspector,
    select_track_c_cases,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = REPOSITORY_ROOT / "data/evaluation/curated_v1_pilot/cases.jsonl"
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "stage5_inspect_track_c_artifacts.py"


def _load_cases() -> tuple[EvaluationCase, ...]:
    return tuple(
        EvaluationCase.model_validate_json(line)
        for line in CASES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _requirements(case: EvaluationCase) -> tuple[InspectionRequirement, ...]:
    grounding = {
        value.original_constraint: value.expected_canonical_constraint
        for value in case.expected_grounding
    }
    return tuple(
        InspectionRequirement(
            requirement_index=index,
            requirement_id=f"req-{index + 1:03d}",
            category=proposal.category,
            quantity=proposal.quantity,
            allocated_budget=100.0,
            effective_budget=100.0,
            unit_max_price=50.0,
            required_features=tuple(
                grounding[value]
                for value in proposal.required_features
                if grounding.get(value) is not None
            ),
        )
        for index, proposal in enumerate(case.expected_goal.ordered_requirement_proposals)
    )


class FakeRecommendation:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    def recommend(self, *, user_id, requirement, top_k):
        self.calls.append((user_id, requirement.requirement_index, top_k))
        count = 2 if top_k == 5 else 3
        base = requirement.requirement_index * 1000
        source = "popularity_fallback" if user_id == "fallback-user" else "hybrid"
        return RecommendationInspection(
            top_k=top_k,
            personalization_status=(
                "canonical_known_model_unseen"
                if user_id == "fallback-user"
                else "personalized"
            ),
            fallback_reason=(
                "canonical_user_not_in_model"
                if user_id == "fallback-user"
                else None
            ),
            candidates=tuple(
                CandidateInspection(
                    rank=rank,
                    item_index=base + rank,
                    parent_asin=f"P{base + rank:05d}",
                    title=f"Product {base + rank}",
                    price=10.0 + rank,
                    categories=(requirement.category,),
                    source=source,
                    price_constraint_passed=True,
                    feature_filter_constraint_passed=True,
                )
                for rank in range(1, count + 1)
            ),
        )


class FakeCatalog:
    def get_facts(self, identities):
        return tuple(
            CatalogFact(
                item_index=item_index,
                parent_asin=parent_asin,
                title=f"Catalog {parent_asin}",
                categories=("Fixture Category",),
                price=20.0,
                features=("HDMI",),
                description="Independent catalog description.",
            )
            for item_index, parent_asin in identities
        )


class FakeEvidence:
    def retrieve(self, *, case_id, requirement, candidates):
        return tuple(
            CandidateEvidenceInspection(
                item_index=value.item_index,
                parent_asin=value.parent_asin,
                query=f"Evidence for {requirement.category}",
                snippets=(
                    EvidenceSnippetInspection(
                        rank=1,
                        evidence_reference=f"chunk-{value.item_index}",
                        source_field="features",
                        text="Candidate-restricted evidence for human review.",
                        retrieval_score=0.75,
                    ),
                ),
            )
            for value in candidates
        )


class FakeIdentity:
    def inspect(self, *, user_id, user_reference):
        fallback = user_id == "fallback-user"
        return IdentityInspection(
            user_reference=user_reference,
            canonical_known=True,
            model_known=not fallback,
            personalization_status=(
                "canonical_known_model_unseen" if fallback else "personalized"
            ),
            fallback_reason="canonical_user_not_in_model" if fallback else None,
        )


class TrackCInspectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.all_cases = _load_cases()
        self.track_c = select_track_c_cases(self.all_cases)
        self.recommendation = FakeRecommendation()
        self.inspector = TrackCArtifactInspector(
            recommendation=self.recommendation,
            catalog=FakeCatalog(),
            evidence=FakeEvidence(),
            identity=FakeIdentity(),
        )

    def _report(self) -> TrackCArtifactInspectionReport:
        reports = []
        for case in self.track_c:
            fallback = case.case_id == "e2e_fallback_001"
            reports.append(
                self.inspector.inspect_case(
                    case=case,
                    user_id="fallback-user" if fallback else "known-user",
                    user_reference="fallback-fixture" if fallback else "known-fixture",
                    requirements=_requirements(case),
                )
            )
        return TrackCArtifactInspectionReport(cases=tuple(reports))

    def test_selects_exactly_six_track_c_cases(self) -> None:
        self.assertEqual(tuple(value.case_id for value in self.track_c), TRACK_C_CASE_IDS)
        self.assertTrue(all("track_c" in value.tags for value in self.track_c))

    def test_top5_top10_identity_catalog_and_evidence_projection(self) -> None:
        report = self._report()
        requirement = report.cases[0].requirements[0]
        self.assertEqual(len(requirement.top5.candidates), 2)
        self.assertEqual(len(requirement.top10.candidates), 3)
        self.assertEqual(requirement.new_candidate_identities, ((3, "P00003"),))
        self.assertEqual(
            tuple((x.item_index, x.parent_asin) for x in requirement.catalog_facts),
            ((1, "P00001"), (2, "P00002"), (3, "P00003")),
        )
        self.assertEqual(len(requirement.evidence), 3)
        self.assertEqual(requirement.evidence[0].snippets[0].source_field, "features")

    def test_fallback_lookup_projection(self) -> None:
        fallback = self._report().cases[-1]
        self.assertTrue(fallback.identity.canonical_known)
        self.assertFalse(fallback.identity.model_known)
        self.assertEqual(
            fallback.identity.personalization_status,
            "canonical_known_model_unseen",
        )
        self.assertTrue(
            all(
                candidate.source == "popularity_fallback"
                for requirement in fallback.requirements
                for candidate in requirement.top10.candidates
            )
        )

    def test_report_is_safe_and_does_not_claim_verification(self) -> None:
        serialized = self._report().model_dump_json()
        self.assertNotIn("sk-", serialized)
        self.assertNotIn("/private/", serialized)
        self.assertNotIn("C:\\", serialized)
        self.assertNotIn("embedding", serialized.casefold())
        self.assertFalse(self._report().verifier_called)
        self.assertFalse(self._report().ground_truth_mutated)

    def test_benchmark_ground_truth_is_not_mutated(self) -> None:
        before = tuple(value.model_dump_json() for value in self.all_cases)
        self._report()
        after = tuple(value.model_dump_json() for value in _load_cases())
        self.assertEqual(after, before)
        self.assertTrue(
            all(not value.independent_product_truth for value in self.track_c)
        )
        self.assertTrue(all(not value.evidence_annotations for value in self.track_c))

    def test_recommendation_calls_use_both_candidate_pool_sizes(self) -> None:
        self._report()
        observed = {top_k for _user, _index, top_k in self.recommendation.calls}
        self.assertEqual(observed, {5, 10})
        requirement_count = sum(len(_requirements(case)) for case in self.track_c)
        self.assertEqual(len(self.recommendation.calls), requirement_count * 2)

    def test_server_script_has_no_llm_workflow_or_verifier_boundary(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(SCRIPT_PATH))
        imported_names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        forbidden = {
            "AgentTaskRunner",
            "StructuredGoalExtractor",
            "EvidenceConstraintVerifier",
        }
        self.assertTrue(forbidden.isdisjoint(imported_names))
        self.assertNotIn(".run_goal(", source)
        self.assertNotIn("agentrec.verification", source)
        self.assertNotIn("cases.jsonl.write", source)


if __name__ == "__main__":
    unittest.main()
