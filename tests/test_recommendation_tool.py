"""Deterministic tests for the Agent-facing recommendation tool adapter."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from pydantic import ValidationError

from src.agentrec.recommendation.contracts import (
    RecommendationResult,
    RecommendedProduct,
)
from src.agentrec.tools import (
    RecommendationToolAdapter,
    RecommendationToolArgs,
    RecommendationToolResult,
)


def _item(*, source: str = "hybrid") -> RecommendedProduct:
    return RecommendedProduct(
        rank=1,
        item_index=42,
        parent_asin="B000TEST42",
        title="Test Product",
        price=49.99,
        categories=("Electronics",),
        features=("HDMI",),
        score=0.75,
        score_components={"internal": 123.0},
        score_source=source,
    )


def _result(
    *,
    status: str = "personalized",
    fallback_reason: str | None = None,
    source: str = "hybrid",
    empty: bool = False,
) -> RecommendationResult:
    return RecommendationResult(
        canonical_user_index=7 if status != "unknown_user" else None,
        model_user_row=2 if status == "personalized" else None,
        personalization_status=status,
        applied_constraints={},
        artifact_version="fixture-v2",
        items=() if empty else (_item(source=source),),
        fallback_reason=fallback_reason,
    )


class FakeService:
    def __init__(self, result: RecommendationResult | None = None) -> None:
        self.result = result or _result()
        self.requests = []

    def recommend(self, request):
        self.requests.append(request)
        return self.result


class RecommendationToolTests(unittest.TestCase):
    def test_valid_args_map_to_request_with_system_and_workflow_fields(self) -> None:
        service = FakeService()
        adapter = RecommendationToolAdapter(service)
        args = RecommendationToolArgs(
            top_k=5,
            category=" Docking Stations ",
            max_price=100,
            required_features=(" HDMI ", "USB-C"),
        )
        adapter.recommend(
            user_id=" system-user ",
            args=args,
            excluded_parent_asins=(" P1 ", "P2"),
        )
        request = service.requests[0]
        self.assertEqual(request.user_id, "system-user")
        self.assertEqual(request.top_k, 5)
        self.assertEqual(request.category, "Docking Stations")
        self.assertEqual(request.max_price, 100.0)
        self.assertEqual(request.required_features, ("HDMI", "USB-C"))
        self.assertEqual(request.excluded_parent_asins, ("P1", "P2"))

    def test_top_k_bounds_are_rejected_before_service(self) -> None:
        service = FakeService()
        adapter = RecommendationToolAdapter(service)
        for invalid in (0, 51):
            with self.subTest(top_k=invalid), self.assertRaises(ValidationError):
                args = RecommendationToolArgs(top_k=invalid)
                adapter.recommend(user_id="u", args=args)
        self.assertEqual(service.requests, [])

    def test_nonpositive_max_price_is_rejected(self) -> None:
        for invalid in (0, -1.0):
            with self.subTest(max_price=invalid), self.assertRaises(ValidationError):
                RecommendationToolArgs(max_price=invalid)

    def test_required_feature_limit_and_empty_feature_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            RecommendationToolArgs(required_features=tuple(f"f{i}" for i in range(21)))
        with self.assertRaises(ValidationError):
            RecommendationToolArgs(required_features=("HDMI", " "))

    def test_llm_schema_excludes_system_and_internal_fields(self) -> None:
        properties = RecommendationToolArgs.model_json_schema()["properties"]
        self.assertEqual(
            set(properties), {"top_k", "category", "max_price", "required_features"}
        )
        for forbidden in (
            "user_id",
            "excluded_parent_asins",
            "alpha",
            "model_user_row",
            "semantic_profile",
            "train_seen_indices",
            "device",
            "embeddings",
            "zscore_stats",
            "popularity_order",
        ):
            self.assertNotIn(forbidden, properties)

    def test_invalid_system_user_and_exclusion_inputs_do_not_call_service(self) -> None:
        service = FakeService()
        adapter = RecommendationToolAdapter(service)
        args = RecommendationToolArgs()
        for user_id in ("", "   "):
            with self.subTest(user_id=user_id), self.assertRaises(ValueError):
                adapter.recommend(user_id=user_id, args=args)
        with self.assertRaises(TypeError):
            adapter.recommend(user_id="u", args=args, excluded_parent_asins=["P1"])
        with self.assertRaises(ValueError):
            adapter.recommend(
                user_id="u",
                args=args,
                excluded_parent_asins=tuple(f"P{i}" for i in range(101)),
            )
        with self.assertRaises(ValueError):
            adapter.recommend(user_id="u", args=args, excluded_parent_asins=("",))
        self.assertEqual(service.requests, [])

    def test_personalized_projection_is_compact_and_preserves_source(self) -> None:
        adapter = RecommendationToolAdapter(FakeService(_result()))
        result = adapter.recommend(user_id="known", args=RecommendationToolArgs())
        self.assertIsInstance(result, RecommendationToolResult)
        self.assertEqual(result.personalization_status, "personalized")
        self.assertEqual(result.returned_count, 1)
        item = result.items[0]
        self.assertEqual(item.score_source, "hybrid")
        self.assertEqual(
            set(type(item).model_fields),
            {"rank", "parent_asin", "title", "price", "score", "score_source"},
        )
        self.assertFalse(hasattr(item, "score_components"))

    def test_fallback_status_reason_and_score_source_are_preserved(self) -> None:
        service_result = _result(
            status="canonical_known_model_unseen",
            fallback_reason="canonical_user_not_in_model",
            source="popularity_fallback",
        )
        result = RecommendationToolAdapter(FakeService(service_result)).recommend(
            user_id="canonical", args=RecommendationToolArgs()
        )
        self.assertEqual(result.personalization_status, "canonical_known_model_unseen")
        self.assertEqual(result.fallback_reason, "canonical_user_not_in_model")
        self.assertEqual(result.items[0].score_source, "popularity_fallback")

    def test_empty_result_is_a_valid_tool_result(self) -> None:
        empty = _result(status="unknown_user", fallback_reason="unknown", empty=True)
        result = RecommendationToolAdapter(FakeService(empty)).recommend(
            user_id="unknown", args=RecommendationToolArgs()
        )
        self.assertEqual(result.returned_count, 0)
        self.assertEqual(result.items, ())

    def test_service_exception_propagates_and_is_not_converted_to_empty(self) -> None:
        class FailingService:
            def recommend(self, _request):
                raise RuntimeError("serving artifact failure")

        adapter = RecommendationToolAdapter(FailingService())
        with self.assertRaisesRegex(RuntimeError, "serving artifact failure"):
            adapter.recommend(user_id="u", args=RecommendationToolArgs())

    def test_adapter_requires_and_reuses_injected_service(self) -> None:
        with self.assertRaises(TypeError):
            RecommendationToolAdapter(None)
        service = FakeService()
        adapter = RecommendationToolAdapter(service)
        adapter.recommend(user_id="u", args=RecommendationToolArgs())
        adapter.recommend(user_id="u", args=RecommendationToolArgs())
        self.assertIs(adapter._service, service)
        self.assertEqual(len(service.requests), 2)

    def test_tool_models_are_immutable(self) -> None:
        args = RecommendationToolArgs()
        result = RecommendationToolAdapter(FakeService()).recommend(user_id="u", args=args)
        with self.assertRaises((ValidationError, FrozenInstanceError)):
            args.top_k = 2
        with self.assertRaises((ValidationError, FrozenInstanceError)):
            result.items[0].rank = 2


if __name__ == "__main__":
    unittest.main()
