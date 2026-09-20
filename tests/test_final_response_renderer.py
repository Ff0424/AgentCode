"""Deterministic and security tests for the V2-09.6 response renderer."""

from __future__ import annotations

import unittest

from src.agentrec.domain import ItemSource
from src.agentrec.knowledge import KnowledgeChunkType
from src.agentrec.replanning import CandidateStatusSummary
from src.agentrec.response import (
    ConflictDecisionSummary,
    ConflictReason,
    ConflictResponseContext,
    DeterministicFinalResponseRenderer,
    EvidenceReference,
    ProductDecisionSummary,
    ReadyResponseContext,
    ResponseKind,
    VerifiedConstraintClaim,
)
from src.agentrec.verification import ConstraintVerificationStatus


RENDERER = DeterministicFinalResponseRenderer()


def claim(
    constraint: str = "HDMI",
    *,
    excerpt: str = "Supports HDMI output.",
    chunk_id: str = "chunk-internal-1",
) -> VerifiedConstraintClaim:
    return VerifiedConstraintClaim(
        requirement_id="dock",
        parent_asin="ASIN-DOCK",
        original_constraint=constraint,
        canonical_constraint=constraint.casefold(),
        status=ConstraintVerificationStatus.SUPPORTED,
        supporting_evidence=(EvidenceReference(
            chunk_id=chunk_id,
            chunk_type=KnowledgeChunkType.FEATURES,
            excerpt=excerpt,
        ),),
    )


def product(
    *,
    requirement_id: str = "dock",
    category: str = "USB Hubs",
    parent_asin: str = "ASIN-DOCK",
    title: str = "USB-C Dock",
    price: float = 99.995,
    quantity: int = 1,
    rationale: str = "selected_by_rank_policy",
    claims: tuple[VerifiedConstraintClaim, ...] | None = None,
) -> ProductDecisionSummary:
    return ProductDecisionSummary(
        requirement_id=requirement_id,
        category=category,
        parent_asin=parent_asin,
        title=title,
        price=price,
        quantity=quantity,
        source=ItemSource.HYBRID,
        selection_rationale=rationale,
        verified_claims=(claim(),) if claims is None else claims,
    )


def ready(*products: ProductDecisionSummary) -> ReadyResponseContext:
    return ReadyResponseContext(
        plan_id="plan",
        plan_version=len(products),
        currency="USD",
        products=products,
        total_spent=sum(value.price * value.quantity for value in products),
        remaining_budget=500.0 - sum(
            value.price * value.quantity for value in products
        ),
    )


def counts() -> CandidateStatusSummary:
    return CandidateStatusSummary(
        total_candidates=10,
        eligible_count=0,
        unverified_count=6,
        ineligible_count=4,
        supported_constraint_count=5,
        contradicted_constraint_count=4,
        unknown_constraint_count=6,
    )


def conflict(
    reason: ConflictReason,
    *,
    unknown: tuple[str, ...] = (),
    contradicted: tuple[str, ...] = (),
) -> ConflictResponseContext:
    exhausted = reason is ConflictReason.REPLAN_ATTEMPTS_EXHAUSTED
    return ConflictResponseContext(
        plan_id="plan",
        plan_version=0,
        currency="USD",
        decision=ConflictDecisionSummary(
            requirement_id="dock",
            category="Single Board Computers" if exhausted else "USB Hubs",
            required_features=("HDMI", "USB-C"),
            reason=reason,
            replan_attempts_performed=1 if exhausted else 0,
            candidate_pool_sizes=(5, 10) if exhausted else (5,),
            verification_status_counts=counts() if exhausted else None,
            failed_constraints=("HDMI", "USB-C") if exhausted else (),
            unknown_constraints=unknown,
            contradicted_constraints=contradicted,
        ),
        total_spent=0.0,
        remaining_budget=500.0,
    )


class FinalResponseRendererTests(unittest.TestCase):
    def test_ready_single_product_contains_product_quantity_price_and_budget(self) -> None:
        result = RENDERER.render(ready(product(quantity=2)))
        self.assertEqual(result.kind, ResponseKind.READY)
        self.assertIn("USB-C Dock", result.text)
        self.assertIn("数量：2", result.text)
        self.assertIn("USD 100.00", result.text)
        self.assertIn("方案总金额：USD 199.99", result.text)
        self.assertIn("剩余预算：USD 300.01", result.text)

    def test_ready_multiple_products_keep_context_order(self) -> None:
        first = product(title="First Product", claims=())
        second = product(
            requirement_id="mouse",
            category="Mice",
            parent_asin="ASIN-MOUSE",
            title="Second Product",
            price=30.0,
            claims=(),
        )
        text = RENDERER.render(ready(first, second)).text
        self.assertLess(text.index("First Product"), text.index("Second Product"))

    def test_verified_claims_are_rendered_without_evidence_payload(self) -> None:
        claims = (claim("HDMI"), claim("USB-C", chunk_id="chunk-internal-2"))
        text = RENDERER.render(ready(product(claims=claims))).text
        self.assertIn("已验证的要求：HDMI、USB-C", text)
        self.assertNotIn("Supports HDMI output", text)
        self.assertNotIn("chunk-internal", text)

    def test_no_recommendation_conflict(self) -> None:
        result = RENDERER.render(conflict(
            ConflictReason.NO_RECOMMENDATION_CANDIDATES
        ))
        self.assertEqual(result.kind, ResponseKind.CONFLICT)
        self.assertIn("当前结构化推荐约束下没有返回候选商品", result.text)
        self.assertNotIn("证据不足", result.text)
        self.assertNotIn("扩展", result.text)

    def test_retry_exhausted_conflict(self) -> None:
        text = RENDERER.render(conflict(
            ConflictReason.REPLAN_ATTEMPTS_EXHAUSTED,
            unknown=("USB-C",),
        )).text
        self.assertIn("一次有界候选池扩展", text)
        self.assertIn("仍未找到所有硬性要求都得到证据支持的商品", text)

    def test_unknown_and_contradicted_wording_remain_distinct(self) -> None:
        text = RENDERER.render(conflict(
            ConflictReason.REPLAN_ATTEMPTS_EXHAUSTED,
            unknown=("USB-C",),
            contradicted=("HDMI",),
        )).text
        self.assertIn("证据不足，无法确认满足要求", text)
        self.assertIn("- USB-C", text)
        self.assertIn("已有证据与要求存在冲突", text)
        self.assertIn("- HDMI", text)
        self.assertNotIn("USB-C 不支持", text)
        self.assertNotIn("HDMI 推荐失败", text)

    def test_security_sensitive_audit_fields_do_not_leak(self) -> None:
        sensitive = (
            "item_index=42 score=0.9 similarity=0.8 embedding model backend "
            "artifact/path chunk-internal-secret"
        )
        context = ready(product(
            rationale=sensitive,
            claims=(claim(excerpt=sensitive, chunk_id="chunk-internal-secret"),),
        ))
        text = RENDERER.render(context).text.casefold()
        for forbidden in (
            "item_index", "score", "similarity", "embedding", "model",
            "backend", "artifact/path", "chunk-internal-secret",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, text)

    def test_render_is_deterministic_and_does_not_mutate_context(self) -> None:
        context = ready(product())
        before = context.model_dump(mode="json")
        first = RENDERER.render(context)
        second = RENDERER.render(context)
        self.assertEqual(first, second)
        self.assertEqual(context.model_dump(mode="json"), before)

    def test_invalid_context_type_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            RENDERER.render({})  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
