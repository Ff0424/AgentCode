"""Deterministic V2-08.2 ShoppingPlanService tests; no runtime artifacts."""

from __future__ import annotations

import unittest

from src.agentrec.domain import ItemSource, PlanStatus, ShoppingRequirement
from src.agentrec.services import (
    CandidateNotFoundError,
    CategoryMismatchError,
    CompletedPlanMutationError,
    DuplicateItemError,
    InvalidToolItemError,
    PlanIssueCode,
    PlanNotReadyError,
    QuantityExceededError,
    RequirementConstraintError,
    ShoppingPlanService,
)
from src.agentrec.tools import (
    RecommendationToolArgs,
    RecommendationToolItem,
    RecommendationToolResult,
)


def result(asin: str, title: str, price: float | None, source: str = "hybrid"):
    return RecommendationToolResult(
        personalization_status="personalized",
        fallback_reason=None,
        returned_count=1,
        items=(RecommendationToolItem(
            rank=1,
            item_index=1,
            parent_asin=asin,
            title=title,
            price=price,
            score=0.75,
            score_source=source,
        ),),
    )


class ShoppingPlanServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ShoppingPlanService()
        self.requirements = (
            ShoppingRequirement(
                requirement_id="dock", category="Dock", max_budget=120,
                required_features=("HDMI",),
            ),
            ShoppingRequirement(requirement_id="mouse", category="Mouse", max_budget=30),
            ShoppingRequirement(requirement_id="headphones", category="Headphones"),
        )
        self.plan = self.service.create_plan(
            plan_id="golden", user_id="user", currency="USD",
            total_budget=500, requirements=self.requirements,
        )

    def select(self, plan, requirement_id, price, *, source="hybrid"):
        features = ("HDMI",) if requirement_id == "dock" else ()
        max_price = {"dock": 120, "mouse": 30}.get(requirement_id)
        return self.service.select_item(
            plan,
            requirement_id=requirement_id,
            recommendation_result=result(
                f"P-{requirement_id}-{price}", requirement_id.title(), price, source
            ),
            recommendation_args=RecommendationToolArgs(
                category=requirement_id.title(), max_price=max_price,
                required_features=features,
            ),
            selected_parent_asin=f"P-{requirement_id}-{price}",
            selected_reason="Deterministic selection",
        )

    def test_tool_projection_and_slot_are_deterministic(self) -> None:
        updated = self.select(self.plan, "dock", 120)
        item = updated.selected_items[0]
        self.assertEqual(item.slot_id, "dock:1")
        self.assertEqual(item.parent_asin, "P-dock-120")
        self.assertEqual(item.title, "Dock")
        self.assertEqual(item.price, 120)
        self.assertEqual(item.recommendation_score, 0.75)
        self.assertTrue(item.constraints_satisfied)
        self.assertEqual(item.source, ItemSource.HYBRID)

    def test_requirement_constraints_cannot_be_silently_relaxed(self) -> None:
        candidate = result("P", "Dock", 100)
        base = dict(
            plan=self.plan, requirement_id="dock", recommendation_result=candidate,
            selected_parent_asin="P", selected_reason="reason",
        )
        with self.assertRaises(RequirementConstraintError):
            self.service.select_item(
                **base,
                recommendation_args=RecommendationToolArgs(
                    category="Dock", required_features=("HDMI",)
                ),
            )
        with self.assertRaises(RequirementConstraintError):
            self.service.select_item(
                **base,
                recommendation_args=RecommendationToolArgs(
                    category="Dock", max_price=121, required_features=("HDMI",)
                ),
            )
        with self.assertRaises(RequirementConstraintError):
            self.service.select_item(
                **{**base, "recommendation_result": result("P", "Dock", 121)},
                recommendation_args=RecommendationToolArgs(
                    category="Dock", max_price=120, required_features=("HDMI",)
                ),
            )

    def test_category_features_quantity_duplicate_and_membership(self) -> None:
        with self.assertRaises(CategoryMismatchError):
            self.service.select_item(
                self.plan, requirement_id="dock", recommendation_result=result("P", "Dock", 50),
                recommendation_args=RecommendationToolArgs(
                    category="Mouse", max_price=120, required_features=("HDMI",)
                ), selected_parent_asin="P", selected_reason="x",
            )
        with self.assertRaises(RequirementConstraintError):
            self.service.select_item(
                self.plan, requirement_id="dock", recommendation_result=result("P", "Dock", 50),
                recommendation_args=RecommendationToolArgs(category="Dock", max_price=120),
                selected_parent_asin="P", selected_reason="x",
            )
        with self.assertRaises(QuantityExceededError):
            self.service.select_item(
                self.plan, requirement_id="dock", recommendation_result=result("P", "Dock", 50),
                recommendation_args=RecommendationToolArgs(
                    category="Dock", max_price=120, required_features=("HDMI",)
                ), selected_parent_asin="P", quantity=2, selected_reason="x",
            )
        with self.assertRaises(CandidateNotFoundError):
            self.service.select_item(
                self.plan, requirement_id="dock",
                recommendation_result=RecommendationToolResult(
                    personalization_status="unknown", fallback_reason="none",
                    returned_count=0, items=(),
                ),
                recommendation_args=RecommendationToolArgs(
                    category="Dock", max_price=120, required_features=("HDMI",)
                ), selected_parent_asin="P", selected_reason="x",
            )
        selected = self.select(self.plan, "dock", 100)
        with self.assertRaises(DuplicateItemError):
            self.service.select_item(
                selected, requirement_id="mouse",
                recommendation_result=result("P-dock-100", "Mouse", 20),
                recommendation_args=RecommendationToolArgs(category="Mouse", max_price=30),
                selected_parent_asin="P-dock-100", selected_reason="x",
            )

    def test_invalid_tool_fields_are_rejected(self) -> None:
        with self.assertRaises(InvalidToolItemError):
            self.service.select_item(
                self.plan, requirement_id="headphones",
                recommendation_result=result("P", "Headphones", None),
                recommendation_args=RecommendationToolArgs(category="Headphones"),
                selected_parent_asin="P", selected_reason="x",
            )
        with self.assertRaises(InvalidToolItemError):
            self.service.select_item(
                self.plan, requirement_id="headphones",
                recommendation_result=result("P", "Headphones", 100, "replacement"),
                recommendation_args=RecommendationToolArgs(category="Headphones"),
                selected_parent_asin="P", selected_reason="x",
            )

    def test_replace_preserves_slot_and_real_provenance(self) -> None:
        selected = self.select(self.plan, "headphones", 180)
        replaced = self.service.replace_item(
            selected, slot_id="headphones:1",
            recommendation_result=result("P-NEW", "Headphones", 250, "popularity_fallback"),
            recommendation_args=RecommendationToolArgs(category="Headphones"),
            selected_parent_asin="P-NEW", selected_reason="Better option",
        )
        self.assertEqual(replaced.selected_items[0].slot_id, "headphones:1")
        self.assertEqual(replaced.selected_items[0].source, ItemSource.POPULARITY_FALLBACK)
        self.assertNotEqual(replaced.selected_items[0].source, ItemSource.REPLACEMENT)

    def test_evaluation_projects_domain_facts_without_mutation(self) -> None:
        evaluation = self.service.evaluate_plan(self.plan)
        self.assertEqual(evaluation.total_spent, self.plan.total_spent)
        self.assertEqual(evaluation.remaining_budget, self.plan.remaining_budget)
        self.assertEqual(evaluation.is_over_budget, self.plan.is_over_budget)
        self.assertEqual(evaluation.plan_version, self.plan.version)
        self.assertEqual(
            [issue.code for issue in evaluation.issues],
            [PlanIssueCode.MISSING_REQUIRED_ITEM] * 3,
        )
        self.assertEqual(evaluation, self.service.evaluate_plan(self.plan))

    def test_golden_demo_ready_conflict_ready(self) -> None:
        ready = self.select(self.select(self.select(self.plan, "dock", 120), "mouse", 30), "headphones", 180)
        self.assertEqual(ready.status, PlanStatus.READY)
        self.assertEqual(ready.total_spent, 330)

        conflict = self.service.replace_item(
            ready, slot_id="headphones:1",
            recommendation_result=result("P-H400", "Headphones", 400),
            recommendation_args=RecommendationToolArgs(category="Headphones"),
            selected_parent_asin="P-H400", selected_reason="Premium replacement",
        )
        conflict_eval = self.service.evaluate_plan(conflict)
        self.assertEqual(conflict.status, PlanStatus.CONFLICT)
        self.assertEqual(conflict.total_spent, 550)
        self.assertIn(PlanIssueCode.TOTAL_BUDGET_EXCEEDED, {x.code for x in conflict_eval.issues})

        ready_again = self.service.replace_item(
            conflict, slot_id="headphones:1",
            recommendation_result=result("P-H250", "Headphones", 250),
            recommendation_args=RecommendationToolArgs(category="Headphones"),
            selected_parent_asin="P-H250", selected_reason="Budget-safe replacement",
        )
        self.assertEqual(ready_again.status, PlanStatus.READY)
        self.assertEqual(ready_again.total_spent, 400)
        completed = self.service.mark_completed(ready_again)
        self.assertEqual(completed.status, PlanStatus.COMPLETED)
        with self.assertRaises(CompletedPlanMutationError):
            self.service.remove_item(completed, slot_id="dock:1")

    def test_not_ready_cannot_complete_and_success_increments_once(self) -> None:
        with self.assertRaises(PlanNotReadyError):
            self.service.mark_completed(self.plan)
        selected = self.select(self.plan, "dock", 100)
        self.assertEqual(selected.version, self.plan.version + 1)


if __name__ == "__main__":
    unittest.main()
