"""Deterministic Shopping Plan and AgentState domain tests for V2-08.1."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.agentrec.domain import (
    AgentState,
    ItemSource,
    PlanStatus,
    RequirementStatus,
    ShoppingPlan,
    ShoppingPlanItem,
    ShoppingRequirement,
)


def golden_requirements() -> tuple[ShoppingRequirement, ...]:
    return (
        ShoppingRequirement(
            requirement_id="dock",
            category="Docking Station",
            required_features=("HDMI",),
            priority=4,
        ),
        ShoppingRequirement(
            requirement_id="mouse",
            category="Mouse",
            soft_preferences=("prefer cheaper",),
            priority=2,
        ),
        ShoppingRequirement(
            requirement_id="headphones",
            category="Headphones",
            soft_preferences=("higher budget priority",),
            priority=5,
        ),
    )


def plan() -> ShoppingPlan:
    return ShoppingPlan(
        plan_id="plan-1",
        user_id="user-1",
        currency="usd",
        total_budget=500,
        requirements=golden_requirements(),
    )


def item(
    requirement_id: str,
    category: str,
    parent_asin: str,
    price: float,
    *,
    slot_id: str | None = None,
    constraints_satisfied: bool = True,
    source: ItemSource = ItemSource.HYBRID,
) -> ShoppingPlanItem:
    return ShoppingPlanItem(
        slot_id=slot_id or f"slot-{requirement_id}",
        requirement_id=requirement_id,
        category=category,
        parent_asin=parent_asin,
        title=f"Synthetic {category}",
        price=price,
        quantity=1,
        selected_reason="Synthetic deterministic selection",
        constraints_satisfied=constraints_satisfied,
        source=source,
        recommendation_score=0.5,
    )


def complete_golden_plan(headphones_price: float = 180) -> ShoppingPlan:
    value = plan()
    value = value.add_item(item("dock", "Docking Station", "P-DOCK", 120))
    value = value.add_item(item("mouse", "Mouse", "P-MOUSE", 30))
    return value.add_item(
        item("headphones", "Headphones", "P-HEADPHONES", headphones_price)
    )


class ShoppingDomainTests(unittest.TestCase):
    def test_requirement_separates_hard_and_soft_constraints(self) -> None:
        dock, mouse, headphones = golden_requirements()
        self.assertEqual(dock.required_features, ("HDMI",))
        self.assertEqual(dock.soft_preferences, ())
        self.assertEqual(mouse.required_features, ())
        self.assertEqual(mouse.soft_preferences, ("prefer cheaper",))
        self.assertEqual(headphones.soft_preferences, ("higher budget priority",))

    def test_invalid_budget_and_price_are_rejected(self) -> None:
        for invalid in (0, -1, float("nan"), float("inf"), None):
            with self.subTest(budget=invalid), self.assertRaises((ValidationError, TypeError)):
                ShoppingPlan(
                    plan_id="p",
                    user_id="u",
                    total_budget=invalid,
                    requirements=golden_requirements(),
                )
        with self.assertRaises((ValidationError, TypeError)):
            item("dock", "Docking Station", "P", None)
        with self.assertRaises(ValidationError):
            item("dock", "Docking Station", "P", -10)

    def test_add_item_updates_version_requirement_and_plan_status(self) -> None:
        original = plan()
        updated = original.add_item(item("dock", "Docking Station", "P-DOCK", 120))
        self.assertEqual(original.selected_items, ())
        self.assertEqual(updated.version, 1)
        self.assertEqual(updated.status, PlanStatus.IN_PROGRESS)
        self.assertEqual(updated.requirements[0].status, RequirementStatus.SATISFIED)

    def test_replace_removes_old_selection_without_residual(self) -> None:
        original = plan().add_item(item("dock", "Docking Station", "P-OLD", 120))
        replacement = item(
            "dock",
            "Docking Station",
            "P-NEW",
            100,
            source=ItemSource.REPLACEMENT,
        )
        updated = original.replace_item("slot-dock", replacement)
        self.assertEqual(updated.version, 2)
        self.assertEqual([value.parent_asin for value in updated.selected_items], ["P-NEW"])
        self.assertNotIn("P-OLD", {value.parent_asin for value in updated.selected_items})

    def test_remove_updates_requirement_and_plan_status(self) -> None:
        ready = complete_golden_plan()
        updated = ready.remove_item("slot-mouse")
        self.assertEqual(updated.status, PlanStatus.IN_PROGRESS)
        self.assertEqual(updated.requirements[1].status, RequirementStatus.PENDING)
        self.assertEqual(updated.total_spent, 300)
        self.assertEqual(updated.version, ready.version + 1)

    def test_quantity_rule_rejects_excess_selection(self) -> None:
        value = plan().add_item(item("dock", "Docking Station", "P1", 100, slot_id="s1"))
        with self.assertRaisesRegex(ValidationError, "quantity exceeds"):
            value.add_item(item("dock", "Docking Station", "P2", 100, slot_id="s2"))

    def test_golden_budget_totals_and_valid_ready_state(self) -> None:
        value = complete_golden_plan()
        self.assertEqual(value.total_spent, 330)
        self.assertEqual(value.remaining_budget, 170)
        self.assertFalse(value.is_over_budget)
        self.assertFalse(value.has_hard_constraint_conflict)
        self.assertEqual(value.status, PlanStatus.READY)
        self.assertTrue(value.is_valid)

    def test_over_budget_golden_variant_becomes_conflict(self) -> None:
        value = complete_golden_plan(headphones_price=400)
        self.assertEqual(value.total_spent, 550)
        self.assertEqual(value.remaining_budget, -50)
        self.assertTrue(value.is_over_budget)
        self.assertTrue(value.has_hard_constraint_conflict)
        self.assertEqual(value.status, PlanStatus.CONFLICT)
        self.assertFalse(value.is_valid)

    def test_category_mismatch_and_unknown_requirement_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationError, "category does not match"):
            plan().add_item(item("dock", "Mouse", "P1", 20))
        with self.assertRaisesRegex(ValidationError, "unknown requirement"):
            plan().add_item(item("missing", "Mouse", "P2", 20))

    def test_duplicate_slot_and_product_are_rejected(self) -> None:
        value = plan().add_item(item("dock", "Docking Station", "P1", 100))
        with self.assertRaisesRegex(ValidationError, "slot_id"):
            value.add_item(item("mouse", "Mouse", "P2", 20, slot_id="slot-dock"))
        with self.assertRaisesRegex(ValidationError, "parent_asin"):
            value.add_item(item("mouse", "Mouse", "P1", 20))

    def test_unsatisfied_hard_constraint_sets_conflict(self) -> None:
        value = plan().add_item(
            item(
                "dock",
                "Docking Station",
                "P1",
                100,
                constraints_satisfied=False,
            )
        )
        self.assertEqual(value.requirements[0].status, RequirementStatus.CONFLICT)
        self.assertEqual(value.status, PlanStatus.CONFLICT)

    def test_ready_plan_can_transition_to_completed(self) -> None:
        ready = complete_golden_plan()
        completed = ready.mark_completed()
        self.assertEqual(completed.status, PlanStatus.COMPLETED)
        self.assertEqual(completed.version, ready.version + 1)
        self.assertTrue(completed.is_valid)
        with self.assertRaises(ValueError):
            completed.remove_item("slot-mouse")

    def test_serialization_round_trip_is_deterministic(self) -> None:
        original = complete_golden_plan()
        payload = original.to_domain_dict()
        restored = ShoppingPlan.from_domain_dict(payload)
        self.assertEqual(restored, original)
        self.assertEqual(restored.model_dump_json(), original.model_dump_json())
        self.assertNotIn("total_spent", payload)
        self.assertNotIn("remaining_budget", payload)
        self.assertNotIn("is_over_budget", payload)

    def test_domain_models_are_frozen(self) -> None:
        value = complete_golden_plan()
        with self.assertRaises(ValidationError):
            value.total_budget = 1000
        with self.assertRaises(ValidationError):
            value.selected_items[0].price = 1
        with self.assertRaises(ValidationError):
            value.requirements[0].status = RequirementStatus.PENDING

    def test_agent_state_is_minimal_validated_and_serializable(self) -> None:
        value = AgentState(
            user_id="user-1",
            session_id="session-1",
            shopping_plan=plan(),
            current_requirement_id="dock",
            conversation_turn=1,
            last_tool_result={"returned_count": 3},
            pending_action="recommend dock",
        )
        self.assertEqual(value.shopping_plan.plan_id, "plan-1")
        self.assertEqual(
            AgentState.model_validate_json(value.model_dump_json()), value
        )
        forbidden_runtime_fields = {
            "llm_client",
            "recommendation_service",
            "database_connection",
            "gpu_tensor",
        }
        self.assertTrue(forbidden_runtime_fields.isdisjoint(type(value).model_fields))
        with self.assertRaises(ValidationError):
            AgentState(
                user_id="different-user",
                session_id="s",
                shopping_plan=plan(),
            )


if __name__ == "__main__":
    unittest.main()
