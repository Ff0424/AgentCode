"""Provider-neutral planning contracts for AgentRec V2."""

from .budget_allocation import (
    DeterministicBudgetAllocator,
    GoalBudgetAllocation,
    RequirementBudgetAllocation,
)
from .contracts import (
    PlannerAction,
    PlannerDecision,
    ReplanProposalDecision,
    RequestRecommendationDecision,
    RequestUserConfirmationDecision,
    SelectCandidateDecision,
    SelectRequirementDecision,
    validate_planner_decision,
)
from .goal_contracts import (
    AllocationPreferenceType,
    GoalAllocationPreference,
    GoalRequirementProposal,
    ShoppingGoalExtractionDecision,
)
from .goal_projection import (
    GoalRequirementProjection,
    GoalToRequirementProjector,
    RequirementAllocationPreference,
)
from .planner import FakePlanner, Planner
from .providers import (
    OpenAICompatiblePlannerProvider,
    PlannerProviderError,
    PlannerSchemaError,
    PlannerTextProvider,
    PlannerTimeoutError,
    StructuredLLMPlanner,
)
from .requirement_extraction import (
    FakeRequirementExtractor,
    RequirementClarificationRequired,
    RequirementExtractionDecision,
    RequirementExtractor,
    StructuredRequirementExtractor,
)

__all__ = [
    "AllocationPreferenceType",
    "DeterministicBudgetAllocator",
    "FakePlanner",
    "FakeRequirementExtractor",
    "GoalAllocationPreference",
    "GoalBudgetAllocation",
    "GoalRequirementProjection",
    "GoalRequirementProposal",
    "GoalToRequirementProjector",
    "Planner",
    "PlannerAction",
    "PlannerDecision",
    "PlannerProviderError",
    "PlannerSchemaError",
    "PlannerTextProvider",
    "PlannerTimeoutError",
    "ReplanProposalDecision",
    "RequestRecommendationDecision",
    "RequestUserConfirmationDecision",
    "RequirementClarificationRequired",
    "RequirementAllocationPreference",
    "RequirementBudgetAllocation",
    "RequirementExtractionDecision",
    "RequirementExtractor",
    "SelectCandidateDecision",
    "SelectRequirementDecision",
    "StructuredLLMPlanner",
    "StructuredRequirementExtractor",
    "ShoppingGoalExtractionDecision",
    "OpenAICompatiblePlannerProvider",
    "validate_planner_decision",
]
