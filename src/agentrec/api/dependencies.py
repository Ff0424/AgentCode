"""Construction of the shared real AgentRec API runtime.

This module owns dependency assembly only. It does not expose HTTP routes or
execute a shopping request during import.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..evidence import GroundedEvidenceService
from ..integration import AgentTaskRunner
from ..planning import (
    GoalRequirementProposal,
    OpenAICompatiblePlannerProvider,
    ShoppingGoalExtractionDecision,
    StructuredGoalExtractor,
    StructuredLLMPlanner,
    StructuredRequirementExtractor,
)
from ..recommendation import RecommendationService
from ..retrieval import (
    BGEM3QueryEncoder,
    ChunkRetrievalArtifacts,
    ChunkRetriever,
    NumPyExactBackend,
    ValidationMode,
)
from ..services import ShoppingPlanService
from ..tools import RecommendationToolAdapter
from ..verification import EvidenceConstraintVerifier


MODEL = "deepseek-v4-flash"

CATEGORY_ALIASES = {
    "docking station": "Docking Stations",
    "docking stations": "Docking Stations",
    "mouse": "Mice",
    "mice": "Mice",
    "headphone": "Headphones",
    "headphones": "Headphones",
}

_REQUIREMENT_PROPOSAL_ALIASES = (
    "ordered_requirement_proposals",
    "requirements",
)


@dataclass(frozen=True)
class AgentRuntime:
    """Process-scoped Agent runner and its deterministic default user."""

    runner: AgentTaskRunner
    default_user_id: str


class _GoalSchemaNormalizingProvider:
    """Normalize approved top-level aliases before strict goal validation.

    ``StructuredGoalExtractor`` validates provider JSON internally, so schema
    drift must be corrected at this provider boundary. Unknown fields and
    ambiguous payloads are deliberately retained and therefore still rejected
    by the frozen Pydantic contract.
    """

    def __init__(self, provider: OpenAICompatiblePlannerProvider) -> None:
        self._provider = provider

    def complete(
        self,
        *,
        messages: tuple[dict[str, str], ...],
        timeout_seconds: float,
    ) -> str:
        raw = self._provider.complete(
            messages=messages,
            timeout_seconds=timeout_seconds,
        )
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            # Preserve malformed responses so StructuredGoalExtractor produces
            # its existing strict, sanitized schema error.
            return raw
        if not isinstance(payload, dict):
            return raw

        present_aliases = tuple(
            alias for alias in _REQUIREMENT_PROPOSAL_ALIASES if alias in payload
        )
        if "requirement_proposals" not in payload and len(present_aliases) == 1:
            alias = present_aliases[0]
            payload["requirement_proposals"] = payload.pop(alias)

        # Canonical+alias or multiple-alias collisions remain untouched. The
        # extra="forbid" contract will fail closed instead of choosing a value.
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )


class NormalizedGoalExtractor:
    """Normalize extracted categories to the frozen Catalog taxonomy."""

    def __init__(self, extractor: StructuredGoalExtractor) -> None:
        self._extractor = extractor

    def extract(self, *, user_request: str) -> ShoppingGoalExtractionDecision:
        decision = self._extractor.extract(user_request=user_request)
        proposals = tuple(
            GoalRequirementProposal(
                category=CATEGORY_ALIASES.get(
                    " ".join(proposal.category.split()).casefold(),
                    proposal.category,
                ),
                quantity=proposal.quantity,
                max_budget=proposal.max_budget,
                required_features=proposal.required_features,
                soft_preferences=proposal.soft_preferences,
                priority=proposal.priority,
            )
            for proposal in decision.requirement_proposals
        )
        return ShoppingGoalExtractionDecision(
            total_budget=decision.total_budget,
            requirement_proposals=proposals,
            allocation_preferences=decision.allocation_preferences,
            clarification_needed=decision.clarification_needed,
            clarification_question=decision.clarification_question,
        )


def build_runtime(
    project_root: Path,
    device: str = "cuda:0",
) -> AgentRuntime:
    """Build one shared real-model runtime without executing a request."""

    root = project_root.resolve()
    provider = OpenAICompatiblePlannerProvider(model=MODEL)
    goal_extractor = StructuredGoalExtractor(
        provider=_GoalSchemaNormalizingProvider(provider),
        timeout_seconds=30,
        schema_retries=1,
    )
    planner = StructuredLLMPlanner(
        provider=provider,
        timeout_seconds=30,
        schema_retries=1,
    )

    recommendation_service = RecommendationService(
        bundle_dir=root / "artifacts/recommendation/serving_v2",
        catalog_path=root / "data/processed/recommendation/product_catalog.jsonl",
        device=device,
    )
    recommendation_tool = RecommendationToolAdapter(recommendation_service)

    retrieval_artifacts = ChunkRetrievalArtifacts(
        retrieval_dir=root / "artifacts/recommendation/retrieval",
        knowledge_dir=root / "artifacts/recommendation/knowledge",
        validation_mode=ValidationMode.STRICT,
        expected_dimension=1024,
    )
    query_encoder = BGEM3QueryEncoder(
        model_path=root / "models/bge-m3",
        device=device,
        use_fp16=True,
        batch_size=8,
        max_length=2048,
        embedding_dimension=1024,
    )
    retriever = ChunkRetriever(
        encoder=query_encoder,
        backend=NumPyExactBackend(retrieval_artifacts.embeddings),
        artifacts=retrieval_artifacts,
    )

    model_user_ids = recommendation_service.artifacts.model_user_ids
    if not model_user_ids:
        raise RuntimeError("Serving bundle contains no model-known users.")
    default_user_id = model_user_ids[0]

    runner = AgentTaskRunner(
        # AgentTaskRunner still requires the legacy single-requirement boundary;
        # run_goal() uses the separately injected goal extractor.
        requirement_extractor=StructuredRequirementExtractor(
            provider=provider,
            timeout_seconds=30,
            schema_retries=1,
        ),
        planner=planner,
        recommendation_tool=recommendation_tool,
        evidence_service=GroundedEvidenceService(retriever=retriever),
        verification_service=EvidenceConstraintVerifier(),
        shopping_plan_service=ShoppingPlanService(),
        goal_extractor=NormalizedGoalExtractor(goal_extractor),
    )
    return AgentRuntime(
        runner=runner,
        default_user_id=default_user_id,
    )
