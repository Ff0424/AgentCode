"""Read-only smoke for real Goal extraction through the AgentRec runtime.

This server-side validation uses the real OpenAI-compatible Goal extractor,
published recommendation and retrieval artifacts, local BGE-M3, evidence
verification, and the existing AgentTaskRunner workflow. It writes no files.

Run from the repository root on the Ubuntu GPU server::

    python scripts/31_smoke_real_goal_e2e.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_PROJECT_ROOT = SCRIPT_PATH.parents[1]
SRC_DIR = DEFAULT_PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agentrec.evidence import GroundedEvidenceService  # noqa: E402
from agentrec.integration import AgentTaskRunner, GoalExecutionStatus  # noqa: E402
from agentrec.planning import (  # noqa: E402
    OpenAICompatiblePlannerProvider,
    StructuredGoalExtractor,
    StructuredLLMPlanner,
    StructuredRequirementExtractor,
)
from agentrec.recommendation import RecommendationService  # noqa: E402
from agentrec.retrieval import (  # noqa: E402
    BGEM3QueryEncoder,
    ChunkRetrievalArtifacts,
    ChunkRetriever,
    NumPyExactBackend,
    ValidationMode,
)
from agentrec.services import ShoppingPlanService  # noqa: E402
from agentrec.tools import RecommendationToolAdapter  # noqa: E402
from agentrec.verification import EvidenceConstraintVerifier  # noqa: E402
from agentrec.workflows import WorkflowRoute  # noqa: E402


MODEL = "deepseek-v4-flash"
REQUEST = """Return JSON only. Follow this exact schema and exact field names:
{
  "total_budget": number,
  "requirement_proposals": [
    {
      "category": string,
      "quantity": integer,
      "required_features": []
    }
  ],
  "allocation_preferences": [
    {
      "target_index": integer,
      "preference": string
    }
  ],
  "clarification_needed": boolean,
  "clarification_question": string or null
}
Do not add extra fields. Do not rename any field. In particular, use
"requirement_proposals", never "ordered_requirement_proposals".

User request:
I need a $500 shopping setup with a docking station,
a mouse, and headphones.
The docking station must support HDMI.
Save more on the mouse and spend more on headphones."""


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
        help="CUDA device used by recommendation serving and local BGE-M3.",
    )
    return parser.parse_args()


def _build_runner(*, project_root: Path, device: str) -> tuple[AgentTaskRunner, str]:
    """Initialize each real dependency once and return a model-known user."""

    provider = OpenAICompatiblePlannerProvider(model=MODEL)
    goal_extractor = StructuredGoalExtractor(
        provider=provider,
        timeout_seconds=30,
        schema_retries=1,
    )
    planner = StructuredLLMPlanner(
        provider=provider,
        timeout_seconds=30,
        schema_retries=1,
    )

    recommendation_service = RecommendationService(
        bundle_dir=project_root / "artifacts/recommendation/serving_v2",
        catalog_path=(
            project_root
            / "data/processed/recommendation/product_catalog.jsonl"
        ),
        device=device,
    )
    recommendation_tool = RecommendationToolAdapter(recommendation_service)

    # Strict validation is intentional for this formal real-artifact smoke.
    retrieval_artifacts = ChunkRetrievalArtifacts(
        retrieval_dir=project_root / "artifacts/recommendation/retrieval",
        knowledge_dir=project_root / "artifacts/recommendation/knowledge",
        validation_mode=ValidationMode.STRICT,
        expected_dimension=1024,
    )
    query_encoder = BGEM3QueryEncoder(
        model_path=project_root / "models/bge-m3",
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
    user_id = model_user_ids[0]

    runner = AgentTaskRunner(
        # Required by the legacy single-requirement entry point; run_goal()
        # uses goal_extractor instead.
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
        goal_extractor=goal_extractor,
    )
    return runner, user_id


def main() -> int:
    if not os.getenv("DEEPSEEK_API_KEY", "").strip():
        print(
            "ERROR: DEEPSEEK_API_KEY is required in the process environment.",
            file=sys.stderr,
        )
        return 2

    args = _parse_args()
    project_root = args.project_root.resolve()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    runner, user_id = _build_runner(
        project_root=project_root,
        device=args.device,
    )
    result = runner.run_goal(
        user_id=user_id,
        session_id="v2-10-5-real-goal-e2e-session",
        plan_id="v2-10-5-real-goal-e2e-plan",
        user_request=REQUEST,
        currency="USD",
        recursion_limit=50,
    )

    if result.workflow_state is None:
        raise AssertionError(
            f"Goal execution produced no workflow state: status={result.status.value}."
        )
    state = result.workflow_state
    plan = state.agent_state.shopping_plan
    response_grounded = (
        result.final_response is not None and result.response_error is None
    )

    if result.status is not GoalExecutionStatus.READY:
        raise AssertionError(f"Expected goal status READY, got {result.status.value}.")
    if state.route is not WorkflowRoute.READY:
        raise AssertionError(f"Expected workflow route READY, got {state.route!r}.")
    if len(plan.requirements) != 3:
        raise AssertionError("Expected exactly three goal requirements.")
    if len(plan.selected_items) != 3:
        raise AssertionError("Expected exactly three selected items.")
    if plan.total_budget != 500.0:
        raise AssertionError(f"Expected total budget 500.00, got {plan.total_budget}.")
    if not response_grounded:
        raise AssertionError("Expected one grounded final response.")

    print("=== Real Goal E2E ===")
    print(f"goal_status={result.status.value}")
    print(f"requirements_count={len(plan.requirements)}")
    print(f"selected_items={len(plan.selected_items)}")
    print(f"total_budget={plan.total_budget:.2f}")
    print(f"spent={plan.total_spent:.2f}")
    print(f"remaining={plan.remaining_budget:.2f}")
    print(f"workflow_route={state.route.value}")
    print(f"response_grounded={response_grounded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
