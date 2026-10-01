"""FastAPI application backed by one process-scoped AgentRec runtime."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..integration import GoalExecutionResult, GoalExecutionStatus
from ..response import ConflictDecisionSummary, ProductDecisionSummary
from .dependencies import AgentRuntime, build_runtime
from .schemas import (
    ChatRequest,
    ChatResponse,
    ClarificationResponse,
    ConflictResponse,
    PlanResponse,
    ProductResponse,
    RequirementResponse,
    VerifiedRequirementResponse,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WEB_DIR = PROJECT_ROOT / "web"
DEVICE = "cuda:0"


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Load heavy model and artifact dependencies once per API process."""

    application.state.agent_runtime = build_runtime(
        project_root=PROJECT_ROOT,
        device=DEVICE,
    )
    yield


app = FastAPI(
    title="AgentRec API",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def get_runtime(request: Request) -> AgentRuntime:
    """Return the process-scoped runtime initialized by the lifespan hook."""

    runtime = getattr(request.app.state, "agent_runtime", None)
    if not isinstance(runtime, AgentRuntime):
        raise RuntimeError("AgentRuntime has not been initialized.")
    return runtime


@app.get("/", response_class=FileResponse)
def web_demo() -> FileResponse:
    """Serve the existing same-origin AgentRec Web Demo."""

    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    """Return a lightweight process health response."""

    return {
        "status": "ok",
        "service": "AgentRec API",
    }


def _project_chat_response(result: GoalExecutionResult) -> ChatResponse:
    """Project one Goal result without exposing runtime implementation state."""

    response_text = (
        None if result.final_response is None else result.final_response.text
    )
    if result.status is GoalExecutionStatus.CLARIFICATION_REQUIRED:
        clarification = ClarificationResponse(
            needed=True,
            question=result.clarification_question,
        )
        return ChatResponse(
            status=result.status.value,
            response=response_text,
            clarification=clarification,
            plan=None,
            requirements=[],
            products=[],
            conflict=None,
        )

    # ERROR is a valid terminal result, but its workflow internals are not a
    # Web-safe business response. Return only the sanitized public status.
    if result.status is GoalExecutionStatus.ERROR:
        return ChatResponse(
            status=result.status.value,
            response=None,
            clarification=None,
            plan=None,
            requirements=[],
            products=[],
            conflict=None,
        )

    workflow_state = result.workflow_state
    if workflow_state is None:
        return ChatResponse(
            status=result.status.value,
            response=response_text,
            clarification=None,
            plan=None,
            requirements=[],
            products=[],
            conflict=None,
        )

    shopping_plan = workflow_state.agent_state.shopping_plan
    allocation = workflow_state.goal_budget_allocation
    allocated_by_requirement = (
        {}
        if allocation is None
        else {
            item.requirement_id: item.allocated_budget
            for item in allocation.allocations
        }
    )
    requirements = [
        RequirementResponse(
            requirement_id=requirement.requirement_id,
            category=requirement.category,
            quantity=requirement.quantity,
            max_budget=requirement.max_budget,
            allocated_budget=allocated_by_requirement.get(
                requirement.requirement_id
            ),
            required_features=list(requirement.required_features),
            soft_preferences=list(requirement.soft_preferences),
            status=requirement.status.value,
        )
        for requirement in shopping_plan.requirements
    ]

    summaries_by_product: dict[tuple[str, str], ProductDecisionSummary] = {}
    if result.final_response is not None and isinstance(
        result.final_response.decision_summary,
        tuple,
    ):
        summaries_by_product = {
            (summary.requirement_id, summary.parent_asin): summary
            for summary in result.final_response.decision_summary
            if isinstance(summary, ProductDecisionSummary)
        }

    products = []
    for item in shopping_plan.selected_items:
        summary = summaries_by_product.get((item.requirement_id, item.parent_asin))
        verified_requirements = (
            []
            if summary is None
            else [
                VerifiedRequirementResponse(
                    constraint=claim.original_constraint,
                    status="supported",
                )
                for claim in summary.verified_claims
            ]
        )
        products.append(
            ProductResponse(
                requirement_id=item.requirement_id,
                category=item.category,
                title=item.title,
                parent_asin=item.parent_asin,
                price=item.price,
                quantity=item.quantity,
                subtotal=item.price * item.quantity,
                source=item.source.value,
                verified_requirements=verified_requirements,
            )
        )

    conflict = None
    if result.final_response is not None and isinstance(
        result.final_response.decision_summary,
        ConflictDecisionSummary,
    ):
        decision = result.final_response.decision_summary
        conflict = ConflictResponse(
            reason=decision.reason.value,
            category=decision.category,
            required_features=list(decision.required_features),
            failed_constraints=list(decision.failed_constraints),
            unknown_constraints=list(decision.unknown_constraints),
            contradicted_constraints=list(decision.contradicted_constraints),
            replan_attempts_performed=decision.replan_attempts_performed,
            user_action_required=decision.user_action_required,
        )

    return ChatResponse(
        status=result.status.value,
        response=response_text,
        clarification=None,
        plan=PlanResponse(
            status=shopping_plan.status.value,
            currency=shopping_plan.currency,
            total_budget=shopping_plan.total_budget,
            total_spent=shopping_plan.total_spent,
            remaining_budget=shopping_plan.remaining_budget,
        ),
        requirements=requirements,
        products=products,
        conflict=conflict,
    )


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    runtime: AgentRuntime = Depends(get_runtime),
) -> ChatResponse:
    """Execute one shopping goal through the shared Agent runtime."""

    effective_user_id = request.user_id or runtime.default_user_id
    result = runtime.runner.run_goal(
        user_id=effective_user_id,
        session_id=request.session_id,
        plan_id=f"api-{uuid4()}",
        user_request=request.query,
    )
    return _project_chat_response(result)
