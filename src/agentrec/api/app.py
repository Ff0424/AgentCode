"""FastAPI application backed by one process-scoped AgentRec runtime."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, Request

from .dependencies import AgentRuntime, build_runtime
from .schemas import ChatRequest, ChatResponse, ProductResponse


PROJECT_ROOT = Path(__file__).resolve().parents[3]
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


def get_runtime(request: Request) -> AgentRuntime:
    """Return the process-scoped runtime initialized by the lifespan hook."""

    runtime = getattr(request.app.state, "agent_runtime", None)
    if not isinstance(runtime, AgentRuntime):
        raise RuntimeError("AgentRuntime has not been initialized.")
    return runtime


@app.get("/health")
def health() -> dict[str, str]:
    """Return a lightweight process health response."""

    return {
        "status": "ok",
        "service": "AgentRec API",
    }


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    runtime: AgentRuntime = Depends(get_runtime),
) -> ChatResponse:
    """Execute one shopping goal through the shared Agent runtime."""

    result = runtime.runner.run_goal(
        user_id=request.user_id,
        session_id=request.session_id,
        plan_id=f"api-{uuid4()}",
        user_request=request.query,
    )

    workflow_state = result.workflow_state
    if workflow_state is None:
        products: list[ProductResponse] = []
        remaining_budget = None
    else:
        plan = workflow_state.agent_state.shopping_plan
        products = [
            ProductResponse(
                category=item.category,
                title=item.title,
                parent_asin=item.parent_asin,
                price=item.price,
            )
            for item in plan.selected_items
        ]
        remaining_budget = plan.remaining_budget

    return ChatResponse(
        status=result.status.value,
        response=(
            None if result.final_response is None else result.final_response.text
        ),
        products=products,
        workflow_status=result.status.value,
        remaining_budget=remaining_budget,
    )
