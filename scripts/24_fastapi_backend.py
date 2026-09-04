"""
FastAPI backend for AgentRec.

This module exposes the completed Agent pipeline through a thin HTTP API.

Responsibilities:
- initialize the Stage 23 LLM adapter and Stage 22 Agent once
- expose health and chat endpoints
- validate HTTP request/response payloads
- avoid leaking full internal tool history to frontend clients

It does not implement retrieval, ranking, RAG construction, tool logic,
or LLM-provider protocol conversion.
"""

import importlib.util
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAIError
from pydantic import BaseModel, Field


# ============================================================
# 1. Project paths and Stage 23 dynamic loading
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / "web"
WEB_INDEX_PATH = WEB_DIR / "index.html"

LLM_ADAPTER_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "23_llm_adapter.py"
)


def _load_llm_adapter_module():
    """
    Load Stage 23 without modifying sys.path.

    Stage 23 already knows how to load Stage 22, and Stage 22 in turn
    owns AgentTools. Stage 24 therefore depends only on Stage 23.
    """

    if not LLM_ADAPTER_PATH.is_file():
        raise FileNotFoundError(
            f"LLM adapter module not found: {LLM_ADAPTER_PATH}"
        )

    module_name = "agentcode_llm_adapter_23_for_api"

    spec = importlib.util.spec_from_file_location(
        module_name,
        LLM_ADAPTER_PATH,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not create an import spec for {LLM_ADAPTER_PATH}."
        )

    sys.dont_write_bytecode = True

    module = importlib.util.module_from_spec(spec)

    # Register the module so dynamically loaded classes remain stable.
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)

        raise ImportError(
            f"Could not load {LLM_ADAPTER_PATH}: {exc}"
        ) from exc

    return module


_llm_adapter_module = _load_llm_adapter_module()

OpenAICompatibleLLMClient = getattr(
    _llm_adapter_module,
    "OpenAICompatibleLLMClient",
    None,
)

Agent = getattr(
    _llm_adapter_module,
    "Agent",
    None,
)

DEFAULT_BASE_URL = getattr(
    _llm_adapter_module,
    "DEFAULT_BASE_URL",
    None,
)

DEFAULT_MODEL = getattr(
    _llm_adapter_module,
    "DEFAULT_MODEL",
    None,
)

DEFAULT_MAX_STEPS = getattr(
    _llm_adapter_module,
    "DEFAULT_MAX_STEPS",
    6,
)


if not isinstance(OpenAICompatibleLLMClient, type):
    raise ImportError(
        "OpenAICompatibleLLMClient was not found in Stage 23."
    )

if not isinstance(Agent, type):
    raise ImportError(
        "Agent was not found in Stage 23."
    )

if not isinstance(DEFAULT_BASE_URL, str):
    raise ImportError(
        "DEFAULT_BASE_URL was not found in Stage 23."
    )

if not isinstance(DEFAULT_MODEL, str):
    raise ImportError(
        "DEFAULT_MODEL was not found in Stage 23."
    )


# ============================================================
# 2. HTTP request / response schemas
# ============================================================

class ChatRequest(BaseModel):
    """
    User-facing chat request.

    Stage 24 is intentionally stateless for now:
    one request contains one fresh user query.
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="User shopping request.",
    )


class ToolCallSummary(BaseModel):
    """
    Compact frontend-safe summary of one Agent tool execution.

    Full tool results remain server-side and are not returned through
    this API.
    """

    step: int
    tool_name: str
    ok: bool


class ChatResponse(BaseModel):
    """
    Compact response returned to the frontend.
    """

    answer: str
    steps: int
    tool_call_count: int
    tool_calls: list[ToolCallSummary]


class HealthResponse(BaseModel):
    status: str
    agent_ready: bool
    model: str


# ============================================================
# 3. Long-lived Agent runtime
# ============================================================

class AgentRuntime:
    """
    Own one reusable Agent instance for the lifetime of the API process.

    The recommendation stack behind AgentTools includes expensive
    resources such as BGE-M3, FAISS, and the product repository.
    These resources must not be recreated for every HTTP request.
    """

    def __init__(self):
        # Stage 23 reads DEEPSEEK_API_KEY from the environment.
        self.llm_client = OpenAICompatibleLLMClient(
            base_url=DEFAULT_BASE_URL,
            model=DEFAULT_MODEL,
        )

        # Agent initialization cascades into the existing tool/service
        # stack. This happens once during application startup.
        self.agent = Agent(
            llm_client=self.llm_client,
            max_steps=DEFAULT_MAX_STEPS,
        )

        # The current Agent/retrieval stack is synchronous and shares
        # GPU/model state. Serialize Agent runs in this first backend
        # version rather than allowing unsafe concurrent inference.
        self._run_lock = threading.Lock()

    def run(self, message: str) -> dict:
        """
        Execute one complete stateless Agent request.
        """

        if not isinstance(message, str):
            raise TypeError("message must be a string.")

        message = message.strip()

        if not message:
            raise ValueError(
                "message must be a non-empty string."
            )

        with self._run_lock:
            return self.agent.run(message)


# ============================================================
# 4. FastAPI application lifecycle
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize heavyweight Agent resources once at process startup.

    If initialization fails, application startup fails immediately
    rather than accepting requests with a broken runtime.
    """

    runtime = AgentRuntime()

    app.state.agent_runtime = runtime

    yield

    # No explicit teardown is currently required.
    app.state.agent_runtime = None


app = FastAPI(
    title="AgentRec API",
    version="1.0.0",
    description=(
        "HTTP backend for the AgentRec shopping recommendation Agent."
    ),
    lifespan=lifespan,
)

# Static assets remain separate from the existing API routes while sharing the
# same origin, so the browser can call /api/chat without CORS configuration.
app.mount(
    "/static",
    StaticFiles(directory=WEB_DIR),
    name="static",
)


# ============================================================
# 5. Helper functions
# ============================================================

def _get_runtime(request: Request) -> AgentRuntime:
    """
    Return the initialized application-level Agent runtime.
    """

    runtime = getattr(
        request.app.state,
        "agent_runtime",
        None,
    )

    if not isinstance(runtime, AgentRuntime):
        raise HTTPException(
            status_code=503,
            detail="Agent runtime is not ready.",
        )

    return runtime


def _build_chat_response(result: dict) -> ChatResponse:
    """
    Convert the complete internal Agent result into a compact,
    frontend-safe response.

    Stage 22 intentionally keeps complete tool history for debugging.
    Stage 24 must not expose those raw recommendation objects, product
    text, retrieval scores, or other internal fields by default.
    """

    if not isinstance(result, dict):
        raise TypeError(
            "Agent result must be a dictionary."
        )

    answer = result.get("answer")
    steps = result.get("steps")
    history = result.get("tool_calls", [])

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(
            "Agent result contained no valid answer."
        )

    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError(
            "Agent result['steps'] must be an integer."
        )

    if not isinstance(history, list):
        raise TypeError(
            "Agent result['tool_calls'] must be a list."
        )

    summaries = []

    for position, item in enumerate(history):
        if not isinstance(item, dict):
            raise TypeError(
                "Agent tool history item at position "
                f"{position} must be a dictionary."
            )

        step = item.get("step")
        tool_name = item.get("tool_name")
        ok = item.get("ok")

        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError(
                f"tool_calls[{position}].step must be an integer."
            )

        if (
            not isinstance(tool_name, str)
            or not tool_name.strip()
        ):
            raise TypeError(
                f"tool_calls[{position}].tool_name "
                "must be a non-empty string."
            )

        if not isinstance(ok, bool):
            raise TypeError(
                f"tool_calls[{position}].ok must be a boolean."
            )

        summaries.append(
            ToolCallSummary(
                step=step,
                tool_name=tool_name,
                ok=ok,
            )
        )

    return ChatResponse(
        answer=answer,
        steps=steps,
        tool_call_count=len(summaries),
        tool_calls=summaries,
    )


# ============================================================
# 6. Web demo and health endpoint
# ============================================================

@app.get(
    "/",
    response_class=FileResponse,
    include_in_schema=False,
)
def web_demo() -> FileResponse:
    """Serve the Stage 25 browser demo from the FastAPI origin."""

    return FileResponse(WEB_INDEX_PATH)


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health(request: Request) -> HealthResponse:
    """
    Lightweight process/readiness check.

    This endpoint does not run BGE-M3 or call the external LLM API.
    """

    runtime = getattr(
        request.app.state,
        "agent_runtime",
        None,
    )

    return HealthResponse(
        status="ok",
        agent_ready=isinstance(
            runtime,
            AgentRuntime,
        ),
        model=DEFAULT_MODEL,
    )


# ============================================================
# 7. Main Agent chat endpoint
# ============================================================

@app.post(
    "/api/chat",
    response_model=ChatResponse,
)
def chat(
    payload: ChatRequest,
    request: Request,
) -> ChatResponse:
    """
    Run one fresh AgentRec shopping request.

    This API is intentionally single-turn/stateless in Stage 24.
    """

    runtime = _get_runtime(request)

    try:
        result = runtime.run(payload.message)

        return _build_chat_response(result)

    # --------------------------------------------------------
    # Provider / upstream failures
    # --------------------------------------------------------
    except OpenAIError as exc:
        raise HTTPException(
            status_code=502,
            detail="LLM provider request failed.",
        ) from exc

    # --------------------------------------------------------
    # User/tool validation failures
    # --------------------------------------------------------
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    # --------------------------------------------------------
    # Runtime/service availability failures
    # --------------------------------------------------------
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
