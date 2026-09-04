"""
Agent orchestrates an LLM-driven tool-calling loop over AgentTools.

It does not implement retrieval, ranking, product storage, or LLM-provider
specific networking.
"""

import copy
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


# ============================================================
# 1. Dynamically loaded Stage 21 tool layer
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_TOOLS_PATH = PROJECT_ROOT / "scripts" / "21_agent_tools.py"


def _load_agent_tools_module():
    """Load Stage 21 without modifying sys.path or creating bytecode cache."""

    if not AGENT_TOOLS_PATH.is_file():
        raise FileNotFoundError(f"Agent tools module not found: {AGENT_TOOLS_PATH}")
    spec = importlib.util.spec_from_file_location(
        "agentcode_agent_tools_21",
        AGENT_TOOLS_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {AGENT_TOOLS_PATH}.")

    sys.dont_write_bytecode = True
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ImportError(f"Could not load {AGENT_TOOLS_PATH}: {exc}") from exc
    return module


_agent_tools_module = _load_agent_tools_module()
AgentTools = getattr(_agent_tools_module, "AgentTools", None)
TOOL_SCHEMAS = getattr(_agent_tools_module, "TOOL_SCHEMAS", None)
if not isinstance(AgentTools, type):
    raise ImportError(f"AgentTools was not found in {AGENT_TOOLS_PATH}.")
if not isinstance(TOOL_SCHEMAS, list):
    raise ImportError(f"TOOL_SCHEMAS was not found in {AGENT_TOOLS_PATH}.")


# ============================================================
# 2. Provider-neutral LLM contracts
# ============================================================

@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall]


class LLMClientProtocol(Protocol):
    def chat(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> LLMResponse:
        ...


SYSTEM_PROMPT = """You are AgentRec, a shopping recommendation assistant.

Use available tools when product search, product lookup, or product comparison
requires factual catalog information. Do not invent product facts. Treat tool
results as the factual source for product information. Do not expose internal
retrieval, semantic, popularity, or ranking scores unless explicitly required
for debugging. If a request requires unavailable information, say that it is
unavailable instead of fabricating it. When enough information is available,
answer the user directly."""


# ============================================================
# 3. Agent orchestration and deterministic tool loop
# ============================================================

class Agent:
    """Run a bounded provider-neutral LLM and AgentTools conversation loop."""

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        tools=None,
        max_steps: int = 6,
    ):
        if not callable(getattr(llm_client, "chat", None)):
            raise TypeError("llm_client must provide a callable chat() method.")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int):
            raise TypeError("max_steps must be a positive integer; bool is invalid.")
        if max_steps <= 0:
            raise ValueError("max_steps must be greater than zero.")

        if tools is None:
            tools = AgentTools()
        if not callable(getattr(tools, "call_tool", None)):
            raise TypeError("tools must provide a callable call_tool() method.")

        self.llm_client = llm_client
        self.tools = tools
        self.max_steps = int(max_steps)

    @staticmethod
    def _validate_user_message(user_message: str) -> None:
        if not isinstance(user_message, str):
            raise TypeError(
                "user_message must be a string, found "
                f"{type(user_message).__name__}."
            )
        if not user_message.strip():
            raise ValueError("user_message must be a non-empty string.")

    @staticmethod
    def _validate_response(response) -> None:
        if not isinstance(response, LLMResponse):
            raise TypeError(
                "llm_client.chat() must return an LLMResponse instance."
            )
        if response.content is not None and not isinstance(response.content, str):
            raise TypeError("LLMResponse.content must be a string or None.")
        if not isinstance(response.tool_calls, list):
            raise TypeError("LLMResponse.tool_calls must be a list.")

    @staticmethod
    def _validate_tool_calls(tool_calls: list[ToolCall]) -> None:
        seen_ids = set()
        for position, call in enumerate(tool_calls):
            if not isinstance(call, ToolCall):
                raise TypeError(
                    f"tool_calls[{position}] must be a ToolCall instance."
                )
            if not isinstance(call.id, str) or not call.id.strip():
                raise ValueError(f"tool_calls[{position}].id must be a non-empty string.")
            if call.id in seen_ids:
                raise ValueError(f"Duplicate tool call id in one response: {call.id!r}.")
            if not isinstance(call.name, str) or not call.name.strip():
                raise ValueError(
                    f"tool_calls[{position}].name must be a non-empty string."
                )
            if not isinstance(call.arguments, dict):
                raise TypeError(f"tool_calls[{position}].arguments must be a dictionary.")
            seen_ids.add(call.id)


    @staticmethod
    def _build_llm_tool_payload(tool_name: str, result: dict) -> dict:
        """
        Build the compact tool result exposed to the LLM.

        The full raw tool result is still preserved in tool_history for
        debugging, logging, and downstream backend use. This projection only
        controls what is appended to the LLM conversation context.
        """

        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("tool_name must be a non-empty string.")
        if not isinstance(result, dict):
            raise TypeError("Tool result must be a dictionary.")

        # --------------------------------------------------------
        # recommend_products
        #
        # Stage 21 already builds a deterministic RAG context.
        # Do not resend the full recommendation objects, raw product
        # text, or internal ranking scores to the LLM.
        # --------------------------------------------------------
        if tool_name == "recommend_products":
            return {
                "tool": result.get("tool"),
                "query": result.get("query"),
                "rag_context": result.get("rag_context"),
                "included_product_ids": copy.deepcopy(
                    result.get("included_product_ids", [])
                ),
                "recommendation_count": result.get(
                    "recommendation_count"
                ),
                "context_truncated": result.get(
                    "context_truncated"
                ),
            }

        # --------------------------------------------------------
        # get_product_details
        #
        # Repository products contain the original combined `text`
        # field. The structured fields already contain the factual
        # information needed by the LLM, so remove only `text`.
        # --------------------------------------------------------
        if tool_name == "get_product_details":
            products = result.get("products", [])
            if not isinstance(products, list):
                raise TypeError(
                    "get_product_details result['products'] must be a list."
                )

            projected_products = []

            for position, product in enumerate(products):
                if not isinstance(product, dict):
                    raise TypeError(
                        "get_product_details product at position "
                        f"{position} must be a dictionary."
                    )

                projected_product = {
                    key: copy.deepcopy(value)
                    for key, value in product.items()
                    if key != "text"
                }
                projected_products.append(projected_product)

            return {
                "tool": result.get("tool"),
                "requested_product_ids": copy.deepcopy(
                    result.get("requested_product_ids", [])
                ),
                "products": projected_products,
                "count": result.get("count"),
            }

        # --------------------------------------------------------
        # compare_products
        #
        # Stage 21 already exposes comparison-safe structured fields
        # and intentionally excludes the raw product `text`.
        # --------------------------------------------------------
        if tool_name == "compare_products":
            return copy.deepcopy(result)

        # Defensive guard: Stage 22 should never silently expose the
        # full payload of an unknown future tool to the LLM.
        raise ValueError(
            f"No LLM payload projection is defined for tool {tool_name!r}."
        )


    def run(self, user_message: str) -> dict:
        """Run a fresh bounded conversation until final content is produced."""

        self._validate_user_message(user_message)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        tool_history = []

        for step in range(1, self.max_steps + 1):
            response = self.llm_client.chat(
                messages=messages,
                tools=TOOL_SCHEMAS,
            )
            self._validate_response(response)
            self._validate_tool_calls(response.tool_calls)

            if not response.tool_calls:
                if response.content is None or not response.content.strip():
                    raise ValueError(
                        "LLMResponse without tool calls must contain a non-empty answer."
                    )
                return {
                    "answer": response.content,
                    "steps": int(step),
                    "tool_calls": tool_history,
                }

            # Intermediate content is retained but is never treated as final when
            # the same response contains one or more structured tool calls.
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "name": call.name,
                            "arguments": copy.deepcopy(call.arguments),
                        }
                        for call in response.tool_calls
                    ],
                }
            )

            for call in response.tool_calls:
                call_arguments = copy.deepcopy(call.arguments)
                try:
                    result = self.tools.call_tool(call.name, call.arguments)
                except (KeyError, TypeError, ValueError) as exc:
                    tool_payload = {
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    history_entry = {
                        "step": int(step),
                        "tool_call_id": call.id,
                        "tool_name": call.name,
                        "arguments": call_arguments,
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                else:
                    # Keep the complete tool result for server-side history,
                    # but expose only the compact projected result to the LLM.
                    llm_result = self._build_llm_tool_payload(
                        call.name,
                        result,
                    )
                    tool_payload = {"ok": True, "result": llm_result}
                    history_entry = {
                        "step": int(step),
                        "tool_call_id": call.id,
                        "tool_name": call.name,
                        "arguments": call_arguments,
                        "ok": True,
                        "result": copy.deepcopy(result),
                    }

                tool_history.append(history_entry)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": json.dumps(
                            tool_payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            allow_nan=False,
                        ),
                    }
                )

        raise RuntimeError(
            "Agent reached max_steps without producing a final answer."
        )


# ============================================================
# 4. Scripted mock client for offline loop testing
# ============================================================

class ScriptedLLMClient:
    """Return predefined responses while recording deep-copied chat inputs."""

    def __init__(self, responses: list[LLMResponse]):
        if not isinstance(responses, list):
            raise TypeError("responses must be a list of LLMResponse objects.")
        for position, response in enumerate(responses):
            if not isinstance(response, LLMResponse):
                raise TypeError(f"responses[{position}] must be an LLMResponse.")
        self.responses = list(responses)
        self._next_response = 0
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        self.calls.append(
            {
                "messages": copy.deepcopy(messages),
                "tools": copy.deepcopy(tools),
            }
        )
        if self._next_response >= len(self.responses):
            raise RuntimeError("ScriptedLLMClient responses are exhausted.")
        response = self.responses[self._next_response]
        self._next_response += 1
        return response


# ============================================================
# 5. Mock two-step CLI demonstration
# ============================================================

def main() -> int:
    scripted_client = ScriptedLLMClient(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="recommend_products",
                        arguments={
                            "query": "USB-C hub with HDMI and ethernet",
                            "top_k": 2,
                            "candidate_k": 50,
                            "rerank_pool_k": 20,
                        },
                    )
                ],
            ),
            LLMResponse(
                content=(
                    "I found two relevant USB-C hubs. "
                    "The first is the stronger match for HDMI and Ethernet."
                ),
                tool_calls=[],
            ),
        ]
    )
    result = Agent(llm_client=scripted_client).run(
        "Recommend a USB-C hub with HDMI and ethernet."
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileNotFoundError,
        ImportError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
