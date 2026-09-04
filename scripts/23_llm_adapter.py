"""
OpenAI-compatible Chat Completions adapter for the provider-neutral Agent.

This module translates between Stage 22 LLM contracts and a synchronous
OpenAI-compatible provider client. It does not orchestrate the Agent, execute
tools, implement retrieval, or persist conversations.
"""

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

from openai import OpenAI, OpenAIError


# ============================================================
# 1. Stage 22 contracts and provider defaults
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_PATH = PROJECT_ROOT / "scripts" / "22_agent.py"

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEFAULT_MAX_STEPS = 6


def _load_agent_module():
    """Load the numeric-prefixed Stage 22 module without changing sys.path."""

    if not AGENT_PATH.is_file():
        raise FileNotFoundError(f"Agent module not found: {AGENT_PATH}")
    module_name = "agentcode_agent_22_for_llm_adapter"
    spec = importlib.util.spec_from_file_location(module_name, AGENT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {AGENT_PATH}.")

    sys.dont_write_bytecode = True
    module = importlib.util.module_from_spec(spec)
    # Registration keeps dataclass/module introspection stable after loading.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise ImportError(f"Could not load {AGENT_PATH}: {exc}") from exc
    return module


_agent_module = _load_agent_module()
ToolCall = getattr(_agent_module, "ToolCall", None)
LLMResponse = getattr(_agent_module, "LLMResponse", None)
Agent = getattr(_agent_module, "Agent", None)
if not all(isinstance(item, type) for item in (ToolCall, LLMResponse, Agent)):
    raise ImportError(f"ToolCall, LLMResponse, or Agent is missing from {AGENT_PATH}.")


# ============================================================
# 2. OpenAI-compatible synchronous client adapter
# ============================================================

class OpenAICompatibleLLMClient:
    """Convert Chat Completions responses into Stage 22 LLMResponse objects."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.2,
    ):
        if api_key is None:
            api_key = os.getenv(DEFAULT_API_KEY_ENV)
            if api_key is None or not api_key.strip():
                raise RuntimeError(
                    "Missing API key. Set DEEPSEEK_API_KEY or pass api_key explicitly."
                )
        if not isinstance(api_key, str):
            raise TypeError("api_key must be a non-empty string.")
        if not api_key.strip():
            raise ValueError("api_key must be a non-empty string.")
        if not isinstance(base_url, str):
            raise TypeError("base_url must be a non-empty string.")
        if not base_url.strip():
            raise ValueError("base_url must be a non-empty string.")
        if not isinstance(model, str):
            raise TypeError("model must be a non-empty string.")
        if not model.strip():
            raise ValueError("model must be a non-empty string.")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise TypeError("temperature must be a finite int or float; bool is invalid.")
        temperature = float(temperature)
        if not math.isfinite(temperature):
            raise ValueError("temperature must be finite.")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be in the inclusive range [0, 2].")

        # The key is passed directly to the SDK and is never logged or retained
        # separately by this adapter.
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    @staticmethod
    def _validate_inputs(messages: list[dict], tools: list[dict]) -> None:
        if not isinstance(messages, list):
            raise TypeError("messages must be a list of dictionaries.")
        if not messages:
            raise ValueError("messages must not be empty.")
        for position, message in enumerate(messages):
            if not isinstance(message, dict):
                raise TypeError(f"messages[{position}] must be a dictionary.")

        if not isinstance(tools, list):
            raise TypeError("tools must be a list of dictionaries.")
        for position, tool in enumerate(tools):
            if not isinstance(tool, dict):
                raise TypeError(f"tools[{position}] must be a dictionary.")

    @staticmethod
    def _to_provider_tools(tools: list[dict]) -> list[dict]:
        """Wrap Stage 21 provider-neutral schemas as function tools."""

        provider_tools = []
        for position, tool in enumerate(tools):
            if tool.get("type") == "function" and isinstance(
                tool.get("function"), dict
            ):
                provider_tools.append(tool)
                continue
            if not isinstance(tool.get("name"), str) or not tool["name"].strip():
                raise ValueError(f"tools[{position}].name must be a non-empty string.")
            if not isinstance(tool.get("parameters"), dict):
                raise ValueError(f"tools[{position}].parameters must be a dictionary.")
            provider_tools.append({"type": "function", "function": tool})
        return provider_tools

    @staticmethod
    def _to_provider_messages(messages: list[dict]) -> list[dict]:
        """Convert Stage 22 assistant tool calls to Chat Completions format."""

        provider_messages = []
        for message_position, message in enumerate(messages):
            role = message.get("role")
            if not isinstance(role, str) or not role.strip():
                raise ValueError(
                    f"messages[{message_position}].role must be a non-empty string."
                )

            if role == "assistant" and "tool_calls" in message:
                internal_calls = message["tool_calls"]
                if not isinstance(internal_calls, list):
                    raise TypeError(
                        f"messages[{message_position}].tool_calls must be a list."
                    )

                provider_calls = []
                for call_position, call in enumerate(internal_calls):
                    context = (
                        f"messages[{message_position}].tool_calls[{call_position}]"
                    )
                    if not isinstance(call, dict):
                        raise TypeError(f"{context} must be a dictionary.")
                    call_id = call.get("id")
                    if not isinstance(call_id, str) or not call_id.strip():
                        raise ValueError(f"{context}.id must be a non-empty string.")
                    tool_name = call.get("name")
                    if not isinstance(tool_name, str) or not tool_name.strip():
                        raise ValueError(f"{context}.name must be a non-empty string.")
                    arguments = call.get("arguments")
                    if not isinstance(arguments, dict):
                        raise TypeError(f"{context}.arguments must be a dictionary.")

                    provider_calls.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(
                                    arguments,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                    allow_nan=False,
                                ),
                            },
                        }
                    )
                provider_messages.append(
                    {
                        "role": role,
                        "content": message.get("content"),
                        "tool_calls": provider_calls,
                    }
                )
                continue

            if role == "tool":
                provider_message = {
                    "role": role,
                    "content": message.get("content"),
                    "tool_call_id": message.get("tool_call_id"),
                }
                if "name" in message:
                    provider_message["name"] = message["name"]
                provider_messages.append(provider_message)
                continue

            # Ordinary system/user/assistant messages need only role and content.
            provider_messages.append(
                {"role": role, "content": message.get("content")}
            )

        return provider_messages

    @staticmethod
    def _parse_tool_calls(provider_tool_calls) -> list:
        if provider_tool_calls is None:
            return []
        if not isinstance(provider_tool_calls, list):
            raise TypeError("LLM message.tool_calls must be a list or None.")

        parsed = []
        for position, call in enumerate(provider_tool_calls):
            call_id = getattr(call, "id", None)
            if not isinstance(call_id, str) or not call_id.strip():
                raise ValueError(
                    f"Provider tool_calls[{position}].id must be a non-empty string."
                )
            function = getattr(call, "function", None)
            if function is None:
                raise ValueError(
                    f"Provider tool_calls[{position}] contained no function."
                )
            tool_name = getattr(function, "name", None)
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ValueError(
                    f"Provider tool_calls[{position}].function.name must be non-empty."
                )
            raw_arguments = getattr(function, "arguments", None)
            if not isinstance(raw_arguments, str):
                raise TypeError(
                    f"Tool arguments for {tool_name!r} must be a JSON string."
                )
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON tool arguments for {tool_name!r}: "
                    f"{exc.msg} at character {exc.pos}."
                ) from exc
            if not isinstance(arguments, dict):
                raise ValueError(
                    f"Tool arguments for {tool_name!r} must decode to an object."
                )
            parsed.append(
                ToolCall(id=call_id, name=tool_name, arguments=arguments)
            )
        return parsed

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        """Call Chat Completions and return the provider-neutral response."""

        self._validate_inputs(messages, tools)
        provider_messages = self._to_provider_messages(messages)
        provider_tools = self._to_provider_tools(tools)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=provider_messages,
            tools=provider_tools,
            tool_choice="auto",
            temperature=self.temperature,
            extra_body={
                "thinking": {
                    "type": "disabled"
                }
            },
        )

        choices = getattr(response, "choices", None)
        if not choices:
            raise RuntimeError("LLM response contained no choices.")
        message = getattr(choices[0], "message", None)
        if message is None:
            raise RuntimeError("LLM response contained no message.")

        content = getattr(message, "content", None)
        if content is not None and not isinstance(content, str):
            raise TypeError("LLM message.content must be a string or None.")
        parsed_tool_calls = self._parse_tool_calls(
            getattr(message, "tool_calls", None)
        )
        return LLMResponse(content=content, tool_calls=parsed_tool_calls)


# ============================================================
# 3. Minimal real-Agent CLI
# ============================================================

def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run AgentRec through an OpenAI-compatible LLM provider."
    )
    parser.add_argument("--query", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--max-steps",
        type=_positive_integer,
        default=DEFAULT_MAX_STEPS,
    )
    return parser.parse_args()


def _compact_cli_result(result: dict) -> dict:
    history = result.get("tool_calls", [])
    return {
        "answer": result.get("answer"),
        "steps": result.get("steps"),
        "tool_call_count": len(history),
        "tool_calls": [
            {
                "step": item.get("step"),
                "tool_name": item.get("tool_name"),
                "ok": item.get("ok"),
            }
            for item in history
        ],
    }


def main() -> int:
    args = parse_args()
    llm_client = OpenAICompatibleLLMClient(
        base_url=args.base_url,
        model=args.model,
    )
    agent = Agent(llm_client=llm_client, max_steps=args.max_steps)

    # Tool-result projection/context minimization is a future optimization;
    # Stage 22 currently returns complete successful tool results to the model.
    result = agent.run(args.query)
    print(
        json.dumps(
            _compact_cli_result(result),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileNotFoundError,
        ImportError,
        KeyError,
        OpenAIError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
