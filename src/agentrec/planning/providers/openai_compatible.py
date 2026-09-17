"""OpenAI-compatible Chat Completions provider for StructuredLLMPlanner."""

from __future__ import annotations

import os
from typing import Any

from .base import PlannerProviderError, PlannerTimeoutError


class OpenAICompatiblePlannerProvider:
    """Call an OpenAI-compatible endpoint and return raw message content."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        api_key_env: str = "DEEPSEEK_API_KEY",
        client: Any | None = None,
    ) -> None:
        for name, value in (("model", model), ("base_url", base_url)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        self._model = model.strip()
        if client is not None:
            self._client = client
            return
        resolved_key = api_key if api_key is not None else os.getenv(api_key_env)
        if not isinstance(resolved_key, str) or not resolved_key.strip():
            raise PlannerProviderError(
                f"Missing API key; pass api_key or set {api_key_env}."
            )
        # Delay the SDK import until a real provider instance is requested.
        from openai import OpenAI

        self._client = OpenAI(api_key=resolved_key.strip(), base_url=base_url.strip())

    def complete(
        self,
        *,
        messages: tuple[dict[str, str], ...],
        timeout_seconds: float,
    ) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=list(messages),
                response_format={"type": "json_object"},
                temperature=0.0,
                timeout=timeout_seconds,
                extra_body={"thinking": {"type": "disabled"}},
            )
        except TimeoutError as exc:
            raise PlannerTimeoutError("Planner provider request timed out.") from exc
        except Exception as exc:
            # Import lazily and recognize SDK errors without making tests depend on it.
            try:
                from openai import APITimeoutError, OpenAIError
            except ImportError:
                APITimeoutError = ()  # type: ignore[assignment]
                OpenAIError = ()  # type: ignore[assignment]
            if APITimeoutError and isinstance(exc, APITimeoutError):
                raise PlannerTimeoutError("Planner provider request timed out.") from exc
            if OpenAIError and isinstance(exc, OpenAIError):
                raise PlannerProviderError("Planner provider request failed.") from exc
            raise PlannerProviderError("Planner provider request failed.") from exc
        choices = getattr(response, "choices", None)
        if not choices:
            raise PlannerProviderError("Planner provider response contained no choices.")
        message = getattr(choices[0], "message", None)
        content = None if message is None else getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise PlannerProviderError("Planner provider response contained no text.")
        return content
