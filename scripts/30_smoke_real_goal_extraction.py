"""Read-only smoke for real structured shopping-goal extraction.

The script calls the configured OpenAI-compatible provider and validates its
JSON response through ``StructuredGoalExtractor``. It does not load or invoke
Agent workflow, recommendation, retrieval, verification, or response layers.

Run from the repository root after exporting ``DEEPSEEK_API_KEY``::

    python scripts/30_smoke_real_goal_extraction.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agentrec.planning import (  # noqa: E402
    OpenAICompatiblePlannerProvider,
    StructuredGoalExtractor,
)


MODEL = "deepseek-v4-flash"
REQUEST = """I need a $500 shopping setup with a docking station,
a mouse, and headphones.
The docking station must support HDMI.
Save more on the mouse and spend more on headphones."""


def _print_decision(decision) -> None:
    print("=== Real Goal Extraction ===")
    print()
    print(f"total_budget={decision.total_budget}")
    print()
    print("requirements:")
    for index, requirement in enumerate(decision.requirement_proposals):
        print()
        print(f"- index={index}")
        print(f"  category={requirement.category}")
        print(f"  quantity={requirement.quantity}")
        print(f"  required_features={requirement.required_features}")

    print()
    print("allocation_preferences:")
    for preference in decision.allocation_preferences:
        print()
        print(f"- target_index={preference.target_index}")
        print(f"  preference={preference.preference.value}")

    print()
    print(f"clarification_needed={decision.clarification_needed}")
    if decision.clarification_question is not None:
        print(f"clarification_question={decision.clarification_question}")


def main() -> int:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not isinstance(api_key, str) or not api_key.strip():
        print(
            "ERROR: DEEPSEEK_API_KEY is not set in the process environment.",
            file=sys.stderr,
        )
        return 2

    # The provider keeps its production default base_url and reads the key from
    # the process environment. This script intentionally does not load .env.
    provider = OpenAICompatiblePlannerProvider(model=MODEL)
    extractor = StructuredGoalExtractor(
        provider=provider,
        timeout_seconds=30,
        schema_retries=1,
    )
    decision = extractor.extract(user_request=REQUEST)
    _print_decision(decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
