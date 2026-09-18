"""Deterministic ShoppingRequirement to evidence-query projection."""

from __future__ import annotations

from ..domain import ShoppingRequirement


class EvidenceQueryBuilder:
    """Build reproducible evidence queries without LLM-generated attributes."""

    def build(self, requirement: ShoppingRequirement) -> str:
        if not isinstance(requirement, ShoppingRequirement):
            raise TypeError("requirement must be a ShoppingRequirement.")
        parts = [f"Category: {requirement.category}."]
        if requirement.required_features:
            parts.append(
                "Required product evidence: "
                + "; ".join(requirement.required_features)
                + "."
            )
        if requirement.soft_preferences:
            parts.append(
                "Preferences: " + "; ".join(requirement.soft_preferences) + "."
            )
        if len(parts) == 1:
            parts.append("Product specifications and capabilities.")
        return " ".join(parts)
