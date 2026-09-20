"""Deterministic user-facing rendering for grounded response contexts.

The renderer consumes only response-safe projected contexts. It performs no
planning, retrieval, verification, service calls, workflow access, or state
mutation, and it never renders raw evidence excerpts or internal rationale.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

from .contracts import (
    ConflictReason,
    ConflictResponseContext,
    FinalResponseResult,
    GroundedResponseContext,
    ReadyResponseContext,
    ResponseKind,
)


def _money(value: float) -> str:
    """Format an already validated finite money fact deterministically."""

    return format(
        Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
        ".2f",
    )


def _feature_list(values: tuple[str, ...]) -> str:
    return "、".join(values)


class DeterministicFinalResponseRenderer:
    """Render a grounded READY or CONFLICT context with fixed templates."""

    def render(self, context: GroundedResponseContext) -> FinalResponseResult:
        if isinstance(context, ReadyResponseContext):
            return self._render_ready(context)
        if isinstance(context, ConflictResponseContext):
            return self._render_conflict(context)
        raise TypeError(
            "context must be a ReadyResponseContext or ConflictResponseContext."
        )

    @staticmethod
    def _render_ready(context: ReadyResponseContext) -> FinalResponseResult:
        lines = ["购物方案已准备完成。", "", "已选择商品："]
        for rank, product in enumerate(context.products, start=1):
            lines.extend((
                f"{rank}. {product.category}",
                f"   商品：{product.title}",
                f"   Product ID：{product.parent_asin}",
                f"   数量：{product.quantity}",
                f"   单价：{context.currency} {_money(product.price)}",
            ))
            if product.verified_claims:
                constraints = tuple(
                    value.original_constraint for value in product.verified_claims
                )
                lines.append(f"   已验证的要求：{_feature_list(constraints)}")
            if rank != len(context.products):
                lines.append("")
        lines.extend((
            "",
            f"方案总金额：{context.currency} {_money(context.total_spent)}",
            f"剩余预算：{context.currency} {_money(context.remaining_budget)}",
        ))
        return FinalResponseResult(
            kind=ResponseKind.READY,
            text="\n".join(lines),
            decision_summary=context.products,
        )

    @staticmethod
    def _render_conflict(context: ConflictResponseContext) -> FinalResponseResult:
        decision = context.decision
        lines = [
            "当前购物要求暂时无法完成。",
            "",
            f"类别：{decision.category}",
        ]
        if decision.required_features:
            lines.append(
                f"硬性要求：{_feature_list(decision.required_features)}"
            )
        lines.append("")

        if decision.reason is ConflictReason.NO_RECOMMENDATION_CANDIDATES:
            if decision.replan_attempts_performed == 1:
                lines.append(
                    "系统已扩大候选范围，但在当前结构化推荐约束下仍没有返回候选商品。"
                )
            else:
                lines.append("在当前结构化推荐约束下没有返回候选商品。")
        elif decision.reason is ConflictReason.REPLAN_ATTEMPTS_EXHAUSTED:
            lines.extend((
                "系统已执行一次有界候选池扩展，"
                "但仍未找到所有硬性要求都得到证据支持的商品。",
            ))
            if decision.unknown_constraints:
                lines.extend((
                    "",
                    "以下要求证据不足，无法确认满足要求：",
                    *(f"- {value}" for value in decision.unknown_constraints),
                ))
            if decision.contradicted_constraints:
                lines.extend((
                    "",
                    "以下要求已有证据与要求存在冲突：",
                    *(f"- {value}" for value in decision.contradicted_constraints),
                ))
        else:  # pragma: no cover - closed enum and validated contract
            raise ValueError(f"Unsupported conflict reason={decision.reason!r}.")

        lines.extend((
            "",
            "可以调整预算、类别或硬性要求后重新尝试。",
        ))
        return FinalResponseResult(
            kind=ResponseKind.CONFLICT,
            text="\n".join(lines),
            decision_summary=decision,
        )
