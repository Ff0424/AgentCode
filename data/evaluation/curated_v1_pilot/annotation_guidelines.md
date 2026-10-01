# Curated Benchmark V1 Annotation Guidelines

## Human ground truth

Annotate the meaning of the user request, not the current system output. Every
record starts as `pending`. A second reviewer must resolve ambiguity before a
case can become `approved` and enter formal scoring.

The following are prohibited label sources:

- A DeepSeek or other model prediction must not become `expected_goal` truth.
- Current Verifier output must not become product or evidence truth.
- Current Recommendation top-1 must not become the only correct product.

## Goal annotation

Preserve the order in which requirements appear. Record one proposal per
semantic requirement. Do not duplicate a proposal to represent quantity.
Allocation `target_index` refers to this stable ordered tuple.

## Hard and soft constraints

A mandatory capability belongs in `required_features`. A preference such as
light weight, style, or a non-mandatory brand belongs in `soft_preferences`.
Do not promote a preference to a hard constraint. Preserve an explicitly stated
unknown hard constraint even when the current vocabulary cannot resolve it.

## Budget

Distinguish total-goal budget from per-requirement maximum budget. Do not infer
a number from phrases such as "inexpensive" or "save money". Missing or
ambiguous required budget information should produce a clarification label.
Monetary annotations use the explicit amount in the request.

## Quantity

Quantity is a positive integer on one requirement proposal. For example, two
mice are one proposal with `quantity=2`, not two repeated proposals.

## Grounding

Grounding labels use only the frozen deterministic vocabulary and wrapper
rules. `grounded` requires a canonical constraint. `unresolved` must preserve
the original text and must not invent a canonical value. No fuzzy semantic
matching is allowed.

## Product truth

Annotate whether a concrete candidate satisfies one hard constraint using an
independent source recorded in `provenance`. Do not treat recommendation rank,
title text, or model score as factual truth. More than one product may be valid.

## Evidence truth

Label evidence as `supported`, `unknown`, or `contradicted` only after reviewing
the referenced content and product identity. Do not copy the production
Verifier result. Preserve enough provenance for another reviewer to repeat the
judgment.

## Conflict and clarification

Clarification is a normal business outcome, not a system failure. Use conflict
only when execution has a complete goal but cannot safely produce a valid plan.
Do not convert provider, artifact, or programming errors into business conflict.

## Re-plan

Record whether bounded retry is not expected, expected to recover, or expected
to exhaust. A re-plan label must preserve category, budget, and hard constraints;
it must not assume that requirements are relaxed.

## Adjudication

1. A primary annotator drafts the goal and business expectation.
2. Deterministic reference rules may derive grounding, allocation, and tool
   arguments, but their output must be reviewed rather than copied blindly.
3. A separate reviewer checks hard/soft separation, budget, quantity, product
   truth, evidence truth, and terminal outcome.
4. Disagreement is marked `disputed`; incomplete work remains `pending`.
5. Only `approved` cases may enter formal scoring.

The Pilot remains a development asset. Using a case to tune prompts, aliases,
or scorers permanently disqualifies it from a future blind test.
