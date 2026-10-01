# AgentRec Curated Benchmark V1 Pilot

This directory is the canonical data location for the first AgentRec evaluation
pilot. It contains 20 manually drafted `EvaluationCase` records and validates
the Stage 5.2 schema before a larger benchmark is annotated.

## Purpose

The pilot checks whether the frozen evaluation contract can represent goal
understanding, deterministic planning, backend feasibility, and end-to-end
business outcomes without using runtime predictions as labels.

It is an annotation and schema validation set. It is not a blind test set, a
complete Agent benchmark, or evidence of statistical significance. It contains
no Base-versus-LoRA conclusion.

## Tracks

- Track A (9 cases): structured goal extraction. These cases focus on ordered
  requirements, budgets, quantity, hard and soft constraints, allocation
  preferences, and clarification.
- Track B (5 cases): deterministic grounding, projection, allocation, tool
  arguments, and identity/order invariants. These cases do not measure LLM
  reasoning.
- Track C (6 cases): full end-to-end business expectations covering normal
  execution, zero candidates, evidence outcomes, bounded re-planning, and
  fallback. Product and evidence annotations remain empty until independent
  artifact review.

Every case is in the `development` split and has `pending` adjudication status.

## Loading

Read `cases.jsonl` one UTF-8 line at a time and validate every object with
`agentrec.evaluation.EvaluationCase.model_validate_json`. Loading the dataset
must not initialize a model, GPU, recommendation service, or external client.

## Product evaluation

A recommended product is not graded by requiring one exact `parent_asin`.
Multiple products may legitimately satisfy the same constraints, and ranking
can vary with the frozen user and recommendation artifacts. Future end-to-end
scoring should check canonical identity, budget compliance, independently
annotated hard-constraint truth, evidence support, and terminal business
outcome.

## Current annotation status

All 20 records are schema-complete drafts but remain pending human
adjudication. Track C product truth, evidence truth, backend feasibility,
re-plan recovery, and fallback outcome require independent artifact review.
They must not be filled from current Recommendation or Verifier output.

## Expansion

After the annotation workflow and scorers are validated, expand to 50–100
independently annotated cases. Freeze train/development/test splits only then.
Cases used for prompt or scorer development must remain outside the final blind
test, and near-duplicate paraphrases must stay in the same split.
