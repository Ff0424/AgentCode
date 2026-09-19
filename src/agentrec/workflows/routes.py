"""Closed routing vocabulary for the deterministic shopping workflow."""

from enum import Enum


class WorkflowRoute(str, Enum):
    CONTINUE = "continue"
    READY = "ready"
    CONFLICT = "conflict"
    ERROR = "error"


class WorkflowAction(str, Enum):
    SELECT_REQUIREMENT = "select_requirement"
    RECOMMEND = "recommend"
    RETRIEVE_EVIDENCE = "retrieve_evidence"
    VERIFY_CONSTRAINTS = "verify_constraints"
    SELECT_CANDIDATE = "select_candidate"
    UPDATE_PLAN = "update_plan"
    EVALUATE = "evaluate"
    READY = "ready"
    CONFLICT = "conflict"
