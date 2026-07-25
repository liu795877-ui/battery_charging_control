"""Phase 7C-R2F：严格教师分支资格与选择规则。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .phase7a_level3_model import Level3MPC, Level3MPCResult, Level3State
from .phase7cr1_config import Phase7CR1Config


class StrictTeacherSelectionError(RuntimeError):
    """所有教师分支均不满足优化成功且预测可行时抛出。"""

    def __init__(self, diagnostics: dict[str, Any]) -> None:
        super().__init__("No optimizer-successful and prediction-feasible branch.")
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class TeacherSelection:
    label: str
    result: Level3MPCResult
    qualified_labels: tuple[str, ...]


def is_qualified_teacher_candidate(result: Level3MPCResult) -> bool:
    return bool(result.optimizer_success and result.prediction_feasible)


def select_qualified_teacher_candidate(
    candidates: Mapping[str, Level3MPCResult],
) -> TeacherSelection:
    qualified = {
        label: result
        for label, result in candidates.items()
        if is_qualified_teacher_candidate(result)
    }
    if not qualified:
        raise StrictTeacherSelectionError(
            {
                label: {
                    "optimizer_success": bool(result.optimizer_success),
                    "prediction_feasible": bool(result.prediction_feasible),
                    "objective_value": float(result.objective_value),
                    "status": result.status,
                }
                for label, result in candidates.items()
            }
        )
    label, result = min(
        qualified.items(), key=lambda item: item[1].objective_value
    )
    return TeacherSelection(
        label=label,
        result=result,
        qualified_labels=tuple(qualified),
    )


def solve_teacher_r2f(
    state: Level3State,
    model: Any,
    config: Phase7CR1Config,
) -> tuple[Level3MPCResult, dict[str, Any]]:
    default = Level3MPC(model).solve(state)
    maximum_step = model.config.constraint.maximum_current_step_a
    lower = max(
        model.inherited.mpc.current_bounds_a[0],
        state.previous_current_a - maximum_step,
    )
    upper = min(
        model.inherited.mpc.current_bounds_a[1],
        state.previous_current_a + maximum_step,
    )
    at_slew_boundary = (
        abs(default.current_a - lower)
        <= config.teacher["slew_boundary_tolerance_a"]
        or abs(default.current_a - upper)
        <= config.teacher["slew_boundary_tolerance_a"]
    )
    retry = (
        not is_qualified_teacher_candidate(default)
        or (
            at_slew_boundary
            and config.teacher["compare_alternative_on_slew_boundary"]
        )
    )
    candidates: dict[str, Level3MPCResult] = {"default": default}
    if retry:
        alternative_mpc = Level3MPC(model)
        alternative_mpc.set_warm_start(
            np.full(
                alternative_mpc.number_of_blocks,
                state.previous_current_a,
            )
        )
        candidates["alternative"] = alternative_mpc.solve(state)
    selection = select_qualified_teacher_candidate(candidates)
    alternative = candidates.get("alternative")
    return selection.result, {
        "teacher_retry_triggered": retry,
        "selected_teacher_branch": selection.label,
        "qualified_teacher_branches": list(selection.qualified_labels),
        "default_optimizer_success": default.optimizer_success,
        "default_prediction_feasible": default.prediction_feasible,
        "default_current_a": default.current_a,
        "default_objective": default.objective_value,
        "alternative_optimizer_success": (
            alternative.optimizer_success if alternative is not None else True
        ),
        "alternative_prediction_feasible": (
            alternative.prediction_feasible if alternative is not None else True
        ),
        "alternative_current_a": (
            alternative.current_a if alternative is not None else np.nan
        ),
        "alternative_objective": (
            alternative.objective_value if alternative is not None else np.nan
        ),
        "alternative_selected": selection.label == "alternative",
        "teacher_branch_objective_improvement": (
            default.objective_value - selection.result.objective_value
        ),
    }
