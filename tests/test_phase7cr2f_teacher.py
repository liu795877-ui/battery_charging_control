import json
from pathlib import Path

import numpy as np
import pytest

from battery_fast_charge.phase7a_level3_model import Level3MPCResult
from battery_fast_charge.phase7cr2f_teacher import (
    StrictTeacherSelectionError,
    select_qualified_teacher_candidate,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/phase7cr2_known_teacher_misselections.json"


def _result(payload: dict) -> Level3MPCResult:
    return Level3MPCResult(
        current_a=payload["current_a"],
        plan_a=np.asarray([payload["current_a"]]),
        objective_value=payload["objective_value"],
        optimizer_success=payload["optimizer_success"],
        prediction_feasible=payload["prediction_feasible"],
        used_fallback=False,
        status=payload["status"],
        solve_time_s=0.0,
        maximum_voltage_v=4.0,
        minimum_constraint_margin=0.1,
        maximum_current_step_a=0.0,
        minimum_slew_margin_a=0.1,
    )


@pytest.mark.parametrize(
    "case",
    json.loads(FIXTURE.read_text(encoding="utf-8")),
    ids=lambda case: case["case_id"],
)
def test_r2_known_failed_alternative_can_never_override_successful_default(
    case: dict,
) -> None:
    selection = select_qualified_teacher_candidate(
        {
            "default": _result(case["default"]),
            "alternative": _result(case["alternative"]),
        }
    )
    assert selection.label == case["expected_selected_branch"]
    assert selection.qualified_labels == ("default",)


def test_successful_alternative_replaces_unqualified_default() -> None:
    failed = _result(
        {
            "current_a": 3.0,
            "objective_value": 1.0,
            "optimizer_success": False,
            "prediction_feasible": True,
            "status": "failed",
        }
    )
    successful = _result(
        {
            "current_a": 2.0,
            "objective_value": 2.0,
            "optimizer_success": True,
            "prediction_feasible": True,
            "status": "success",
        }
    )
    assert select_qualified_teacher_candidate(
        {"default": failed, "alternative": successful}
    ).label == "alternative"


def test_objective_is_compared_only_among_qualified_candidates() -> None:
    first = _result(
        {
            "current_a": 3.0,
            "objective_value": 2.0,
            "optimizer_success": True,
            "prediction_feasible": True,
            "status": "success",
        }
    )
    second = _result(
        {
            "current_a": 2.0,
            "objective_value": 1.0,
            "optimizer_success": True,
            "prediction_feasible": True,
            "status": "success",
        }
    )
    assert select_qualified_teacher_candidate(
        {"default": first, "alternative": second}
    ).label == "alternative"


def test_all_unqualified_candidates_raise_strict_failure() -> None:
    failed = _result(
        {
            "current_a": 3.0,
            "objective_value": 1.0,
            "optimizer_success": False,
            "prediction_feasible": True,
            "status": "failed",
        }
    )
    infeasible = _result(
        {
            "current_a": 2.0,
            "objective_value": 2.0,
            "optimizer_success": True,
            "prediction_feasible": False,
            "status": "infeasible",
        }
    )
    with pytest.raises(StrictTeacherSelectionError):
        select_qualified_teacher_candidate(
            {"default": failed, "alternative": infeasible}
        )
