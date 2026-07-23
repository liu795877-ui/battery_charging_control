import json
from pathlib import Path

import pandas as pd

from battery_fast_charge.identification import build_ocv_function
from battery_fast_charge.mpc import ConstrainedMPC, ReducedBatteryModel, ReducedState
from battery_fast_charge.phase3_config import load_phase_three_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _model() -> tuple[ReducedBatteryModel, object]:
    config = load_phase_three_config(PROJECT_ROOT / "configs" / "phase3.yaml")
    parameters = json.loads(
        (PROJECT_ROOT / config.artifacts.identified_parameters).read_text(
            encoding="utf-8"
        )
    )
    ocv = pd.read_csv(PROJECT_ROOT / config.artifacts.ocv_curve)
    return ReducedBatteryModel(config, build_ocv_function(ocv), parameters), config


def test_reduced_step_uses_positive_charge_current() -> None:
    """正电流必须提高 SOC，并使端电压高于 OCV。"""
    model, config = _model()
    state = ReducedState(0.10, 0.0, 0.0, 25.0, 25.0, 0.0)
    next_state, output = model.step(state, 5.0)

    assert next_state.soc > state.soc
    assert output.terminal_voltage_v > model.ocv(next_state.soc)
    assert output.average_temperature_c >= 25.0
    assert next_state.previous_current_a == 5.0


def test_first_mpc_action_is_bounded_and_feasible() -> None:
    """低 SOC 初始状态下的首个动作应服从电流变化率和预测约束。"""
    model, config = _model()
    controller = ConstrainedMPC(model, config)
    state = ReducedState(0.10, 0.0, 0.0, 25.0, 25.0, 0.0)
    result = controller.solve(state)

    assert 0.0 <= result.current_a <= config.constraints.maximum_current_a
    assert result.current_a <= config.constraints.maximum_current_change_a_per_step + 1.0e-6
    assert result.prediction_feasible
    assert result.predicted_maximum_voltage_v <= config.constraints.mpc_maximum_voltage_v + 1.0e-4
    assert result.predicted_maximum_temperature_c <= config.constraints.mpc_maximum_temperature_c + 1.0e-4


def test_temperature_just_above_tightened_limit_can_recover() -> None:
    """当前温度不可改变；只要下一步能冷却回边界，就不应把 MPC 判为无解。"""
    model, config = _model()
    state = ReducedState(0.30, 0.0, 0.0, 33.51, 33.51, 0.0)
    _, zero_current_output = model.step(state, 0.0)
    result = ConstrainedMPC(model, config).solve(state)

    assert zero_current_output.constraint_temperature_c < 33.51
    assert result.prediction_feasible
