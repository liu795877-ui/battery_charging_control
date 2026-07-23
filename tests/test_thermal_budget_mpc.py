import json
from pathlib import Path

import numpy as np
import pandas as pd

from battery_fast_charge.hybrid_teacher import HybridMinimumTimeTeacher
from battery_fast_charge.identification import build_ocv_function
from battery_fast_charge.mpc import ReducedBatteryModel, ReducedState
from battery_fast_charge.phase3_config import load_phase_three_config
from battery_fast_charge.phase4b_config import load_phase_four_b_config
from battery_fast_charge.thermal_budget_mpc import ThermalBudgetMPC

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _setup():
    phase3 = load_phase_three_config(PROJECT_ROOT / "configs" / "phase3.yaml")
    phase4b = load_phase_four_b_config(PROJECT_ROOT / "configs" / "phase4b.yaml")
    parameters = json.loads(
        (PROJECT_ROOT / phase3.artifacts.identified_parameters).read_text(
            encoding="utf-8"
        )
    )
    ocv = pd.read_csv(PROJECT_ROOT / phase3.artifacts.ocv_curve)
    model = ReducedBatteryModel(phase3, build_ocv_function(ocv), parameters)
    return model, phase3, phase4b


def test_target_terminated_prediction_stops_at_eighty_percent() -> None:
    """预测到达目标后必须保持80% SOC，而不是虚构继续充电状态。"""
    model, phase3, phase4b = _setup()
    controller = ThermalBudgetMPC(model, phase3, phase4b)
    state = ReducedState(0.799, 0.02, 0.03, 30.0, 30.0, 2.0)

    prediction = controller._predict(state, np.full(60, 10.0))

    assert np.max(prediction["soc"]) <= phase3.battery.target_soc + 1.0e-12
    assert np.isclose(prediction["soc"][-1], phase3.battery.target_soc)


def test_hybrid_teacher_uses_terminal_governor_after_release_soc() -> None:
    """20% SOC以后由已验证的终端参考调节器接管。"""
    model, phase3, phase4b = _setup()
    teacher = HybridMinimumTimeTeacher(model, phase3, phase4b)
    state = ReducedState(0.65, 0.03, 0.04, 33.0, 33.0, 4.0)

    decision = teacher.decide(state)

    assert decision.mode == "terminal_reference_governor"
    assert decision.used_fallback is False
    assert decision.safety_override is False


def test_hybrid_teacher_ramps_startup_without_optimizer_fallback() -> None:
    """启动参考调节器应按电流变化约束从零平滑爬升。"""
    model, phase3, phase4b = _setup()
    teacher = HybridMinimumTimeTeacher(model, phase3, phase4b)
    state = ReducedState(0.10, 0.0, 0.0, 25.0, 25.0, 0.0)

    decision = teacher.decide(state)

    assert decision.mode == "startup_reference_governor"
    assert np.isclose(decision.current_a, 2.0)
    assert decision.used_fallback is False
    assert decision.safety_override is False
