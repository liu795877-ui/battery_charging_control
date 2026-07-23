"""混合最短时间教师的降阶模型与Chen2020 DFN闭环。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .closed_loop import (
    Chen2020DFNPlant,
    _cap_current_at_target,
    _correct_reduced_state_from_dfn,
    initial_reduced_state,
)
from .hybrid_teacher import HybridMinimumTimeTeacher, HybridTeacherDecision
from .mpc import ReducedBatteryModel, ReducedState
from .phase3_config import PhaseThreeConfig
from .phase4b_config import PhaseFourBConfig


def _initial_record(
    model: ReducedBatteryModel, state: ReducedState, source: str
) -> dict[str, Any]:
    """生成统一的初始轨迹行。"""
    return {
        "time_s": 0.0,
        "charge_current_a": 0.0,
        "soc": state.soc,
        "terminal_voltage_v": model.ocv(state.soc),
        "average_temperature_c": model.average_temperature(state),
        "control_mode": "initial",
        "optimizer_success": True,
        "prediction_feasible": True,
        "used_fallback": False,
        "safety_override": False,
        "target_current_cap_active": False,
        "solve_time_s": 0.0,
        "predicted_maximum_voltage_v": np.nan,
        "predicted_maximum_temperature_c": np.nan,
        "source": source,
    }


def _control_record(
    time_s: float,
    current_a: float,
    soc: float,
    voltage_v: float,
    temperature_c: float,
    decision: HybridTeacherDecision,
    target_cap: bool,
    source: str,
) -> dict[str, Any]:
    """保留模式、优化器、终端调节器和物理输出诊断。"""
    return {
        "time_s": time_s,
        "charge_current_a": current_a,
        "soc": soc,
        "terminal_voltage_v": voltage_v,
        "average_temperature_c": temperature_c,
        "control_mode": decision.mode,
        "optimizer_success": decision.optimizer_success,
        "prediction_feasible": decision.prediction_feasible,
        "used_fallback": decision.used_fallback,
        "safety_override": decision.safety_override,
        "target_current_cap_active": target_cap,
        "solve_time_s": decision.solve_time_s,
        "predicted_maximum_voltage_v": decision.predicted_maximum_voltage_v,
        "predicted_maximum_temperature_c": decision.predicted_maximum_temperature_c,
        "source": source,
    }


def simulate_hybrid_reduced_closed_loop(
    model: ReducedBatteryModel,
    phase3: PhaseThreeConfig,
    phase4b: PhaseFourBConfig,
) -> pd.DataFrame:
    """在降阶模型上每5 s更新一次混合教师动作。"""
    teacher = HybridMinimumTimeTeacher(model, phase3, phase4b)
    state = initial_reduced_state(phase3)
    records = [_initial_record(model, state, "reduced_model")]
    maximum_steps = int(
        np.ceil(
            phase3.control.maximum_simulation_time_s
            / phase3.control.control_interval_s
        )
    )
    for step_index in range(1, maximum_steps + 1):
        decision = teacher.decide(state)
        current, target_cap = _cap_current_at_target(
            decision.current_a, state.soc, phase3
        )
        state, output = model.step(state, current)
        records.append(
            _control_record(
                step_index * phase3.control.control_interval_s,
                current,
                state.soc,
                output.terminal_voltage_v,
                output.average_temperature_c,
                decision,
                target_cap,
                "reduced_model",
            )
        )
        if state.soc >= phase3.battery.target_soc - phase3.validation.target_soc_tolerance:
            break
    return pd.DataFrame.from_records(records)


def simulate_hybrid_dfn_closed_loop(
    model: ReducedBatteryModel,
    phase3: PhaseThreeConfig,
    phase4b: PhaseFourBConfig,
) -> pd.DataFrame:
    """把混合教师动作施加到DFN并反馈SOC与平均温度。"""
    teacher = HybridMinimumTimeTeacher(model, phase3, phase4b)
    plant = Chen2020DFNPlant(phase3)
    state = initial_reduced_state(phase3)
    records = [_initial_record(model, state, "chen2020_dfn")]
    maximum_steps = int(
        np.ceil(
            phase3.control.maximum_simulation_time_s
            / phase3.control.control_interval_s
        )
    )
    for _ in range(maximum_steps):
        decision = teacher.decide(state)
        current, target_cap = _cap_current_at_target(
            decision.current_a, state.soc, phase3
        )
        predicted_state, _ = model.step(state, current)
        measurement = plant.step(current)
        state = _correct_reduced_state_from_dfn(
            predicted_state, measurement, model, current
        )
        records.append(
            _control_record(
                measurement["time_s"],
                current,
                measurement["soc"],
                measurement["terminal_voltage_v"],
                measurement["average_temperature_c"],
                decision,
                target_cap,
                "chen2020_dfn",
            )
        )
        if state.soc >= phase3.battery.target_soc - phase3.validation.target_soc_tolerance:
            break
    return pd.DataFrame.from_records(records)


def hybrid_closed_loop_metrics(
    frame: pd.DataFrame,
    phase3: PhaseThreeConfig,
    phase4b: PhaseFourBConfig,
) -> dict[str, Any]:
    """同时检查物理约束、MPC可靠性和终端安全调节器。"""
    tolerance = phase3.validation.physical_constraint_tolerance
    control_rows = frame.iloc[1:]
    mpc_rows = control_rows[
        control_rows["control_mode"] == "thermal_budget_mpc"
    ]
    terminal_rows = control_rows[
        control_rows["control_mode"] == "terminal_reference_governor"
    ]
    startup_rows = control_rows[
        control_rows["control_mode"] == "startup_reference_governor"
    ]
    current_change = frame["charge_current_a"].diff().abs().fillna(0.0)
    reached = bool(
        frame["soc"].iloc[-1]
        >= phase3.battery.target_soc - phase3.validation.target_soc_tolerance
    )
    optimizer_fraction = float(mpc_rows["optimizer_success"].mean())
    feasible_fraction = float(mpc_rows["prediction_feasible"].mean())
    metrics = {
        "source": str(frame["source"].iloc[-1]),
        "reached_target_soc": reached,
        "final_soc": float(frame["soc"].iloc[-1]),
        "charge_time_min": float(frame["time_s"].iloc[-1] / 60.0) if reached else None,
        "maximum_voltage_v": float(frame["terminal_voltage_v"].max()),
        "maximum_temperature_c": float(frame["average_temperature_c"].max()),
        "maximum_current_a": float(frame["charge_current_a"].max()),
        "maximum_current_change_a": float(current_change.max()),
        "voltage_limit_exceeded": bool(
            frame["terminal_voltage_v"].max()
            > phase3.constraints.physical_maximum_voltage_v + tolerance
        ),
        "temperature_limit_exceeded": bool(
            frame["average_temperature_c"].max()
            > phase3.constraints.physical_maximum_temperature_c + tolerance
        ),
        "current_limit_exceeded": bool(
            frame["charge_current_a"].max()
            > phase3.constraints.maximum_current_a + tolerance
        ),
        "current_change_limit_exceeded": bool(
            current_change.max()
            > phase3.constraints.maximum_current_change_a_per_step + tolerance
        ),
        "mpc_step_count": int(len(mpc_rows)),
        "startup_governor_step_count": int(len(startup_rows)),
        "terminal_governor_step_count": int(len(terminal_rows)),
        "optimizer_success_fraction": optimizer_fraction,
        "prediction_feasible_fraction": feasible_fraction,
        "fallback_count": int(mpc_rows["used_fallback"].sum()),
        "reference_governor_safety_override_count": int(
            pd.concat([startup_rows, terminal_rows])["safety_override"].sum()
        ),
        "mean_mpc_solve_time_ms": float(mpc_rows["solve_time_s"].mean() * 1000.0),
        "maximum_mpc_solve_time_ms": float(mpc_rows["solve_time_s"].max() * 1000.0),
    }
    criteria = phase4b.success_criteria
    metrics["success"] = bool(
        reached
        and not metrics["voltage_limit_exceeded"]
        and not metrics["temperature_limit_exceeded"]
        and not metrics["current_limit_exceeded"]
        and not metrics["current_change_limit_exceeded"]
        and optimizer_fraction >= criteria.minimum_optimizer_success_fraction
        and feasible_fraction >= criteria.minimum_optimizer_success_fraction
        and (
            metrics["fallback_count"] == 0
            if criteria.require_zero_fallbacks
            else True
        )
        and metrics["reference_governor_safety_override_count"] == 0
    )
    return metrics
