"""使用一步约束过滤器构造与 MPC 同边界的非优化充电基线。"""

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
from .mpc import ReducedBatteryModel
from .phase3_config import PhaseThreeConfig
from .teacher_data import filter_feasible_current


def _record(
    time_s: float,
    current_a: float,
    soc: float,
    voltage_v: float,
    temperature_c: float,
    safety_override: bool,
    source: str,
) -> dict[str, Any]:
    """统一降阶和DFN公平基线的列结构。"""
    return {
        "time_s": time_s,
        "charge_current_a": current_a,
        "soc": soc,
        "terminal_voltage_v": voltage_v,
        "average_temperature_c": temperature_c,
        "safety_override": safety_override,
        "source": source,
    }


def simulate_filtered_baseline_reduced(
    model: ReducedBatteryModel,
    config: PhaseThreeConfig,
    desired_current_a: float,
    maximum_time_s: float,
) -> pd.DataFrame:
    """在降阶模型上运行“固定目标电流＋一步安全过滤”基线。"""
    state = initial_reduced_state(config)
    records = [
        _record(
            0.0,
            0.0,
            state.soc,
            model.ocv(state.soc),
            model.average_temperature(state),
            False,
            "reduced_model",
        )
    ]
    maximum_steps = int(np.ceil(maximum_time_s / config.control.control_interval_s))
    for step_index in range(1, maximum_steps + 1):
        filtered = filter_feasible_current(model, state, desired_current_a, config)
        current, _ = _cap_current_at_target(filtered.current_a, state.soc, config)
        state, output = model.step(state, current)
        records.append(
            _record(
                step_index * config.control.control_interval_s,
                current,
                state.soc,
                output.terminal_voltage_v,
                output.average_temperature_c,
                filtered.safety_override,
                "reduced_model",
            )
        )
        if state.soc >= config.battery.target_soc - config.validation.target_soc_tolerance:
            break
    return pd.DataFrame.from_records(records)


def simulate_filtered_baseline_dfn(
    model: ReducedBatteryModel,
    config: PhaseThreeConfig,
    desired_current_a: float,
    maximum_time_s: float,
) -> pd.DataFrame:
    """将同一个非优化基线逐步施加到Chen2020 DFN并反馈状态。"""
    plant = Chen2020DFNPlant(config)
    state = initial_reduced_state(config)
    records = [
        _record(
            0.0,
            0.0,
            state.soc,
            model.ocv(state.soc),
            model.average_temperature(state),
            False,
            "chen2020_dfn",
        )
    ]
    maximum_steps = int(np.ceil(maximum_time_s / config.control.control_interval_s))
    for _ in range(maximum_steps):
        filtered = filter_feasible_current(model, state, desired_current_a, config)
        current, _ = _cap_current_at_target(filtered.current_a, state.soc, config)
        predicted_state, _ = model.step(state, current)
        measurement = plant.step(current)
        state = _correct_reduced_state_from_dfn(
            predicted_state, measurement, model, current
        )
        records.append(
            _record(
                measurement["time_s"],
                current,
                measurement["soc"],
                measurement["terminal_voltage_v"],
                measurement["average_temperature_c"],
                filtered.safety_override,
                "chen2020_dfn",
            )
        )
        if state.soc >= config.battery.target_soc - config.validation.target_soc_tolerance:
            break
    return pd.DataFrame.from_records(records)


def filtered_baseline_metrics(
    frame: pd.DataFrame, config: PhaseThreeConfig
) -> dict[str, Any]:
    """计算与第三阶段A相同的终端、约束和时间指标。"""
    tolerance = config.validation.physical_constraint_tolerance
    reached = bool(
        frame["soc"].iloc[-1]
        >= config.battery.target_soc - config.validation.target_soc_tolerance
    )
    current_difference = frame["charge_current_a"].diff().abs().fillna(0.0)
    metrics = {
        "source": str(frame["source"].iloc[-1]),
        "reached_target_soc": reached,
        "final_soc": float(frame["soc"].iloc[-1]),
        "charge_time_min": float(frame["time_s"].iloc[-1] / 60.0) if reached else None,
        "maximum_voltage_v": float(frame["terminal_voltage_v"].max()),
        "maximum_temperature_c": float(frame["average_temperature_c"].max()),
        "maximum_current_a": float(frame["charge_current_a"].max()),
        "maximum_current_change_a": float(current_difference.max()),
        "safety_override_count": int(frame["safety_override"].sum()),
        "voltage_limit_exceeded": bool(
            frame["terminal_voltage_v"].max()
            > config.constraints.physical_maximum_voltage_v + tolerance
        ),
        "temperature_limit_exceeded": bool(
            frame["average_temperature_c"].max()
            > config.constraints.physical_maximum_temperature_c + tolerance
        ),
        "current_limit_exceeded": bool(
            frame["charge_current_a"].max()
            > config.constraints.maximum_current_a + tolerance
        ),
        "current_change_limit_exceeded": bool(
            current_difference.max()
            > config.constraints.maximum_current_change_a_per_step + tolerance
        ),
    }
    metrics["success"] = bool(
        reached
        and not metrics["voltage_limit_exceeded"]
        and not metrics["temperature_limit_exceeded"]
        and not metrics["current_limit_exceeded"]
        and not metrics["current_change_limit_exceeded"]
    )
    return metrics
