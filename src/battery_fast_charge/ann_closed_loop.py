"""将小型ANN控制器接入降阶模型和Chen2020 DFN做闭环验证。"""

from __future__ import annotations

from time import perf_counter_ns
from typing import Any

import numpy as np
import pandas as pd

from .ann_model import TinyANN
from .closed_loop import (
    Chen2020DFNPlant,
    _cap_current_at_target,
    _correct_reduced_state_from_dfn,
    initial_reduced_state,
)
from .mpc import ReducedBatteryModel, ReducedState
from .phase3_config import PhaseThreeConfig
from .phase4_config import PhaseFourAConfig
from .teacher_data import filter_feasible_current


def ann_features(model: ReducedBatteryModel, state: ReducedState) -> np.ndarray:
    """按训练时约定的顺序构造五个ANN输入。"""
    return np.array(
        [
            state.soc,
            state.polarization_fast_v,
            state.polarization_slow_v,
            model.average_temperature(state),
            state.previous_current_a,
        ],
        dtype=float,
    )


def _ann_decision(
    ann: TinyANN,
    model: ReducedBatteryModel,
    state: ReducedState,
    phase3: PhaseThreeConfig,
) -> dict[str, Any]:
    """计算ANN动作并用与阶段3B相同的一步可行性过滤器包装。"""
    features = ann_features(model, state)
    start = perf_counter_ns()
    unclipped = float(ann.predict_unclipped(features))
    requested = float(ann.predict(features))
    inference_time_s = (perf_counter_ns() - start) * 1.0e-9
    filtered = filter_feasible_current(model, state, requested, phase3)
    return {
        "features": features,
        "unclipped_current_a": unclipped,
        "requested_current_a": requested,
        "filtered_current_a": filtered.current_a,
        "filter_intervened": abs(filtered.current_a - requested) > 1.0e-8,
        "safety_override": filtered.safety_override,
        "inference_time_s": inference_time_s,
    }


def _initial_record(
    model: ReducedBatteryModel, state: ReducedState, source: str
) -> dict[str, Any]:
    """统一降阶模型和DFN轨迹的初始行。"""
    return {
        "time_s": 0.0,
        "charge_current_a": 0.0,
        "ann_unclipped_current_a": np.nan,
        "ann_requested_current_a": np.nan,
        "safety_filtered_current_a": 0.0,
        "safety_filter_intervened": False,
        "safety_override": False,
        "target_current_cap_active": False,
        "ann_inference_time_s": 0.0,
        "soc": state.soc,
        "terminal_voltage_v": model.ocv(state.soc),
        "average_temperature_c": model.average_temperature(state),
        "source": source,
    }


def _control_record(
    time_s: float,
    applied_current_a: float,
    decision: dict[str, Any],
    target_cap_active: bool,
    soc: float,
    voltage_v: float,
    temperature_c: float,
    source: str,
) -> dict[str, Any]:
    """保存原始ANN动作、过滤动作和物理输出，便于区分网络与安全层责任。"""
    return {
        "time_s": time_s,
        "charge_current_a": applied_current_a,
        "ann_unclipped_current_a": decision["unclipped_current_a"],
        "ann_requested_current_a": decision["requested_current_a"],
        "safety_filtered_current_a": decision["filtered_current_a"],
        "safety_filter_intervened": decision["filter_intervened"],
        "safety_override": decision["safety_override"],
        "target_current_cap_active": target_cap_active,
        "ann_inference_time_s": decision["inference_time_s"],
        "soc": soc,
        "terminal_voltage_v": voltage_v,
        "average_temperature_c": temperature_c,
        "source": source,
    }


def simulate_ann_reduced_closed_loop(
    ann: TinyANN,
    model: ReducedBatteryModel,
    phase3: PhaseThreeConfig,
    phase4: PhaseFourAConfig,
) -> pd.DataFrame:
    """每5 s重新计算ANN动作并在降阶模型上执行。"""
    state = initial_reduced_state(phase3)
    records = [_initial_record(model, state, "reduced_model")]
    maximum_steps = int(
        np.ceil(
            phase4.closed_loop.maximum_simulation_time_s
            / phase3.control.control_interval_s
        )
    )
    for step_index in range(1, maximum_steps + 1):
        decision = _ann_decision(ann, model, state, phase3)
        applied, target_cap = _cap_current_at_target(
            decision["filtered_current_a"], state.soc, phase3
        )
        state, output = model.step(state, applied)
        records.append(
            _control_record(
                step_index * phase3.control.control_interval_s,
                applied,
                decision,
                target_cap,
                state.soc,
                output.terminal_voltage_v,
                output.average_temperature_c,
                "reduced_model",
            )
        )
        if state.soc >= phase3.battery.target_soc - phase3.validation.target_soc_tolerance:
            break
    return pd.DataFrame.from_records(records)


def simulate_ann_dfn_closed_loop(
    ann: TinyANN,
    model: ReducedBatteryModel,
    phase3: PhaseThreeConfig,
    phase4: PhaseFourAConfig,
) -> pd.DataFrame:
    """把ANN安全动作逐步施加到Chen2020 DFN并反馈SOC与平均温度。"""
    plant = Chen2020DFNPlant(phase3)
    state = initial_reduced_state(phase3)
    records = [_initial_record(model, state, "chen2020_dfn")]
    maximum_steps = int(
        np.ceil(
            phase4.closed_loop.maximum_simulation_time_s
            / phase3.control.control_interval_s
        )
    )
    for _ in range(maximum_steps):
        decision = _ann_decision(ann, model, state, phase3)
        applied, target_cap = _cap_current_at_target(
            decision["filtered_current_a"], state.soc, phase3
        )
        predicted_state, _ = model.step(state, applied)
        measurement = plant.step(applied)
        state = _correct_reduced_state_from_dfn(
            predicted_state, measurement, model, applied
        )
        records.append(
            _control_record(
                measurement["time_s"],
                applied,
                decision,
                target_cap,
                measurement["soc"],
                measurement["terminal_voltage_v"],
                measurement["average_temperature_c"],
                "chen2020_dfn",
            )
        )
        if state.soc >= phase3.battery.target_soc - phase3.validation.target_soc_tolerance:
            break
    return pd.DataFrame.from_records(records)


def ann_closed_loop_metrics(
    frame: pd.DataFrame,
    phase3: PhaseThreeConfig,
    material_intervention_threshold_a: float = 0.1,
) -> dict[str, Any]:
    """计算终端、物理约束以及安全层介入频率和幅度。

    二元介入标记会把毫安级数值修正和超过1 A的实质修正同等计数，容易
    误导主动学习判断。因此同时报告绝对修正量及超过研究阈值的比例。
    """
    tolerance = phase3.validation.physical_constraint_tolerance
    control_rows = frame.iloc[1:]
    if "safety_filtered_current_a" in control_rows:
        correction = (
            control_rows["ann_requested_current_a"]
            - control_rows["safety_filtered_current_a"]
        ).abs()
    else:
        # 兼容阶段4A已保存的旧CSV；最后一步目标电量封顶可能被计入该近似量。
        correction = (
            control_rows["ann_requested_current_a"]
            - control_rows["charge_current_a"]
        ).abs()
    current_change = frame["charge_current_a"].diff().abs().fillna(0.0)
    reached = bool(
        frame["soc"].iloc[-1]
        >= phase3.battery.target_soc - phase3.validation.target_soc_tolerance
    )
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
        "safety_filter_intervention_count": int(
            control_rows["safety_filter_intervened"].sum()
        ),
        "safety_filter_intervention_fraction": float(
            control_rows["safety_filter_intervened"].mean()
        ),
        "mean_safety_filter_correction_a": float(correction.mean()),
        "p95_safety_filter_correction_a": float(correction.quantile(0.95)),
        "maximum_safety_filter_correction_a": float(correction.max()),
        "material_intervention_threshold_a": float(
            material_intervention_threshold_a
        ),
        "material_safety_filter_intervention_count": int(
            (correction > material_intervention_threshold_a).sum()
        ),
        "material_safety_filter_intervention_fraction": float(
            (correction > material_intervention_threshold_a).mean()
        ),
        "safety_override_count": int(control_rows["safety_override"].sum()),
        "target_current_cap_count": int(
            control_rows["target_current_cap_active"].sum()
        ),
        "mean_ann_inference_time_ms": float(
            control_rows["ann_inference_time_s"].mean() * 1000.0
        ),
        "maximum_ann_inference_time_ms": float(
            control_rows["ann_inference_time_s"].max() * 1000.0
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
