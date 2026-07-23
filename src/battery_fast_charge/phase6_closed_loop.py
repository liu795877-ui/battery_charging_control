"""不带安全过滤器的 Phase 6 DNN 闭环与论文口径对照指标。"""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter_ns
from typing import Any

import numpy as np
import pandas as pd

from .ann_model import TinyANN
from .closed_loop import (
    Chen2020DFNPlant,
    _correct_reduced_state_from_dfn,
    initial_reduced_state,
)
from .mpc import ReducedBatteryModel, ReducedState
from .phase3_config import PhaseThreeConfig
from .phase5a_config import PhaseFiveAConfig
from .phase6_config import PhaseSixConfig
from .robustness import (
    _estimated_state,
    generate_reduced_stress_scenarios,
    perturb_identified_parameters,
)


def pure_dnn_features(model: ReducedBatteryModel, state: ReducedState) -> np.ndarray:
    """按 Phase 6 五维状态顺序构造显式 DNN 输入。"""
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


def _raw_dnn_current(
    ann: TinyANN, model: ReducedBatteryModel, state: ReducedState
) -> tuple[float, float]:
    """返回未经电流裁剪或安全过滤的网络输出及推理时间。"""
    start = perf_counter_ns()
    current = float(ann.predict_unclipped(pure_dnn_features(model, state)))
    elapsed = (perf_counter_ns() - start) * 1.0e-9
    return current, elapsed


def _initial_record(model: ReducedBatteryModel, state: ReducedState, source: str) -> dict[str, Any]:
    return {
        "time_s": 0.0,
        "charge_current_a": 0.0,
        "soc": state.soc,
        "terminal_voltage_v": model.ocv(state.soc),
        "average_temperature_c": model.average_temperature(state),
        "dnn_inference_time_s": 0.0,
        "source": source,
    }


def simulate_pure_dnn_reduced_closed_loop(
    ann: TinyANN,
    model: ReducedBatteryModel,
    phase3: PhaseThreeConfig,
    maximum_simulation_time_s: float,
) -> pd.DataFrame:
    """在名义降阶对象上直接施加裸 DNN 输出。"""
    state = initial_reduced_state(phase3)
    records = [_initial_record(model, state, "reduced_model_pure_dnn")]
    steps = int(np.ceil(maximum_simulation_time_s / phase3.control.control_interval_s))
    for step_index in range(1, steps + 1):
        current, inference_time = _raw_dnn_current(ann, model, state)
        state, output = model.step(state, current)
        records.append(
            {
                "time_s": step_index * phase3.control.control_interval_s,
                "charge_current_a": current,
                "soc": state.soc,
                "terminal_voltage_v": output.terminal_voltage_v,
                "average_temperature_c": output.average_temperature_c,
                "dnn_inference_time_s": inference_time,
                "source": "reduced_model_pure_dnn",
            }
        )
        if state.soc >= phase3.battery.target_soc - phase3.validation.target_soc_tolerance:
            break
        if not np.isfinite(current) or abs(current) > 50.0:
            break
    return pd.DataFrame.from_records(records)


def simulate_pure_dnn_dfn_closed_loop(
    ann: TinyANN,
    model: ReducedBatteryModel,
    phase3: PhaseThreeConfig,
    maximum_simulation_time_s: float,
) -> pd.DataFrame:
    """把裸 DNN 电流直接施加到 Chen2020 DFN，不调用安全过滤器。"""
    plant = Chen2020DFNPlant(phase3)
    state = initial_reduced_state(phase3)
    records = [_initial_record(model, state, "chen2020_dfn_pure_dnn")]
    steps = int(np.ceil(maximum_simulation_time_s / phase3.control.control_interval_s))
    for _ in range(steps):
        current, inference_time = _raw_dnn_current(ann, model, state)
        if not np.isfinite(current) or abs(current) > 50.0:
            break
        predicted_state, _ = model.step(state, current)
        measurement = plant.step(current)
        state = _correct_reduced_state_from_dfn(
            predicted_state, measurement, model, current
        )
        records.append(
            {
                **measurement,
                "charge_current_a": current,
                "dnn_inference_time_s": inference_time,
                "source": "chen2020_dfn_pure_dnn",
            }
        )
        if state.soc >= phase3.battery.target_soc - phase3.validation.target_soc_tolerance:
            break
    return pd.DataFrame.from_records(records)


def pure_dnn_closed_loop_metrics(
    frame: pd.DataFrame,
    phase3: PhaseThreeConfig,
    config: PhaseSixConfig,
) -> dict[str, Any]:
    """报告实际越界量，而不只给出二元安全标签。"""
    constraints = phase3.constraints
    changes = frame["charge_current_a"].diff().abs().fillna(0.0)
    reached = bool(
        frame["soc"].iloc[-1]
        >= phase3.battery.target_soc - phase3.validation.target_soc_tolerance
    )
    voltage_violation = max(0.0, float(frame["terminal_voltage_v"].max()) - constraints.physical_maximum_voltage_v)
    temperature_violation = max(0.0, float(frame["average_temperature_c"].max()) - constraints.physical_maximum_temperature_c)
    upper_current_violation = max(0.0, float(frame["charge_current_a"].max()) - constraints.maximum_current_a)
    lower_current_violation = max(0.0, -float(frame["charge_current_a"].min()))
    current_violation = max(upper_current_violation, lower_current_violation)
    slew_violation = max(0.0, float(changes.max()) - constraints.maximum_current_change_a_per_step)
    criteria = config.success_criteria
    serious = bool(
        voltage_violation > criteria.maximum_voltage_violation_v
        or temperature_violation > criteria.maximum_temperature_violation_c
        or current_violation > criteria.maximum_current_violation_a
        or slew_violation > criteria.maximum_current_change_violation_a
    )
    control_rows = frame.iloc[1:]
    return {
        "reached_target_soc": reached,
        "final_soc": float(frame["soc"].iloc[-1]),
        "charge_time_min": float(frame["time_s"].iloc[-1] / 60.0) if reached else None,
        "maximum_voltage_v": float(frame["terminal_voltage_v"].max()),
        "maximum_temperature_c": float(frame["average_temperature_c"].max()),
        "minimum_current_a": float(frame["charge_current_a"].min()),
        "maximum_current_a": float(frame["charge_current_a"].max()),
        "maximum_current_change_a": float(changes.max()),
        "voltage_violation_v": voltage_violation,
        "temperature_violation_c": temperature_violation,
        "current_violation_a": current_violation,
        "current_change_violation_a": slew_violation,
        "serious_physical_violation": serious,
        "mean_dnn_inference_time_ms": float(control_rows["dnn_inference_time_s"].mean() * 1000.0),
        "maximum_dnn_inference_time_ms": float(control_rows["dnn_inference_time_s"].max() * 1000.0),
        "success": bool(reached and not serious),
    }


def compare_with_teacher(
    dnn_frame: pd.DataFrame,
    dnn_metrics: dict[str, Any],
    teacher_frame: pd.DataFrame,
    teacher_metrics: dict[str, Any],
    config: PhaseSixConfig,
) -> dict[str, Any]:
    """按共同时间网格计算论文口径电流 NRMSE、时间差和推理加速。"""
    end_time = min(float(dnn_frame["time_s"].max()), float(teacher_frame["time_s"].max()))
    comparison_rows = dnn_frame[(dnn_frame["time_s"] > 0.0) & (dnn_frame["time_s"] <= end_time)]
    teacher_current = np.interp(
        comparison_rows["time_s"].to_numpy(dtype=float),
        teacher_frame["time_s"].to_numpy(dtype=float),
        teacher_frame["charge_current_a"].to_numpy(dtype=float),
    )
    dnn_current = comparison_rows["charge_current_a"].to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean((dnn_current - teacher_current) ** 2)))
    nrmse = rmse / config.nominal_validation.current_nrmse_normalization_a
    teacher_time = float(teacher_metrics["charge_time_min"])
    dnn_time = float(dnn_metrics["charge_time_min"]) if dnn_metrics["charge_time_min"] is not None else np.inf
    time_gap = abs(dnn_time - teacher_time) / teacher_time
    speedup = float(teacher_metrics["mean_mpc_solve_time_ms"]) / dnn_metrics["mean_dnn_inference_time_ms"]
    return {
        "comparison_step_count": int(len(comparison_rows)),
        "current_rmse_a": rmse,
        "current_nrmse": nrmse,
        "charge_time_gap_fraction": float(time_gap),
        "inference_speedup_over_mpc": float(speedup),
    }


def temperature_anchor_config(
    phase3: PhaseThreeConfig, temperature_c: float, maximum_time_s: float
) -> PhaseThreeConfig:
    """创建不修改原始配置的独立温度锚点。"""
    battery = replace(
        phase3.battery,
        initial_temperature_c=temperature_c,
        ambient_temperature_c=temperature_c,
    )
    return replace(
        phase3,
        battery=battery,
        control=replace(phase3.control, maximum_simulation_time_s=maximum_time_s),
    )


def run_pure_dnn_phase5a_stress(
    ann: TinyANN,
    nominal_parameters: dict[str, Any],
    ocv_function,
    phase3: PhaseThreeConfig,
    phase5a: PhaseFiveAConfig,
    phase6: PhaseSixConfig,
) -> pd.DataFrame:
    """把 Phase 5A 的同一组扰动施加给无安全过滤器的 Phase 6 DNN。"""
    summaries: list[dict[str, Any]] = []
    stress = phase5a.reduced_stress_test
    for scenario_index, scenario in generate_reduced_stress_scenarios(phase5a).iterrows():
        controller_battery = replace(
            phase3.battery,
            initial_soc=float(scenario["initial_soc"]),
            initial_temperature_c=float(scenario["ambient_temperature_c"]),
            ambient_temperature_c=float(scenario["ambient_temperature_c"]),
        )
        controller_config = replace(
            phase3,
            battery=controller_battery,
            control=replace(phase3.control, maximum_simulation_time_s=stress.maximum_simulation_time_s),
        )
        true_config = replace(
            controller_config,
            battery=replace(
                controller_battery,
                nominal_capacity_ah=phase3.battery.nominal_capacity_ah * float(scenario["capacity_multiplier"]),
            ),
        )
        controller_model = ReducedBatteryModel(controller_config, ocv_function, nominal_parameters)
        true_model = ReducedBatteryModel(
            true_config,
            ocv_function,
            perturb_identified_parameters(nominal_parameters, scenario),
        )
        true_state = initial_reduced_state(true_config)
        random = np.random.default_rng(phase5a.random_seed + 1009 * scenario_index)
        noise = {
            "soc": 0.0,
            "temperature_c": 0.0,
            "polarization_fast_v": 0.0,
            "polarization_slow_v": 0.0,
        }
        sigmas = {
            "soc": stress.soc_noise_standard_deviation,
            "temperature_c": stress.temperature_noise_standard_deviation_c,
            "polarization_fast_v": stress.polarization_noise_standard_deviation_v,
            "polarization_slow_v": stress.polarization_noise_standard_deviation_v,
        }
        rho = 0.95
        innovation = np.sqrt(1.0 - rho**2)
        currents = [0.0]
        voltages = [true_model.ocv(true_state.soc)]
        temperatures = [true_model.average_temperature(true_state)]
        controller_complete = False
        maximum_steps = int(np.ceil(stress.maximum_simulation_time_s / phase3.control.control_interval_s))
        for _ in range(maximum_steps):
            for key, sigma in sigmas.items():
                noise[key] = rho * noise[key] + innovation * sigma * float(scenario["noise_scale"]) * random.normal()
            estimate = _estimated_state(true_state, controller_model, scenario, noise)
            if estimate.soc >= phase3.battery.target_soc - phase3.validation.target_soc_tolerance:
                controller_complete = True
                break
            current, _ = _raw_dnn_current(ann, controller_model, estimate)
            if not np.isfinite(current) or abs(current) > 50.0:
                break
            true_state, output = true_model.step(true_state, current)
            currents.append(current)
            voltages.append(output.terminal_voltage_v)
            temperatures.append(output.average_temperature_c)
            if true_state.soc >= 0.815:
                break
        current_array = np.asarray(currents)
        change = np.abs(np.diff(current_array, prepend=current_array[0]))
        terminal_error = float(true_state.soc - phase3.battery.target_soc)
        criteria = phase6.success_criteria
        voltage_violation = max(0.0, max(voltages) - phase3.constraints.physical_maximum_voltage_v)
        temperature_violation = max(0.0, max(temperatures) - phase3.constraints.physical_maximum_temperature_c)
        current_violation = max(0.0, max(current_array) - phase3.constraints.maximum_current_a, -min(current_array))
        slew_violation = max(0.0, max(change) - phase3.constraints.maximum_current_change_a_per_step)
        serious = bool(
            voltage_violation > criteria.maximum_voltage_violation_v
            or temperature_violation > criteria.maximum_temperature_violation_c
            or current_violation > criteria.maximum_current_violation_a
            or slew_violation > criteria.maximum_current_change_violation_a
        )
        summaries.append(
            {
                **scenario.to_dict(),
                "controller_declared_complete": controller_complete,
                "completion_success": bool(controller_complete and abs(terminal_error) <= stress.terminal_true_soc_tolerance),
                "final_true_soc": true_state.soc,
                "terminal_true_soc_error": terminal_error,
                "charge_time_min": (len(currents) - 1) * phase3.control.control_interval_s / 60.0,
                "maximum_voltage_v": max(voltages),
                "maximum_temperature_c": max(temperatures),
                "minimum_current_a": float(min(current_array)),
                "maximum_current_a": float(max(current_array)),
                "maximum_current_change_a": float(max(change)),
                "serious_physical_violation": serious,
            }
        )
    return pd.DataFrame.from_records(summaries)
