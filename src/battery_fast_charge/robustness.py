"""阶段5A降阶有界压力测试与Chen2020 DFN温度锚点。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .ann_closed_loop import (
    _ann_decision,
    ann_closed_loop_metrics,
    simulate_ann_dfn_closed_loop,
)
from .ann_model import TinyANN
from .closed_loop import _cap_current_at_target, initial_reduced_state
from .mpc import ReducedBatteryModel, ReducedState
from .phase3_config import PhaseThreeConfig
from .phase4_config import PhaseFourAConfig
from .phase5a_config import PhaseFiveAConfig


SCENARIO_COLUMNS = [
    "initial_soc",
    "ambient_temperature_c",
    "capacity_multiplier",
    "resistance_multiplier",
    "time_constant_multiplier",
    "heat_capacity_multiplier",
    "thermal_resistance_multiplier",
    "heat_gain_multiplier",
    "soc_bias",
    "temperature_bias_c",
    "polarization_fast_bias_v",
    "polarization_slow_bias_v",
]


def _scale_unit(value: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    """把0到1的拉丁超立方坐标映射到物理范围。"""
    low, high = limits
    return low + np.asarray(value) * (high - low)


def generate_reduced_stress_scenarios(config: PhaseFiveAConfig) -> pd.DataFrame:
    """生成名义、四个定向角点和拉丁超立方随机压力场景。"""
    stress = config.reduced_stress_test
    nominal = {
        "scenario_id": "nominal",
        "scenario_kind": "nominal",
        "initial_soc": 0.10,
        "ambient_temperature_c": 25.0,
        "capacity_multiplier": 1.0,
        "resistance_multiplier": 1.0,
        "time_constant_multiplier": 1.0,
        "heat_capacity_multiplier": 1.0,
        "thermal_resistance_multiplier": 1.0,
        "heat_gain_multiplier": 1.0,
        "soc_bias": 0.0,
        "temperature_bias_c": 0.0,
        "polarization_fast_bias_v": 0.0,
        "polarization_slow_bias_v": 0.0,
        "noise_scale": 0.0,
    }
    corners = [
        {
            **nominal,
            "scenario_id": "corner_hot_resistive_optimistic",
            "scenario_kind": "directed_corner",
            "ambient_temperature_c": stress.ambient_temperature_c_range[1],
            "capacity_multiplier": stress.capacity_multiplier_range[0],
            "resistance_multiplier": stress.resistance_multiplier_range[1],
            "heat_capacity_multiplier": stress.heat_capacity_multiplier_range[0],
            "thermal_resistance_multiplier": stress.thermal_resistance_multiplier_range[1],
            "heat_gain_multiplier": stress.heat_gain_multiplier_range[1],
            "soc_bias": stress.soc_bias_range[1],
            "temperature_bias_c": stress.temperature_bias_c_range[0],
            "polarization_fast_bias_v": stress.polarization_bias_v_range[0],
            "polarization_slow_bias_v": stress.polarization_bias_v_range[0],
            "noise_scale": 1.0,
        },
        {
            **nominal,
            "scenario_id": "corner_cold_resistive",
            "scenario_kind": "directed_corner",
            "ambient_temperature_c": stress.ambient_temperature_c_range[0],
            "resistance_multiplier": stress.resistance_multiplier_range[1],
            "time_constant_multiplier": stress.time_constant_multiplier_range[1],
            "polarization_fast_bias_v": stress.polarization_bias_v_range[0],
            "polarization_slow_bias_v": stress.polarization_bias_v_range[0],
            "noise_scale": 1.0,
        },
        {
            **nominal,
            "scenario_id": "corner_observer_optimistic",
            "scenario_kind": "directed_corner",
            "initial_soc": stress.initial_soc_range[1],
            "soc_bias": stress.soc_bias_range[1],
            "temperature_bias_c": stress.temperature_bias_c_range[0],
            "polarization_fast_bias_v": stress.polarization_bias_v_range[0],
            "polarization_slow_bias_v": stress.polarization_bias_v_range[0],
            "noise_scale": 1.0,
        },
        {
            **nominal,
            "scenario_id": "corner_observer_pessimistic",
            "scenario_kind": "directed_corner",
            "initial_soc": stress.initial_soc_range[1],
            "soc_bias": stress.soc_bias_range[0],
            "temperature_bias_c": stress.temperature_bias_c_range[1],
            "polarization_fast_bias_v": stress.polarization_bias_v_range[1],
            "polarization_slow_bias_v": stress.polarization_bias_v_range[1],
            "noise_scale": 1.0,
        },
    ]
    sampler = qmc.LatinHypercube(d=len(SCENARIO_COLUMNS), seed=config.random_seed)
    unit = sampler.random(stress.random_scenario_count)
    ranges = [
        stress.initial_soc_range,
        stress.ambient_temperature_c_range,
        stress.capacity_multiplier_range,
        stress.resistance_multiplier_range,
        stress.time_constant_multiplier_range,
        stress.heat_capacity_multiplier_range,
        stress.thermal_resistance_multiplier_range,
        stress.heat_gain_multiplier_range,
        stress.soc_bias_range,
        stress.temperature_bias_c_range,
        stress.polarization_bias_v_range,
        stress.polarization_bias_v_range,
    ]
    scaled = np.column_stack(
        [_scale_unit(unit[:, index], limits) for index, limits in enumerate(ranges)]
    )
    random_records = []
    for index, values in enumerate(scaled):
        record = {
            "scenario_id": f"lhs_{index:03d}",
            "scenario_kind": "latin_hypercube",
            "noise_scale": 1.0,
        }
        record.update(dict(zip(SCENARIO_COLUMNS, values)))
        random_records.append(record)
    return pd.DataFrame.from_records([nominal, *corners, *random_records])


def perturb_identified_parameters(
    nominal: dict[str, Any], scenario: pd.Series
) -> dict[str, Any]:
    """按场景倍率构造真实降阶对象，控制器仍保留名义参数。"""
    values = deepcopy(nominal)
    electrical = values["electrical_2rc"]
    for key in ("r0_ohm", "r1_ohm", "r2_ohm"):
        electrical[key] *= float(scenario["resistance_multiplier"])
    for key in ("tau1_s", "tau2_s"):
        electrical[key] *= float(scenario["time_constant_multiplier"])
    # 电容只用于参数审计，动态实现直接使用R与tau；保持C=tau/R一致。
    electrical["c1_f"] = electrical["tau1_s"] / electrical["r1_ohm"]
    electrical["c2_f"] = electrical["tau2_s"] / electrical["r2_ohm"]
    thermal = values["thermal_two_node"]
    thermal["total_heat_capacity_j_per_k"] *= float(
        scenario["heat_capacity_multiplier"]
    )
    thermal["r_core_surface_k_per_w"] *= float(
        scenario["thermal_resistance_multiplier"]
    )
    thermal["r_surface_ambient_k_per_w"] *= float(
        scenario["thermal_resistance_multiplier"]
    )
    thermal["heat_gain"] *= float(scenario["heat_gain_multiplier"])
    return values


def _estimated_state(
    true_state: ReducedState,
    controller_model: ReducedBatteryModel,
    scenario: pd.Series,
    correlated_noise: dict[str, float],
) -> ReducedState:
    """把有偏且带相关噪声的估计状态提供给ANN和安全过滤器。"""
    temperature_error = float(scenario["temperature_bias_c"]) + correlated_noise[
        "temperature_c"
    ]
    return ReducedState(
        soc=float(
            np.clip(
                true_state.soc
                + float(scenario["soc_bias"])
                + correlated_noise["soc"],
                0.10,
                0.80,
            )
        ),
        polarization_fast_v=float(
            true_state.polarization_fast_v
            + float(scenario["polarization_fast_bias_v"])
            + correlated_noise["polarization_fast_v"]
        ),
        polarization_slow_v=float(
            true_state.polarization_slow_v
            + float(scenario["polarization_slow_bias_v"])
            + correlated_noise["polarization_slow_v"]
        ),
        core_temperature_c=true_state.core_temperature_c + temperature_error,
        surface_temperature_c=true_state.surface_temperature_c + temperature_error,
        # 上一时刻电流是控制器自身已发出的指令，不是带噪声的传感器量。
        previous_current_a=true_state.previous_current_a,
    )


def simulate_reduced_stress_scenario(
    ann: TinyANN,
    nominal_parameters: dict[str, Any],
    ocv_function,
    phase3: PhaseThreeConfig,
    phase4: PhaseFourAConfig,
    config: PhaseFiveAConfig,
    scenario: pd.Series,
    scenario_index: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """名义控制器闭环驱动一个参数扰动的真实降阶对象。"""
    stress = config.reduced_stress_test
    controller_battery = replace(
        phase3.battery,
        initial_soc=float(scenario["initial_soc"]),
        initial_temperature_c=float(scenario["ambient_temperature_c"]),
        ambient_temperature_c=float(scenario["ambient_temperature_c"]),
    )
    controller_config = replace(
        phase3,
        battery=controller_battery,
        control=replace(
            phase3.control,
            maximum_simulation_time_s=stress.maximum_simulation_time_s,
        ),
    )
    true_battery = replace(
        controller_battery,
        nominal_capacity_ah=(
            phase3.battery.nominal_capacity_ah
            * float(scenario["capacity_multiplier"])
        ),
    )
    true_config = replace(controller_config, battery=true_battery)
    controller_model = ReducedBatteryModel(
        controller_config, ocv_function, nominal_parameters
    )
    true_model = ReducedBatteryModel(
        true_config,
        ocv_function,
        perturb_identified_parameters(nominal_parameters, scenario),
    )
    true_state = initial_reduced_state(true_config)
    random = np.random.default_rng(config.random_seed + 1009 * scenario_index)
    correlated_noise = {
        "soc": 0.0,
        "temperature_c": 0.0,
        "polarization_fast_v": 0.0,
        "polarization_slow_v": 0.0,
    }
    standard_deviations = {
        "soc": stress.soc_noise_standard_deviation,
        "temperature_c": stress.temperature_noise_standard_deviation_c,
        "polarization_fast_v": stress.polarization_noise_standard_deviation_v,
        "polarization_slow_v": stress.polarization_noise_standard_deviation_v,
    }
    noise_scale = float(scenario["noise_scale"])
    rho = 0.95
    innovation_scale = np.sqrt(1.0 - rho**2)
    records = [
        {
            "scenario_id": scenario["scenario_id"],
            "time_s": 0.0,
            "true_soc": true_state.soc,
            "estimated_soc": true_state.soc,
            "charge_current_a": 0.0,
            "terminal_voltage_v": true_model.ocv(true_state.soc),
            "true_average_temperature_c": true_model.average_temperature(true_state),
            "estimated_average_temperature_c": controller_model.average_temperature(
                true_state
            ),
            "ann_requested_current_a": np.nan,
            "safety_filtered_current_a": 0.0,
            "filter_correction_a": 0.0,
        }
    ]
    maximum_steps = int(
        np.ceil(stress.maximum_simulation_time_s / phase3.control.control_interval_s)
    )
    controller_complete = False
    for step_index in range(1, maximum_steps + 1):
        for key, sigma in standard_deviations.items():
            correlated_noise[key] = (
                rho * correlated_noise[key]
                + innovation_scale * sigma * noise_scale * random.normal()
            )
        estimate = _estimated_state(
            true_state, controller_model, scenario, correlated_noise
        )
        # 与既有闭环验证一致：控制器在观测 SOC 首次进入目标容差时结束充电。
        # 不额外记录停机后的零电流点，避免把任务结束误判成正常控制步的斜率越界。
        if estimate.soc >= phase3.battery.target_soc - phase3.validation.target_soc_tolerance:
            controller_complete = True
            break
        decision = _ann_decision(ann, controller_model, estimate, controller_config)
        applied, _ = _cap_current_at_target(
            decision["filtered_current_a"], estimate.soc, controller_config
        )
        true_state, output = true_model.step(true_state, applied)
        correction = abs(decision["requested_current_a"] - decision["filtered_current_a"])
        records.append(
            {
                "scenario_id": scenario["scenario_id"],
                "time_s": step_index * phase3.control.control_interval_s,
                "true_soc": true_state.soc,
                "estimated_soc": estimate.soc,
                "charge_current_a": applied,
                "terminal_voltage_v": output.terminal_voltage_v,
                "true_average_temperature_c": output.average_temperature_c,
                "estimated_average_temperature_c": controller_model.average_temperature(
                    estimate
                ),
                "ann_requested_current_a": decision["requested_current_a"],
                "safety_filtered_current_a": decision["filtered_current_a"],
                "filter_correction_a": correction,
            }
        )
        if true_state.soc >= 0.815:
            break
    frame = pd.DataFrame.from_records(records)
    current_change = frame["charge_current_a"].diff().abs().fillna(0.0)
    tolerance = phase3.validation.physical_constraint_tolerance
    terminal_error = float(frame["true_soc"].iloc[-1] - phase3.battery.target_soc)
    true_target_acceptable = abs(terminal_error) <= stress.terminal_true_soc_tolerance
    physical_safe = bool(
        frame["terminal_voltage_v"].max()
        <= phase3.constraints.physical_maximum_voltage_v + tolerance
        and frame["true_average_temperature_c"].max()
        <= phase3.constraints.physical_maximum_temperature_c + tolerance
        and frame["charge_current_a"].max()
        <= phase3.constraints.maximum_current_a + tolerance
        and current_change.max()
        <= phase3.constraints.maximum_current_change_a_per_step + tolerance
    )
    summary = scenario.to_dict()
    summary.update(
        {
            "controller_declared_complete": controller_complete,
            "true_target_acceptable": true_target_acceptable,
            "completion_success": bool(controller_complete and true_target_acceptable),
            "physical_safe": physical_safe,
            "charge_time_min": float(frame["time_s"].iloc[-1] / 60.0),
            "final_true_soc": float(frame["true_soc"].iloc[-1]),
            "terminal_true_soc_error": terminal_error,
            "maximum_voltage_v": float(frame["terminal_voltage_v"].max()),
            "maximum_temperature_c": float(
                frame["true_average_temperature_c"].max()
            ),
            "maximum_current_a": float(frame["charge_current_a"].max()),
            "maximum_current_change_a": float(current_change.max()),
            "material_intervention_fraction": float(
                (frame["filter_correction_a"].iloc[1:] > 0.1).mean()
            ),
            "mean_filter_correction_a": float(
                frame["filter_correction_a"].iloc[1:].mean()
            ),
        }
    )
    return summary, frame


def run_reduced_stress_test(
    ann: TinyANN,
    nominal_parameters: dict[str, Any],
    ocv_function,
    phase3: PhaseThreeConfig,
    phase4: PhaseFourAConfig,
    config: PhaseFiveAConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """运行全部有界降阶场景并只保留代表性最坏轨迹。"""
    scenarios = generate_reduced_stress_scenarios(config)
    summaries = []
    trajectories: dict[str, pd.DataFrame] = {}
    for index, scenario in scenarios.iterrows():
        summary, frame = simulate_reduced_stress_scenario(
            ann,
            nominal_parameters,
            ocv_function,
            phase3,
            phase4,
            config,
            scenario,
            index,
        )
        summaries.append(summary)
        trajectories[str(scenario["scenario_id"])] = frame
    summary_frame = pd.DataFrame.from_records(summaries)
    selected_ids = {"nominal"}
    for column, ascending in (
        ("charge_time_min", False),
        ("maximum_voltage_v", False),
        ("maximum_temperature_c", False),
        ("material_intervention_fraction", False),
        ("terminal_true_soc_error", True),
    ):
        selected_ids.add(
            str(summary_frame.sort_values(column, ascending=ascending).iloc[0]["scenario_id"])
        )
    worst = pd.concat(
        [trajectories[scenario_id] for scenario_id in sorted(selected_ids)],
        ignore_index=True,
    )
    return summary_frame, worst


def run_dfn_temperature_anchors(
    ann: TinyANN,
    nominal_parameters: dict[str, Any],
    ocv_function,
    phase3: PhaseThreeConfig,
    phase4: PhaseFourAConfig,
    config: PhaseFiveAConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在15、25、30 ℃陈列独立DFN闭环锚点。"""
    summaries = []
    trajectories = []
    anchor = config.dfn_temperature_anchors
    threshold = 0.1
    for temperature in anchor.temperatures_c:
        battery = replace(
            phase3.battery,
            initial_soc=anchor.initial_soc,
            initial_temperature_c=temperature,
            ambient_temperature_c=temperature,
        )
        phase3_anchor = replace(
            phase3,
            battery=battery,
            control=replace(
                phase3.control,
                maximum_simulation_time_s=anchor.maximum_simulation_time_s,
            ),
        )
        model = ReducedBatteryModel(
            phase3_anchor, ocv_function, nominal_parameters
        )
        frame = simulate_ann_dfn_closed_loop(
            ann, model, phase3_anchor, phase4
        )
        frame["anchor_temperature_c"] = temperature
        metrics = ann_closed_loop_metrics(frame, phase3_anchor, threshold)
        metrics["anchor_temperature_c"] = temperature
        summaries.append(metrics)
        trajectories.append(frame)
    return pd.DataFrame.from_records(summaries), pd.concat(
        trajectories, ignore_index=True
    )
