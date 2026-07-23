"""Phase 2R-B：MPC 控制律单值性与输入状态充分性审计。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .identification import build_ocv_function
from .mpc import ConstrainedMPC, ReducedBatteryModel, ReducedState
from .phase2r_config import PhaseTwoRConfig
from .phase3_config import load_phase_three_config


def _row_to_state(row: pd.Series) -> ReducedState:
    return ReducedState(
        soc=float(row["state_soc"]),
        polarization_fast_v=float(row["state_polarization_fast_v"]),
        polarization_slow_v=float(row["state_polarization_slow_v"]),
        core_temperature_c=float(row["state_core_temperature_c"]),
        surface_temperature_c=float(row["state_surface_temperature_c"]),
        previous_current_a=float(row["state_previous_current_a"]),
    )


def replay_teacher_with_control_memory(
    dataset: pd.DataFrame,
    config: PhaseTwoRConfig,
    root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """严格重放已接受轨迹，补记上一最优序列摘要和当前模式。"""
    phase3 = load_phase_three_config(root / config.sources.phase3_config)
    parameters = json.loads((root / config.sources.fixed_parameters).read_text(encoding="utf-8"))
    ocv = build_ocv_function(pd.read_csv(root / config.sources.fixed_ocv_curve))
    model = ReducedBatteryModel(phase3, ocv, parameters)
    records: list[dict[str, Any]] = []
    maximum_difference = 0.0
    mismatch_count = 0

    for trajectory_id, trajectory in dataset.groupby("trajectory_id", sort=True):
        trajectory = trajectory.sort_values("step_index")
        controller = ConstrainedMPC(model, phase3)
        previous_plan: np.ndarray | None = None
        for _, source in trajectory.iterrows():
            # 每一步使用冻结 CSV 中的原始教师状态，避免把前一步数值重放差异累积到后续状态；
            # 控制器实例仍按轨迹顺序保留 warm-start/上一最优序列记忆。
            state = _row_to_state(source)
            if previous_plan is None:
                prior_first = prior_mean = prior_last = state.previous_current_a
            else:
                prior_first = float(previous_plan[0])
                prior_mean = float(np.mean(previous_plan))
                prior_last = float(previous_plan[-1])
            result = controller.solve(state)
            difference = abs(float(result.current_a) - float(source["teacher_current_a"]))
            maximum_difference = max(maximum_difference, difference)
            mismatch_count += int(difference > config.state_audit.replay_current_tolerance_a)
            plan = controller.last_optimal_block_currents_a
            if plan is None:
                raise RuntimeError("MPC 重放没有产生可审计控制块。")
            voltage_active = bool(result.predicted_maximum_voltage_v >= phase3.constraints.mpc_maximum_voltage_v - 0.01)
            temperature_active = bool(result.predicted_maximum_temperature_c >= phase3.constraints.mpc_maximum_temperature_c - 0.10)
            slew_active = bool(abs(result.current_a - state.previous_current_a) >= phase3.constraints.maximum_current_change_a_per_step - 0.05)
            if result.used_fallback:
                mode = "fallback"
            elif voltage_active and temperature_active:
                mode = "voltage_temperature"
            elif voltage_active:
                mode = "voltage"
            elif temperature_active:
                mode = "temperature"
            elif slew_active:
                mode = "slew"
            else:
                mode = "interior"
            records.append(
                {
                    **source.to_dict(),
                    "replayed_teacher_current_a": float(result.current_a),
                    "replay_absolute_difference_a": difference,
                    "control_block_phase": int(source["step_index"]) % phase3.control.control_block_steps,
                    "mpc_mode": mode,
                    "mode_voltage_active": float(voltage_active),
                    "mode_temperature_active": float(temperature_active),
                    "mode_slew_active": float(slew_active),
                    "previous_plan_first_a": prior_first,
                    "previous_plan_mean_a": prior_mean,
                    "previous_plan_last_a": prior_last,
                    "current_plan_mean_a": float(np.mean(plan)),
                    "current_plan_last_a": float(plan[-1]),
                    "predicted_terminal_soc": result.predicted_terminal_soc,
                    "target_cap_active": float(result.predicted_terminal_soc >= phase3.battery.target_soc - 0.002),
                    "r0_ohm": parameters["electrical_2rc"]["r0_ohm"],
                    "tau1_s": parameters["electrical_2rc"]["tau1_s"],
                    "tau2_s": parameters["electrical_2rc"]["tau2_s"],
                }
            )
            previous_plan = plan
    augmented = pd.DataFrame.from_records(records)
    metrics = summarize_replay_differences(augmented, config)
    return augmented, metrics


def summarize_replay_differences(
    augmented: pd.DataFrame,
    config: PhaseTwoRConfig,
) -> dict[str, Any]:
    """同时保留严格重放结论与统计记忆摘要可用性结论。"""
    differences = augmented["replay_absolute_difference_a"].to_numpy(dtype=float)
    strict_mismatch = int((differences > config.state_audit.replay_current_tolerance_a).sum())
    large_fraction = float(np.mean(differences > config.state_audit.large_replay_difference_a))
    p95 = float(np.quantile(differences, 0.95))
    exact_success = strict_mismatch == 0
    memory_summary_usable = bool(
        p95 <= config.state_audit.replay_p95_tolerance_a
        and large_fraction <= config.state_audit.maximum_large_replay_difference_fraction
    )
    return {
        "sample_count": int(len(augmented)),
        "trajectory_count": int(augmented["trajectory_id"].nunique()),
        "mean_replay_current_difference_a": float(np.mean(differences)),
        "median_replay_current_difference_a": float(np.median(differences)),
        "p95_replay_current_difference_a": p95,
        "p99_replay_current_difference_a": float(np.quantile(differences, 0.99)),
        "maximum_replay_current_difference_a": float(np.max(differences)),
        "replay_mismatch_count": strict_mismatch,
        "tolerance_a": config.state_audit.replay_current_tolerance_a,
        "large_difference_threshold_a": config.state_audit.large_replay_difference_a,
        "large_difference_count": int((differences > config.state_audit.large_replay_difference_a).sum()),
        "large_difference_fraction": large_fraction,
        "exact_success": exact_success,
        "memory_summary_usable": memory_summary_usable,
        "success": exact_success,
    }


def _local_variance_metrics(
    frame: pd.DataFrame,
    features: list[str],
    neighbor_count: int,
) -> dict[str, float]:
    values = frame[features].to_numpy(dtype=float)
    if any(float(np.std(values[:, index])) <= 1.0e-12 for index in range(values.shape[1])):
        variable = np.std(values, axis=0) > 1.0e-12
        values = values[:, variable]
    scaled = StandardScaler().fit_transform(values)
    neighbors = min(neighbor_count + 1, len(frame))
    distances, indices = NearestNeighbors(n_neighbors=neighbors).fit(scaled).kneighbors(scaled)
    neighbor_indices = indices[:, 1:]
    target = frame["teacher_current_a"].to_numpy(dtype=float)
    labels = target[neighbor_indices]
    local_variance = np.var(labels, axis=1, ddof=1)
    local_standard_deviation = np.sqrt(local_variance)
    prediction = np.mean(labels, axis=1)
    nearest_difference = np.abs(target - labels[:, 0])
    error = prediction - target
    return {
        "feature_count": len(features),
        "effective_variable_feature_count": int(values.shape[1]),
        "mean_conditional_variance_a2": float(np.mean(local_variance)),
        "median_conditional_variance_a2": float(np.median(local_variance)),
        "p95_conditional_variance_a2": float(np.quantile(local_variance, 0.95)),
        "mean_local_standard_deviation_a": float(np.mean(local_standard_deviation)),
        "p95_local_standard_deviation_a": float(np.quantile(local_standard_deviation, 0.95)),
        "nearest_neighbor_label_difference_mean_a": float(np.mean(nearest_difference)),
        "nearest_neighbor_label_difference_p95_a": float(np.quantile(nearest_difference, 0.95)),
        "knn_leave_one_out_rmse_a": float(np.sqrt(np.mean(error**2))),
        "knn_leave_one_out_nrmse_percent_of_10a": float(10.0 * np.sqrt(np.mean(error**2))),
        "mean_neighbor_distance": float(np.mean(distances[:, 1:])),
    }


def run_state_sufficiency_audit(
    augmented: pd.DataFrame,
    config: PhaseTwoRConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """对嵌套输入集估计局部条件方差，并标记不可识别候选。"""
    base = [
        "state_soc",
        "state_polarization_fast_v",
        "state_polarization_slow_v",
        "state_average_temperature_c",
    ]
    feature_sets: list[tuple[str, list[str]]] = [
        ("electrothermal_4", base),
        ("current_dnn_5_plus_previous_current", [*base, "state_previous_current_a"]),
        (
            "plus_thermal_split",
            [
                "state_soc", "state_polarization_fast_v", "state_polarization_slow_v",
                "state_core_temperature_c", "state_surface_temperature_c", "state_previous_current_a",
            ],
        ),
        ("plus_control_block_phase", [*base, "state_previous_current_a", "control_block_phase"]),
        (
            "plus_mpc_mode",
            [*base, "state_previous_current_a", "mode_voltage_active", "mode_temperature_active", "mode_slew_active"],
        ),
        (
            "plus_previous_plan_summary",
            [*base, "state_previous_current_a", "previous_plan_first_a", "previous_plan_mean_a", "previous_plan_last_a"],
        ),
        ("plus_target_cap_state", [*base, "state_previous_current_a", "target_cap_active"]),
        (
            "all_online_or_diagnostic_variables",
            [
                "state_soc", "state_polarization_fast_v", "state_polarization_slow_v",
                "state_core_temperature_c", "state_surface_temperature_c", "state_previous_current_a",
                "control_block_phase", "mode_voltage_active", "mode_temperature_active", "mode_slew_active",
                "previous_plan_first_a", "previous_plan_mean_a", "previous_plan_last_a", "target_cap_active",
            ],
        ),
    ]
    records: list[dict[str, Any]] = []
    for name, features in feature_sets:
        records.append(
            {
                "feature_set": name,
                "features": ",".join(features),
                **_local_variance_metrics(augmented, features, config.state_audit.neighbor_count),
            }
        )
    metrics = pd.DataFrame.from_records(records)
    baseline_variance = float(metrics.loc[metrics["feature_set"] == "electrothermal_4", "mean_conditional_variance_a2"].iloc[0])
    current_variance = float(metrics.loc[metrics["feature_set"] == "current_dnn_5_plus_previous_current", "mean_conditional_variance_a2"].iloc[0])
    metrics["variance_reduction_from_electrothermal_4_fraction"] = 1.0 - metrics["mean_conditional_variance_a2"] / baseline_variance
    metrics["variance_reduction_from_current_dnn_5_fraction"] = 1.0 - metrics["mean_conditional_variance_a2"] / current_variance
    metrics["significant_reduction_vs_current_dnn"] = metrics["variance_reduction_from_current_dnn_5_fraction"] >= config.state_audit.significant_variance_reduction_fraction

    availability = pd.DataFrame.from_records(
        [
            {"candidate": "I_k_minus_1", "status": "available", "detail": "state_previous_current_a 已在当前五状态 DNN 中"},
            {"candidate": "control_block_phase", "status": "recoverable_proxy", "detail": "使用 step_index mod control_block_steps；滚动 MPC 每步重算，物理含义需谨慎"},
            {"candidate": "ambient_temperature", "status": "constant_not_identifiable", "detail": "Phase 6R 教师数据全部为 25 ℃"},
            {"candidate": "current_mpc_mode", "status": "diagnostic_only", "detail": "由求解后的活跃约束定义，不能直接作为求解前在线输入"},
            {"candidate": "previous_optimal_sequence_summary", "status": "replayed_approximate", "detail": "记录上一控制块首项/均值/末项；CSV 状态重启在少数切换点不能严格复现"},
            {"candidate": "identifiable_parameters", "status": "constant_not_identifiable", "detail": "Phase 6R 使用同一组固定 2RC/热参数"},
            {"candidate": "target_cap_state", "status": "replayed_diagnostic", "detail": "由预测终端 SOC 是否接近目标构造"},
        ]
    )
    current_row = metrics[metrics["feature_set"] == "current_dnn_5_plus_previous_current"].iloc[0]
    sufficient = bool(
        current_row["mean_local_standard_deviation_a"] <= config.state_audit.sufficient_local_standard_deviation_a
        and current_row["nearest_neighbor_label_difference_p95_a"] <= config.state_audit.sufficient_p95_neighbor_label_difference_a
    )
    summary = {
        "current_dnn_input_locally_sufficient": sufficient,
        "current_dnn_mean_local_standard_deviation_a": float(current_row["mean_local_standard_deviation_a"]),
        "current_dnn_p95_nearest_neighbor_label_difference_a": float(current_row["nearest_neighbor_label_difference_p95_a"]),
        "significant_additions_vs_current_dnn": metrics.loc[metrics["significant_reduction_vs_current_dnn"], "feature_set"].tolist(),
        "thresholds": {
            "mean_local_standard_deviation_a": config.state_audit.sufficient_local_standard_deviation_a,
            "p95_nearest_neighbor_label_difference_a": config.state_audit.sufficient_p95_neighbor_label_difference_a,
            "significant_variance_reduction_fraction": config.state_audit.significant_variance_reduction_fraction,
        },
    }
    return metrics, availability, summary
