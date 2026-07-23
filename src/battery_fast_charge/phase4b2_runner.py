"""组织主动数据聚合、混合教师重标、ANN v2训练和双层闭环验证。"""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .active_learning import (
    FEATURE_COLUMNS,
    active_dataset_metrics,
    combine_candidate_states,
    generate_ann_centered_rollout,
    generate_ann_dfn_rollout,
    generate_active_rollouts,
    label_with_hybrid_teacher,
    prepare_legacy_states,
    sample_active_states,
    sample_dense_on_policy_states,
)
from .ann_closed_loop import (
    ann_closed_loop_metrics,
    simulate_ann_dfn_closed_loop,
    simulate_ann_reduced_closed_loop,
)
from .ann_model import TinyANN, train_tiny_ann
from .identification import build_ocv_function
from .mpc import ReducedBatteryModel
from .phase3_config import load_phase_three_config
from .phase4_config import load_phase_four_a_config
from .phase4b_config import load_phase_four_b_config
from .phase4b2_config import PhaseFourB2Config
from .phase4b2_plotting import (
    plot_active_coverage,
    plot_ann_v2_dfn_comparison,
    plot_offline_active_imitation,
)


def _load_context(config: PhaseFourB2Config, project_root: Path):
    """加载冻结的模型、旧训练协议和已经验收的4B教师。"""
    phase3 = load_phase_three_config(project_root / config.source_phase3_config)
    phase4a = load_phase_four_a_config(project_root / config.source_phase4a_config)
    phase4b = load_phase_four_b_config(project_root / config.source_phase4b_config)
    phase2_validation = json.loads(
        (project_root / phase3.artifacts.validation_metrics).read_text(encoding="utf-8")
    )
    phase4b_metrics = json.loads(
        (project_root / "outputs" / "metrics" / "phase4b_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    if not phase2_validation.get("success", False):
        raise RuntimeError("第二阶段模型验证未通过，禁止主动数据聚合。")
    if not phase4b_metrics.get("ready_for_active_data_aggregation", False):
        raise RuntimeError("阶段4B-1闸门未通过，禁止使用新教师重新标注。")
    parameters = json.loads(
        (project_root / phase3.artifacts.identified_parameters).read_text(
            encoding="utf-8"
        )
    )
    ocv = pd.read_csv(project_root / phase3.artifacts.ocv_curve)
    model = ReducedBatteryModel(phase3, build_ocv_function(ocv), parameters)
    return phase3, phase4a, phase4b, phase4b_metrics, model


def _coverage_metrics(legacy: pd.DataFrame, active: pd.DataFrame) -> dict[str, Any]:
    """用旧训练集标准化距离量化主动状态对覆盖的扩展。"""
    reference = legacy[legacy["split"] == "train"][FEATURE_COLUMNS].to_numpy(float)
    mean = reference.mean(axis=0)
    scale = reference.std(axis=0)
    scale[scale < 1.0e-12] = 1.0
    query = active[FEATURE_COLUMNS].to_numpy(float)
    distances = np.sqrt(
        np.min(
            np.sum(((query[:, None, :] - reference[None, :, :]) / scale) ** 2, axis=2),
            axis=1,
        )
    )
    return {
        "active_state_count": int(len(active)),
        "mean_nearest_legacy_train_distance": float(np.mean(distances)),
        "p95_nearest_legacy_train_distance": float(np.quantile(distances, 0.95)),
        "maximum_nearest_legacy_train_distance": float(np.max(distances)),
        "fraction_beyond_unit_standardized_distance": float(np.mean(distances > 1.0)),
    }


def _mode_test_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    """逐教师模式报告测试误差，避免总体均值掩盖启动段。"""
    test = predictions[predictions["split"] == "test"]
    result = {}
    for mode, group in test.groupby("teacher_control_mode"):
        error = group["ann_current_a"] - group["teacher_current_a"]
        result[str(mode)] = {
            "sample_count": int(len(group)),
            "mae_a": float(error.abs().mean()),
            "rmse_a": float(np.sqrt(np.mean(error**2))),
            "maximum_absolute_error_a": float(error.abs().max()),
        }
    return result


def _comparison_frame(
    old: dict[str, Any], new: dict[str, Any], teacher: dict[str, Any]
) -> pd.DataFrame:
    """汇总新旧ANN和新教师的DFN时间、安全层与极值。"""
    return pd.DataFrame.from_records(
        [
            {
                "controller": "ANN v1 + safety filter",
                "charge_time_min": old["charge_time_min"],
                "material_intervention_fraction": old[
                    "material_safety_filter_intervention_fraction"
                ],
                "mean_filter_correction_a": old["mean_safety_filter_correction_a"],
                "maximum_voltage_v": old["maximum_voltage_v"],
                "maximum_temperature_c": old["maximum_temperature_c"],
                "success": old["success"],
            },
            {
                "controller": "Hybrid thermal-budget teacher",
                "charge_time_min": teacher["charge_time_min"],
                "material_intervention_fraction": 0.0,
                "mean_filter_correction_a": 0.0,
                "maximum_voltage_v": teacher["maximum_voltage_v"],
                "maximum_temperature_c": teacher["maximum_temperature_c"],
                "success": teacher["success"],
            },
            {
                "controller": "ANN v2 + safety filter",
                "charge_time_min": new["charge_time_min"],
                "material_intervention_fraction": new[
                    "material_safety_filter_intervention_fraction"
                ],
                "mean_filter_correction_a": new["mean_safety_filter_correction_a"],
                "maximum_voltage_v": new["maximum_voltage_v"],
                "maximum_temperature_c": new["maximum_temperature_c"],
                "success": new["success"],
            },
        ]
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    """不引入tabulate依赖地生成小型Markdown表格。"""
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for _, row in frame.iterrows():
        values = []
        for header in headers:
            value = row[header]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(path: Path, payload: dict[str, Any], comparison: pd.DataFrame) -> None:
    """生成强调主动数据证据和未解除安全层的中文报告。"""
    data = payload["active_dataset"]
    offline = payload["offline_imitation"]["split_metrics"]["test"]
    dfn = payload["ann_v2_dfn_closed_loop"]
    lines = [
        "# 第四阶段 B-2 报告：主动数据聚合与ANN v2",
        "",
        "## 结论",
        "",
        f"阶段4B-2验收：{'通过' if payload['success'] else '未通过'}。",
        "",
        "## 数据与标签",
        "",
        f"- 候选状态：{data['candidate_count']}；接受标签：{data['accepted_count']}；教师接受率：{100 * data['teacher_acceptance_fraction']:.2f}%；",
        f"- 训练/验证/测试：{data['split_counts']}；",
        f"- 教师模式：{data['teacher_mode_counts']}；",
        "- 原168个可达状态全部由新混合教师重标；新增状态来自旧ANN周围12条受约束轨迹、降阶在策略轨迹和DFN在策略轨迹；",
        "- 标准化和权重只拟合训练轨迹，验证轨迹选模，测试轨迹只做最终评价。",
        "",
        "## 离线模仿",
        "",
        f"- 网络结构：{payload['offline_imitation']['architecture']}；参数量：{payload['offline_imitation']['parameter_count']}；",
        f"- 选中L2：{payload['offline_imitation']['selected_regularization_alpha']}；种子：{payload['offline_imitation']['selected_initialization_seed']}；在策略训练权重：{payload['offline_imitation']['training_weight']['maximum']:.1f}；",
        f"- 测试MAE：{offline['mae_a']:.4f} A；RMSE：{offline['rmse_a']:.4f} A；最大误差：{offline['maximum_absolute_error_a']:.4f} A；",
        "",
        "## Chen2020 DFN闭环",
        "",
        _markdown_table(comparison),
        "",
        f"- ANN v2实质安全过滤介入：{100 * dfn['material_safety_filter_intervention_fraction']:.2f}%（阈值{dfn['material_intervention_threshold_a']:.2f} A）；",
        f"- 平均过滤修正：{dfn['mean_safety_filter_correction_a']:.4f} A；",
        f"- 相对混合教师时间差：{100 * payload['dfn_time_gap_fraction_from_hybrid_teacher']:.2f}%；",
        "",
        "## 边界",
        "",
        "- 主动数据改善只在当前仿真域内成立，安全过滤器仍保留；",
        "- 即使本轮某条轨迹零介入，也不能据此声明裸ANN可部署；",
        "- 下一步仍需多温度、参数扰动、老化和观测器误差验证。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase_four_b2(
    config: PhaseFourB2Config, project_root: str | Path
) -> dict[str, Any]:
    """执行主动采集、重标、ANN v2训练及降阶/DFN闭环验收。"""
    project_root = Path(project_root)
    data_dir = project_root / "data" / "phase4b2"
    metrics_dir = project_root / "outputs" / "metrics"
    model_dir = project_root / "outputs" / "models"
    figures_dir = project_root / "outputs" / "figures"
    for directory in (data_dir, metrics_dir, model_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    phase3, phase4a, phase4b, phase4b_metrics, model = _load_context(
        config, project_root
    )
    seed_ann = TinyANN.load(project_root / config.seed_ann_model)
    legacy = prepare_legacy_states(
        pd.read_csv(project_root / config.legacy_teacher_dataset)
    )
    rollouts = generate_active_rollouts(seed_ann, model, phase3, config)
    active = sample_active_states(rollouts, config)
    candidates = combine_candidate_states(legacy, active)
    attempts, accepted = label_with_hybrid_teacher(
        candidates, model, phase3, phase4b, config
    )
    dataset_metrics = active_dataset_metrics(attempts, accepted, config)
    if not dataset_metrics["success"]:
        raise RuntimeError("主动教师数据闸门未通过，禁止训练ANN v2。")

    rollouts.to_csv(data_dir / "ann_centered_reachable_rollouts.csv", index=False)
    active.to_csv(data_dir / "active_sampled_states.csv", index=False)
    attempts.to_csv(data_dir / "hybrid_teacher_label_audit.csv", index=False)
    accepted.to_csv(data_dir / "active_teacher_dataset.csv", index=False)

    # 第一轮网络先吸收旧ANN周围的12条轨迹；它只作为DAgger采集器，
    # 不直接作为阶段最终模型。
    round1_ann, round1_selection, round1_predictions, round1_offline = (
        train_tiny_ann(accepted, phase4a)
    )
    round1_ann.save(model_dir / "phase4b2_round1_tiny_ann.npz")
    round1_selection.to_csv(
        data_dir / "round1_hyperparameter_selection.csv", index=False
    )
    round1_predictions.to_csv(
        data_dir / "round1_offline_predictions.csv", index=False
    )

    dagger_nominal_rollout = generate_ann_centered_rollout(
        round1_ann,
        model,
        phase3,
        config.active_data.rollouts[0],
    )
    dagger_states = sample_dense_on_policy_states(dagger_nominal_rollout, config)
    # 移除与已有状态完全相同的行，确保五维输入不跨轨迹重复。
    existing_index = pd.MultiIndex.from_frame(accepted[FEATURE_COLUMNS])
    dagger_index = pd.MultiIndex.from_frame(dagger_states[FEATURE_COLUMNS])
    dagger_states = dagger_states.loc[~dagger_index.isin(existing_index)].copy()
    dagger_attempts, dagger_accepted = label_with_hybrid_teacher(
        dagger_states, model, phase3, phase4b, config
    )
    attempts = pd.concat([attempts, dagger_attempts], ignore_index=True, sort=False)
    accepted = pd.concat([accepted, dagger_accepted], ignore_index=True, sort=False)
    dataset_metrics = active_dataset_metrics(attempts, accepted, config)
    if not dataset_metrics["success"]:
        raise RuntimeError("DAgger精炼后的教师数据闸门未通过，禁止训练最终ANN。")
    dagger_nominal_rollout.to_csv(
        data_dir / "round1_ann_nominal_rollout.csv", index=False
    )
    dagger_states.to_csv(data_dir / "dagger_round2_sampled_states.csv", index=False)
    attempts.to_csv(data_dir / "hybrid_teacher_label_audit.csv", index=False)
    accepted.to_csv(data_dir / "active_teacher_dataset.csv", index=False)

    round2_ann, round2_selection, round2_predictions, round2_offline = (
        train_tiny_ann(accepted, phase4a)
    )
    round2_ann.save(model_dir / "phase4b2_round2_tiny_ann.npz")
    round2_selection.to_csv(
        data_dir / "round2_hyperparameter_selection.csv", index=False
    )
    round2_predictions.to_csv(
        data_dir / "round2_offline_predictions.csv", index=False
    )

    # 降阶轨迹上的修正量不能代表DFN反馈后的状态分布。用一条安全包装的
    # DFN名义轨迹收集动作前状态，只加入训练集并保持原验证/测试集冻结。
    round2_dfn_rollout = generate_ann_dfn_rollout(
        round2_ann, model, phase3
    )
    dfn_dagger_states = sample_dense_on_policy_states(
        round2_dfn_rollout,
        config,
        trajectory_id="dagger_round_3_dfn_nominal",
        source_dataset="dagger_round_3_dfn",
        samples_per_soc_bin=config.dfn_refinement.samples_per_soc_bin,
    )
    existing_index = pd.MultiIndex.from_frame(accepted[FEATURE_COLUMNS])
    dfn_dagger_index = pd.MultiIndex.from_frame(dfn_dagger_states[FEATURE_COLUMNS])
    dfn_dagger_states = dfn_dagger_states.loc[
        ~dfn_dagger_index.isin(existing_index)
    ].copy()
    dfn_attempts, dfn_accepted = label_with_hybrid_teacher(
        dfn_dagger_states, model, phase3, phase4b, config
    )
    attempts = pd.concat([attempts, dfn_attempts], ignore_index=True, sort=False)
    accepted = pd.concat([accepted, dfn_accepted], ignore_index=True, sort=False)
    dataset_metrics = active_dataset_metrics(attempts, accepted, config)
    if not dataset_metrics["success"]:
        raise RuntimeError("DFN精炼后的教师数据闸门未通过，禁止训练最终ANN。")
    round2_dfn_rollout.to_csv(
        data_dir / "round2_ann_dfn_nominal_rollout.csv", index=False
    )
    dfn_dagger_states.to_csv(
        data_dir / "dagger_round3_dfn_sampled_states.csv", index=False
    )
    attempts.to_csv(data_dir / "hybrid_teacher_label_audit.csv", index=False)
    accepted.to_csv(data_dir / "active_teacher_dataset.csv", index=False)

    accepted["training_weight"] = 1.0
    on_policy = accepted["source_dataset"].isin(
        ["dagger_round_2", "dagger_round_3_dfn"]
    ) & (accepted["split"] == "train")
    accepted.loc[on_policy, "training_weight"] = (
        config.final_network.on_policy_training_weight
    )
    accepted.to_csv(data_dir / "active_teacher_dataset.csv", index=False)
    final_network = replace(
        phase4a.network,
        hidden_layer_sizes=config.final_network.hidden_layer_sizes,
        regularization_candidates=config.final_network.regularization_candidates,
        initialization_seeds=config.final_network.initialization_seeds,
        maximum_iterations=config.final_network.maximum_iterations,
        convergence_tolerance=config.final_network.convergence_tolerance,
    )
    final_training_config = replace(phase4a, network=final_network)
    ann, selection, predictions, offline = train_tiny_ann(
        accepted,
        final_training_config,
        sample_weight_column="training_weight",
    )
    model_path = ann.save(model_dir / "phase4b2_tiny_ann.npz")
    reloaded = TinyANN.load(model_path)
    features = accepted[list(phase4a.features)].to_numpy(float)
    round_trip_error = float(
        np.max(np.abs(ann.predict(features) - reloaded.predict(features)))
    )
    selection.to_csv(data_dir / "hyperparameter_selection.csv", index=False)
    predictions.to_csv(data_dir / "offline_predictions.csv", index=False)

    reduced = simulate_ann_reduced_closed_loop(
        reloaded, model, phase3, final_training_config
    )
    dfn = simulate_ann_dfn_closed_loop(
        reloaded, model, phase3, final_training_config
    )
    reduced.to_csv(data_dir / "ann_v2_reduced_closed_loop.csv", index=False)
    dfn.to_csv(data_dir / "ann_v2_chen2020_dfn_closed_loop.csv", index=False)
    threshold = config.success_criteria.material_intervention_threshold_a
    reduced_metrics = ann_closed_loop_metrics(reduced, phase3, threshold)
    dfn_metrics = ann_closed_loop_metrics(dfn, phase3, threshold)

    old_dfn_frame = pd.read_csv(
        project_root / "data" / "phase4a" / "ann_chen2020_dfn_closed_loop.csv"
    )
    old_metrics = ann_closed_loop_metrics(old_dfn_frame, phase3, threshold)
    teacher_metrics = phase4b_metrics["hybrid_dfn_closed_loop"]
    teacher_time = float(teacher_metrics["charge_time_min"])
    dfn_time = float(dfn_metrics["charge_time_min"]) if dfn_metrics["charge_time_min"] else np.inf
    time_gap = abs(dfn_time - teacher_time) / teacher_time
    speedup = float(teacher_metrics["mean_mpc_solve_time_ms"]) / dfn_metrics[
        "mean_ann_inference_time_ms"
    ]
    test = offline["split_metrics"]["test"]
    criteria = config.success_criteria
    checks = {
        "active_dataset": dataset_metrics["success"],
        "test_mae": test["mae_a"] <= criteria.maximum_test_mae_a,
        "test_rmse": test["rmse_a"] <= criteria.maximum_test_rmse_a,
        "numpy_model_round_trip": round_trip_error <= 1.0e-12,
        "reduced_closed_loop": reduced_metrics["success"],
        "dfn_closed_loop": dfn_metrics["success"],
        "material_intervention_fraction": dfn_metrics[
            "material_safety_filter_intervention_fraction"
        ]
        <= criteria.maximum_material_intervention_fraction,
        "mean_filter_correction": dfn_metrics["mean_safety_filter_correction_a"]
        <= criteria.maximum_mean_filter_correction_a,
        "dfn_time_gap_from_hybrid_teacher": time_gap
        <= criteria.maximum_dfn_time_gap_fraction_from_hybrid_teacher,
        "improvement_over_phase4a": (
            dfn_time < float(old_metrics["charge_time_min"])
            if criteria.require_improvement_over_phase4a
            else True
        ),
        "inference_speedup": speedup
        >= criteria.minimum_inference_speedup_over_hybrid_mpc,
    }
    comparison = _comparison_frame(old_metrics, dfn_metrics, teacher_metrics)
    comparison.to_csv(metrics_dir / "phase4b2_controller_comparison.csv", index=False)
    plot_active_coverage(accepted, figures_dir / "phase4b2_active_coverage.png")
    plot_offline_active_imitation(
        predictions, figures_dir / "phase4b2_offline_imitation.png"
    )
    hybrid_frame = pd.read_csv(
        project_root / "data" / "phase4b" / "hybrid_teacher_chen2020_dfn_closed_loop.csv"
    )
    plot_ann_v2_dfn_comparison(
        dfn,
        old_dfn_frame,
        hybrid_frame,
        phase3,
        figures_dir / "phase4b2_ann_v2_dfn_comparison.png",
    )
    payload = {
        "configuration": asdict(config),
        "coverage_expansion": _coverage_metrics(
            legacy,
            pd.concat(
                [active, dagger_states, dfn_dagger_states],
                ignore_index=True,
                sort=False,
            ),
        ),
        "active_dataset": dataset_metrics,
        "round1_offline_imitation": round1_offline,
        "round2_offline_imitation": round2_offline,
        "offline_imitation": offline,
        "test_metrics_by_teacher_mode": _mode_test_metrics(predictions),
        "model_round_trip_maximum_error_a": round_trip_error,
        "ann_v2_reduced_closed_loop": reduced_metrics,
        "ann_v2_dfn_closed_loop": dfn_metrics,
        "ann_v1_dfn_reference": old_metrics,
        "hybrid_teacher_reference": teacher_metrics,
        "dfn_time_gap_fraction_from_hybrid_teacher": float(time_gap),
        "inference_speedup_over_hybrid_mpc": float(speedup),
        "checks": checks,
        "standalone_ann_ready": False,
        "ready_for_robustness_validation": bool(all(checks.values())),
        "success": bool(all(checks.values())),
    }
    metrics_path = metrics_dir / "phase4b2_metrics.json"
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = project_root / "outputs" / "phase4b2_report.md"
    _write_report(report_path, payload, comparison)
    return {"metrics": payload, "comparison": comparison}
