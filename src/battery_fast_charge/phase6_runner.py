"""组织 Phase 6 论文方法迁移的数据、DNN 和分级闭环验证。"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ann_model import TinyANN
from .identification import build_ocv_function
from .mpc import ReducedBatteryModel
from .paper_method import (
    generate_initial_state_design,
    generate_paper_teacher_dataset,
    train_paper_dnn,
)
from .phase3_config import load_phase_three_config
from .phase5a_config import load_phase_five_a_config
from .phase6_closed_loop import (
    compare_with_teacher,
    pure_dnn_closed_loop_metrics,
    run_pure_dnn_phase5a_stress,
    simulate_pure_dnn_dfn_closed_loop,
    simulate_pure_dnn_reduced_closed_loop,
    temperature_anchor_config,
)
from .phase6_config import PhaseSixConfig
from .phase6_plotting import (
    plot_nominal_closed_loop,
    plot_paper_dataset_audit,
    plot_paper_dnn_offline,
)


def _load_context(config: PhaseSixConfig, root: Path):
    phase3 = load_phase_three_config(root / config.source_phase3_config)
    validation = json.loads(
        (root / phase3.artifacts.validation_metrics).read_text(encoding="utf-8")
    )
    if not validation.get("success", False):
        raise RuntimeError("Phase 2 降阶模型未通过，禁止生成 Phase 6 MPC 标签。")
    parameters = json.loads(
        (root / phase3.artifacts.identified_parameters).read_text(encoding="utf-8")
    )
    ocv_function = build_ocv_function(pd.read_csv(root / phase3.artifacts.ocv_curve))
    return phase3, parameters, ocv_function


def _nominal_checks(
    dfn_metrics: dict[str, Any], comparison: dict[str, Any], config: PhaseSixConfig
) -> dict[str, bool]:
    criteria = config.success_criteria
    return {
        "closed_loop_current_nrmse": comparison["current_nrmse"] <= criteria.maximum_nominal_current_nrmse,
        "no_serious_physical_violation": not dfn_metrics["serious_physical_violation"],
        "reached_target_soc": bool(dfn_metrics["reached_target_soc"]),
        "charge_time_gap": comparison["charge_time_gap_fraction"] <= criteria.maximum_nominal_charge_time_gap_fraction,
        "inference_speedup": comparison["inference_speedup_over_mpc"] >= criteria.minimum_inference_speedup_over_mpc,
    }


def _write_report(path: Path, metrics: dict[str, Any]) -> None:
    dataset = metrics["paper_dataset"]
    offline = metrics["offline_dnn"]["split_metrics"]["test"]
    nominal = metrics["nominal_25c"]
    comparison = nominal["comparison"]
    dfn = nominal["dfn"]
    lines = [
        "# Phase 6 报告：论文式 DNN 显式 MPC 方法迁移验证",
        "",
        "## 结论",
        "",
        f"25 ℃ 名义论文方法门槛：{'通过' if nominal['success'] else '未通过'}。",
        f"Phase 6 总体：{'通过' if metrics['success'] else '尚未通过'}。",
        "",
        "## 方法边界",
        "",
        "- 对照论文 Section 3 和 EHM case study：混合 DOCE 初态采样、一次 MPC 求解生成短轨迹、轨迹展开、三隐层 DNN 显式控制；",
        "- 本项目使用 Chen2020 的 2RC＋双节点平均热模型和五维状态，因此属于方法迁移，不是 EHM 数值复现；",
        "- DNN 闭环直接施加未经裁剪的网络输出，没有调用 Phase 4–5 安全过滤器。",
        "",
        "## 数据集",
        "",
        f"- 初态尝试/接受：{dataset['attempted_initial_state_count']}/{dataset['accepted_initial_state_count']}；接受率 {100 * dataset['teacher_acceptance_fraction']:.2f}%；",
        f"- 展开样本：{dataset['unfolded_sample_count']}；平均单次教师求解 {dataset['mean_teacher_solve_time_ms']:.2f} ms；",
        "",
        "## 离线 DNN",
        "",
        f"- 结构：{metrics['offline_dnn']['architecture']}；参数：{metrics['offline_dnn']['parameter_count']}；",
        f"- 测试 RMSE：{offline['rmse_a']:.4f} A；测试 NRMSE：{100 * offline['nrmse']:.3f}%；",
        "- 训练/验证/测试误差接近；当前结果更像容量或标签映射复杂度限制，而非单纯测试集过拟合；",
        "",
        "## 25 ℃ Chen2020 DFN 闭环",
        "",
        f"- 电流 NRMSE：{100 * comparison['current_nrmse']:.3f}%；",
        f"- DNN 充电时间：{dfn['charge_time_min']} min；相对 MPC 时间差：{100 * comparison['charge_time_gap_fraction']:.3f}%；",
        f"- 推理相对 MPC 加速：{comparison['inference_speedup_over_mpc']:.1f} 倍；",
        f"- 最大电压/平均温度：{dfn['maximum_voltage_v']:.4f} V / {dfn['maximum_temperature_c']:.4f} ℃；",
        f"- 最大电流/单步变化：{dfn['maximum_current_a']:.4f} A / {dfn['maximum_current_change_a']:.4f} A；",
        "- 更短充电时间不能解释为胜过教师：只有同时满足电流复现、时间偏差和物理约束门槛才算通过；",
        "",
        "## 分级验证",
        "",
        "名义门槛未通过，因此 15/30 ℃ 和 Phase 5A 压力测试按预注册顺序不运行。",
        "下一轮应作为 Phase 6B 独立实验，不能通过降低门槛或加入安全裁剪改写本阶段结论。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase_six(config: PhaseSixConfig, project_root: str | Path) -> dict[str, Any]:
    """执行论文式数据生成、纯 DNN 训练和分级闭环验证。"""
    root = Path(project_root)
    data_dir = root / "data" / "phase6_paper_method_validation"
    model_dir = root / "outputs" / "models"
    metrics_dir = root / "outputs" / "metrics"
    figures_dir = root / "outputs" / "figures"
    for directory in (data_dir, model_dir, metrics_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    phase3, parameters, ocv_function = _load_context(config, root)
    reduced_model = ReducedBatteryModel(phase3, ocv_function, parameters)
    design = generate_initial_state_design(config)
    attempts, dataset, dataset_metrics = generate_paper_teacher_dataset(
        design, reduced_model, phase3, config
    )
    attempts.to_csv(data_dir / "initial_state_audit.csv", index=False)
    dataset.to_csv(data_dir / "paper_teacher_dataset.csv", index=False)
    plot_paper_dataset_audit(
        attempts, dataset, figures_dir / "phase6_paper_dataset_audit.png"
    )
    if not dataset_metrics["success"]:
        payload = {
            "configuration": asdict(config),
            "paper_dataset": dataset_metrics,
            "nominal_25c": {"status": "not_run_dataset_gate_failed", "success": False},
            "extended_validation": {"status": "not_run_nominal_gate_failed"},
            "success": False,
        }
        (metrics_dir / "phase6_metrics.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return payload

    ann, selection, predictions, offline_metrics = train_paper_dnn(dataset, config)
    model_path = ann.save(model_dir / "phase6_paper_dnn.npz")
    reloaded = TinyANN.load(model_path)
    round_trip = float(
        np.max(
            np.abs(
                ann.predict_unclipped(dataset[list(ann.feature_names)].to_numpy())
                - reloaded.predict_unclipped(dataset[list(ann.feature_names)].to_numpy())
            )
        )
    )
    selection.to_csv(data_dir / "network_selection.csv", index=False)
    predictions.to_csv(data_dir / "offline_predictions.csv", index=False)
    plot_paper_dnn_offline(
        predictions, figures_dir / "phase6_paper_dnn_offline.png"
    )

    nominal = temperature_anchor_config(
        phase3,
        config.nominal_validation.temperature_c,
        config.nominal_validation.maximum_simulation_time_s,
    )
    nominal_model = ReducedBatteryModel(nominal, ocv_function, parameters)
    reduced_frame = simulate_pure_dnn_reduced_closed_loop(
        reloaded, nominal_model, nominal, config.nominal_validation.maximum_simulation_time_s
    )
    reduced_frame.to_csv(data_dir / "pure_dnn_reduced_25c.csv", index=False)
    dfn_frame = simulate_pure_dnn_dfn_closed_loop(
        reloaded, nominal_model, nominal, config.nominal_validation.maximum_simulation_time_s
    )
    dfn_frame.to_csv(data_dir / "pure_dnn_dfn_25c.csv", index=False)
    reduced_metrics = pure_dnn_closed_loop_metrics(reduced_frame, nominal, config)
    dfn_metrics = pure_dnn_closed_loop_metrics(dfn_frame, nominal, config)
    teacher_frame = pd.read_csv(root / config.nominal_validation.teacher_trajectory)
    teacher_metrics = json.loads(
        (root / config.nominal_validation.teacher_metrics).read_text(encoding="utf-8")
    )["dfn_closed_loop"]
    comparison = compare_with_teacher(
        dfn_frame, dfn_metrics, teacher_frame, teacher_metrics, config
    )
    nominal_checks = _nominal_checks(dfn_metrics, comparison, config)
    nominal_success = bool(all(nominal_checks.values()))
    plot_nominal_closed_loop(
        dfn_frame, teacher_frame, figures_dir / "phase6_nominal_25c_comparison.png"
    )

    extended: dict[str, Any] = {"status": "not_run_nominal_gate_failed"}
    if nominal_success:
        anchor_records = []
        anchor_frames = []
        for temperature in (15.0, 25.0, 30.0):
            anchor_config = temperature_anchor_config(
                phase3, temperature, config.nominal_validation.maximum_simulation_time_s
            )
            anchor_model = ReducedBatteryModel(anchor_config, ocv_function, parameters)
            frame = simulate_pure_dnn_dfn_closed_loop(
                reloaded,
                anchor_model,
                anchor_config,
                config.nominal_validation.maximum_simulation_time_s,
            )
            frame["anchor_temperature_c"] = temperature
            anchor_frames.append(frame)
            anchor_records.append(
                {
                    "anchor_temperature_c": temperature,
                    **pure_dnn_closed_loop_metrics(frame, anchor_config, config),
                }
            )
        anchor_summary = pd.DataFrame.from_records(anchor_records)
        anchor_summary.to_csv(data_dir / "pure_dnn_temperature_anchor_summary.csv", index=False)
        pd.concat(anchor_frames, ignore_index=True).to_csv(
            data_dir / "pure_dnn_temperature_anchor_trajectories.csv", index=False
        )
        phase5a = load_phase_five_a_config(root / config.source_phase5a_config)
        stress_summary = run_pure_dnn_phase5a_stress(
            reloaded, parameters, ocv_function, phase3, phase5a, config
        )
        stress_summary.to_csv(data_dir / "pure_dnn_phase5a_stress_summary.csv", index=False)
        all_complete = bool(anchor_summary["reached_target_soc"].all())
        no_serious = bool((~anchor_summary["serious_physical_violation"]).all())
        extended_success = bool(
            (all_complete if config.success_criteria.require_all_temperature_anchors_complete else True)
            and (no_serious if config.success_criteria.require_all_temperature_anchors_without_serious_violation else True)
        )
        extended = {
            "status": "completed",
            "temperature_anchors": anchor_summary.to_dict("records"),
            "phase5a_stress_completion_fraction": float(stress_summary["completion_success"].mean()),
            "phase5a_stress_no_serious_violation_fraction": float((~stress_summary["serious_physical_violation"]).mean()),
            "success": extended_success,
        }

    payload = {
        "configuration": asdict(config),
        "paper_dataset": dataset_metrics,
        "offline_dnn": offline_metrics,
        "model_round_trip_maximum_error_a": round_trip,
        "nominal_25c": {
            "reduced_model": reduced_metrics,
            "dfn": dfn_metrics,
            "teacher": teacher_metrics,
            "comparison": comparison,
            "checks": nominal_checks,
            "success": nominal_success,
        },
        "extended_validation": extended,
        "success": bool(nominal_success and extended.get("success", False)),
    }
    metrics_path = metrics_dir / "phase6_metrics.json"
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(root / "outputs" / "phase6_report.md", payload)
    return payload
