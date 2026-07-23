"""组织第二阶段的数据生成、参数辨识、独立验证和结果落盘。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .characterization import (
    generate_dynamic_profile,
    generate_ocv_curve,
    generate_pulse_data,
)
from .identification import (
    build_ocv_function,
    error_metrics,
    fit_electrical_2rc,
    fit_two_node_thermal,
)
from .phase2_config import PhaseTwoConfig
from .phase2_plotting import plot_characterization, plot_validation
from .reduced_model import simulate_electrical_2rc, simulate_two_node_thermal


def _electrical_prediction(
    frame: pd.DataFrame,
    initial_soc: float,
    config: PhaseTwoConfig,
    ocv_table: pd.DataFrame,
    parameters: dict[str, float],
) -> pd.DataFrame:
    """在一条指定电流轨迹上运行已辨识的2RC模型。"""
    return simulate_electrical_2rc(
        frame["time_s"].to_numpy(),
        frame["charge_current_a"].to_numpy(),
        initial_soc,
        config.battery.nominal_capacity_ah,
        build_ocv_function(ocv_table),
        parameters,
    )


def _combined_prediction(
    frame: pd.DataFrame,
    initial_soc: float,
    config: PhaseTwoConfig,
    ocv_table: pd.DataFrame,
    electrical_parameters: dict[str, float],
    thermal_parameters: dict[str, float],
) -> pd.DataFrame:
    """串联2RC和双节点热模型，形成未来MPC使用的预测模型。"""
    electrical = _electrical_prediction(
        frame, initial_soc, config, ocv_table, electrical_parameters
    )
    thermal = simulate_two_node_thermal(
        frame["time_s"].to_numpy(),
        electrical["electrical_loss_predicted_w"].to_numpy(),
        float(frame["average_temperature_c"].iloc[0]),
        config.battery.ambient_temperature_c,
        config.identification.core_heat_capacity_fraction,
        thermal_parameters,
    )
    return pd.concat([frame.reset_index(drop=True), electrical, thermal], axis=1)


def _write_report(
    path: Path,
    electrical_parameters: dict[str, float],
    thermal_parameters: dict[str, float],
    metrics: dict[str, Any],
) -> None:
    """生成便于非代码阅读的阶段报告。"""
    voltage = metrics["validation"]["voltage"]
    temperature = metrics["validation"]["average_temperature"]
    lines = [
        "# 第二阶段报告：2RC＋双节点热模型辨识",
        "",
        "## 结论",
        "",
        (
            "已使用 Chen2020 DFN 虚拟试验完成 OCV、充电脉冲、热响应数据生成，"
            "并在独立动态电流轨迹上验证降阶模型。"
        ),
        "",
        "## 2RC 电模型参数",
        "",
        "| 参数 | 数值 |",
        "|---|---:|",
        f"| R0 | {electrical_parameters['r0_ohm']:.6f} ohm |",
        f"| R1 | {electrical_parameters['r1_ohm']:.6f} ohm |",
        f"| C1 | {electrical_parameters['c1_f']:.2f} F |",
        f"| tau1 | {electrical_parameters['tau1_s']:.2f} s |",
        f"| R2 | {electrical_parameters['r2_ohm']:.6f} ohm |",
        f"| C2 | {electrical_parameters['c2_f']:.2f} F |",
        f"| tau2 | {electrical_parameters['tau2_s']:.2f} s |",
        "",
        "## 双节点热模型参数",
        "",
        "| 参数 | 数值 |",
        "|---|---:|",
        (
            "| 总热容量 | "
            f"{thermal_parameters['total_heat_capacity_j_per_k']:.2f} J/K |"
        ),
        (
            "| 核心—表面热阻 | "
            f"{thermal_parameters['r_core_surface_k_per_w']:.4f} K/W |"
        ),
        (
            "| 表面—环境热阻 | "
            f"{thermal_parameters['r_surface_ambient_k_per_w']:.4f} K/W |"
        ),
        f"| 损耗热增益 | {thermal_parameters['heat_gain']:.4f} |",
        "",
        "## 独立验证",
        "",
        f"- 电压 RMSE：{voltage['rmse_mv']:.2f} mV；",
        f"- 电压最大绝对误差：{voltage['maximum_absolute_error_mv']:.2f} mV；",
        f"- 平均温度 RMSE：{temperature['rmse_c']:.3f} degC；",
        (
            "- 平均温度最大绝对误差："
            f"{temperature['maximum_absolute_error_c']:.3f} degC；"
        ),
        f"- 第一版验收：{'通过' if metrics['success'] else '未通过'}。",
        (
            "- 双节点内部温差独立验证：未完成；核心—表面热阻命中优化下限，"
            "当前结构退化为接近集总热行为。"
        ),
        "",
        "## 适用边界",
        "",
        (
            "Chen2020 集总热模型只提供平均温度。双节点模型的加权平均温度得到"
            "验证，但核心与表面温度是潜在内部状态，尚无独立空间温度数据验证。"
        ),
        (
            "当前参数只适用于 25 degC、10%–80% SOC 附近和本阶段电流激励范围，"
            "后续 MPC 不应在未验证区域外推。"
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase_two(config: PhaseTwoConfig, project_root: str | Path) -> dict[str, Any]:
    """执行第二阶段全流程并返回参数、指标和输出路径。"""
    project_root = Path(project_root)
    data_dir = project_root / "data" / "phase2"
    metrics_dir = project_root / "outputs" / "metrics"
    figures_dir = project_root / "outputs" / "figures"
    for directory in (data_dir, metrics_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    np.random.seed(config.random_seed)
    ocv_table = generate_ocv_curve(config)
    pulse_data = generate_pulse_data(config)
    thermal_training = generate_dynamic_profile(
        config.experiment.thermal_training_profile,
        config.experiment.thermal_training_initial_soc,
        config,
        "thermal_training",
    )
    validation = generate_dynamic_profile(
        config.experiment.validation_profile,
        config.experiment.validation_initial_soc,
        config,
        "independent_validation",
    )

    ocv_table.to_csv(data_dir / "ocv_curve.csv", index=False)
    pulse_data.to_csv(data_dir / "pulse_characterization.csv", index=False)
    thermal_training.to_csv(data_dir / "thermal_training.csv", index=False)
    validation.to_csv(data_dir / "independent_validation.csv", index=False)

    electrical_parameters, electrical_diagnostics = fit_electrical_2rc(
        pulse_data,
        ocv_table,
        config.battery.nominal_capacity_ah,
        config.identification.maximum_function_evaluations,
    )
    training_electrical = _electrical_prediction(
        thermal_training,
        config.experiment.thermal_training_initial_soc,
        config,
        ocv_table,
        electrical_parameters,
    )
    thermal_parameters, thermal_diagnostics = fit_two_node_thermal(
        thermal_training,
        training_electrical["electrical_loss_predicted_w"].to_numpy(),
        config.battery.ambient_temperature_c,
        config.identification.core_heat_capacity_fraction,
        config.identification.maximum_function_evaluations,
    )
    validation_result = _combined_prediction(
        validation,
        config.experiment.validation_initial_soc,
        config,
        ocv_table,
        electrical_parameters,
        thermal_parameters,
    )
    validation_result.to_csv(data_dir / "validation_predictions.csv", index=False)

    voltage_raw = error_metrics(
        validation_result["terminal_voltage_v"],
        validation_result["terminal_voltage_predicted_v"],
    )
    temperature_raw = error_metrics(
        validation_result["average_temperature_c"],
        validation_result["average_temperature_predicted_c"],
    )
    soc_raw = error_metrics(
        validation_result["soc"], validation_result["soc_predicted"]
    )
    voltage_metrics = {
        "rmse_mv": voltage_raw["rmse"] * 1000.0,
        "mae_mv": voltage_raw["mae"] * 1000.0,
        "maximum_absolute_error_mv": voltage_raw["maximum_absolute_error"] * 1000.0,
    }
    temperature_metrics = {
        "rmse_c": temperature_raw["rmse"],
        "mae_c": temperature_raw["mae"],
        "maximum_absolute_error_c": temperature_raw["maximum_absolute_error"],
    }
    metrics: dict[str, Any] = {
        "electrical_training": electrical_diagnostics,
        "thermal_training": thermal_diagnostics,
        "validation": {
            "voltage": voltage_metrics,
            "average_temperature": temperature_metrics,
            "soc": soc_raw,
        },
        "success_criteria": {
            "voltage_rmse_mv": config.success_criteria.validation_voltage_rmse_mv,
            "average_temperature_rmse_c": (
                config.success_criteria.validation_average_temperature_rmse_c
            ),
        },
    }
    metrics["success"] = bool(
        voltage_metrics["rmse_mv"] <= config.success_criteria.validation_voltage_rmse_mv
        and temperature_metrics["rmse_c"]
        <= config.success_criteria.validation_average_temperature_rmse_c
    )
    metrics["scope_assessment"] = {
        "average_outputs_validated": metrics["success"],
        "core_surface_temperature_split_validated": False,
        "safe_for_first_mpc_average_temperature_prediction": metrics["success"],
        "safe_for_claiming_validated_core_temperature": False,
    }
    parameter_payload = {
        "electrical_2rc": electrical_parameters,
        "thermal_two_node": thermal_parameters,
        "core_heat_capacity_fraction": (
            config.identification.core_heat_capacity_fraction
        ),
    }
    (metrics_dir / "phase2_identified_parameters.json").write_text(
        json.dumps(parameter_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (metrics_dir / "phase2_validation_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    characterization_figure = plot_characterization(
        ocv_table, pulse_data, figures_dir / "phase2_characterization.png"
    )
    validation_figure = plot_validation(
        validation_result, figures_dir / "phase2_reduced_model_validation.png"
    )
    report_path = project_root / "outputs" / "phase2_report.md"
    _write_report(report_path, electrical_parameters, thermal_parameters, metrics)
    return {
        "parameters": parameter_payload,
        "metrics": metrics,
        "outputs": {
            "characterization_figure": str(characterization_figure),
            "validation_figure": str(validation_figure),
            "report": str(report_path),
        },
    }
