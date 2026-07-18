"""从 DFN 虚拟试验中辨识二阶 RC 与双节点热模型参数。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares

from .reduced_model import simulate_electrical_2rc, simulate_two_node_thermal


def build_ocv_function(ocv_table: pd.DataFrame) -> PchipInterpolator:
    """构建保持单调形状的 OCV–SOC 插值函数，并允许边界附近外推。"""
    ordered = ocv_table.sort_values("soc")
    return PchipInterpolator(
        ordered["soc"].to_numpy(), ordered["ocv_v"].to_numpy(), extrapolate=True
    )


def _electrical_parameters(log_values: np.ndarray) -> dict[str, float]:
    values = np.exp(log_values)
    return {
        "r0_ohm": float(values[0]),
        "r1_ohm": float(values[1]),
        "tau1_s": float(values[2]),
        "r2_ohm": float(values[3]),
        "tau2_s": float(values[4]),
        "c1_f": float(values[2] / values[1]),
        "c2_f": float(values[4] / values[3]),
    }


def fit_electrical_2rc(
    pulse_data: pd.DataFrame,
    ocv_table: pd.DataFrame,
    nominal_capacity_ah: float,
    maximum_function_evaluations: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    """对全部 SOC 脉冲轨迹做全局非线性最小二乘辨识。"""
    ocv_function = build_ocv_function(ocv_table)
    groups = [group.copy() for _, group in pulse_data.groupby("profile_name")]

    def residual(log_values: np.ndarray) -> np.ndarray:
        parameters = _electrical_parameters(log_values)
        errors: list[np.ndarray] = []
        for frame in groups:
            prediction = simulate_electrical_2rc(
                frame["time_s"].to_numpy(),
                frame["charge_current_a"].to_numpy(),
                float(frame["initial_soc"].iloc[0]),
                nominal_capacity_ah,
                ocv_function,
                parameters,
            )
            errors.append(
                prediction["terminal_voltage_predicted_v"].to_numpy()
                - frame["terminal_voltage_v"].to_numpy()
            )
        return np.concatenate(errors)

    initial = np.log([0.015, 0.010, 20.0, 0.015, 300.0])
    lower = np.log([1.0e-4, 1.0e-5, 1.0, 1.0e-5, 50.0])
    upper = np.log([0.10, 0.20, 200.0, 0.20, 5000.0])
    result = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        max_nfev=maximum_function_evaluations,
    )
    parameters = _electrical_parameters(result.x)
    errors_v = residual(result.x)
    diagnostics: dict[str, Any] = {
        "success": bool(result.success),
        "message": str(result.message),
        "function_evaluations": int(result.nfev),
        "training_voltage_rmse_mv": float(np.sqrt(np.mean(errors_v**2)) * 1000),
        "training_voltage_mae_mv": float(np.mean(np.abs(errors_v)) * 1000),
    }
    return parameters, diagnostics


def _thermal_parameters(log_values: np.ndarray) -> dict[str, float]:
    values = np.exp(log_values)
    return {
        "total_heat_capacity_j_per_k": float(values[0]),
        "r_core_surface_k_per_w": float(values[1]),
        "r_surface_ambient_k_per_w": float(values[2]),
        "heat_gain": float(values[3]),
    }


def fit_two_node_thermal(
    frame: pd.DataFrame,
    heat_input_w: np.ndarray,
    ambient_temperature_c: float,
    core_fraction: float,
    maximum_function_evaluations: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    """拟合双节点模型的加权平均温度，不虚构核心/表面温度观测。"""
    time_s = frame["time_s"].to_numpy()
    measured_c = frame["average_temperature_c"].to_numpy()
    initial_c = float(measured_c[0])

    def residual(log_values: np.ndarray) -> np.ndarray:
        prediction = simulate_two_node_thermal(
            time_s,
            heat_input_w,
            initial_c,
            ambient_temperature_c,
            core_fraction,
            _thermal_parameters(log_values),
        )
        return prediction["average_temperature_predicted_c"].to_numpy() - measured_c

    initial = np.log([100.0, 1.0, 5.0, 1.0])
    lower = np.log([20.0, 0.01, 0.10, 0.10])
    upper = np.log([1000.0, 50.0, 50.0, 10.0])
    result = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        max_nfev=maximum_function_evaluations,
    )
    parameters = _thermal_parameters(result.x)
    errors_c = residual(result.x)
    parameter_names = [
        "total_heat_capacity_j_per_k",
        "r_core_surface_k_per_w",
        "r_surface_ambient_k_per_w",
        "heat_gain",
    ]
    parameters_at_bounds = [
        name
        for name, active in zip(parameter_names, result.active_mask, strict=True)
        if active != 0
    ]
    diagnostics: dict[str, Any] = {
        "success": bool(result.success),
        "message": str(result.message),
        "function_evaluations": int(result.nfev),
        "training_average_temperature_rmse_c": float(np.sqrt(np.mean(errors_c**2))),
        "training_average_temperature_mae_c": float(np.mean(np.abs(errors_c))),
        "parameters_at_optimization_bounds": parameters_at_bounds,
        "internal_temperature_states_independently_validated": False,
        "identifiability_note": (
            "DFN lumped thermal data only observes average temperature; core/surface "
            "temperatures are latent and the core heat-capacity fraction is fixed. "
            "A core-surface resistance on its lower bound indicates that the two-node "
            "model has collapsed toward lumped behavior."
        ),
    }
    return parameters, diagnostics


def error_metrics(
    actual: Sequence[float], predicted: Sequence[float]
) -> dict[str, float]:
    """计算可直接用于验收的 RMSE、MAE 和最大绝对误差。"""
    error = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "maximum_absolute_error": float(np.max(np.abs(error))),
    }
