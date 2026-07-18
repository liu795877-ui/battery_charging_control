"""面向 MPC 的二阶 RC 电模型与双节点热模型。

统一符号约定：充电电流 ``I`` 为正。电模型端电压等于 OCV、欧姆压升和
两个极化压升之和。热模型忽略可逆熵热，使用电模型损耗乘以辨识热增益。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.linalg import expm


def simulate_electrical_2rc(
    time_s: np.ndarray,
    charge_current_a: np.ndarray,
    initial_soc: float,
    nominal_capacity_ah: float,
    ocv_function: Callable[[np.ndarray], np.ndarray],
    parameters: dict[str, float],
) -> pd.DataFrame:
    """用精确零阶保持离散式模拟二阶 RC 电模型。

    状态为 SOC、快极化电压 ``v1`` 和慢极化电压 ``v2``。每个采样区间内
    假定电流恒定，这与未来数字控制器每 5 s 更新一次电流的设置一致。
    """
    time_s = np.asarray(time_s, dtype=float)
    current_a = np.asarray(charge_current_a, dtype=float)
    if time_s.shape != current_a.shape or time_s.ndim != 1:
        raise ValueError("时间和电流必须是一维且长度相同。")
    if len(time_s) == 0 or np.any(np.diff(time_s) <= 0):
        raise ValueError("时间必须非空且严格递增。")

    r0 = parameters["r0_ohm"]
    r1 = parameters["r1_ohm"]
    r2 = parameters["r2_ohm"]
    tau1 = parameters["tau1_s"]
    tau2 = parameters["tau2_s"]

    soc = np.empty_like(time_s)
    v1 = np.zeros_like(time_s)
    v2 = np.zeros_like(time_s)
    soc[0] = initial_soc

    for k in range(1, len(time_s)):
        dt = time_s[k] - time_s[k - 1]
        interval_current = current_a[k - 1]
        soc[k] = soc[k - 1] + interval_current * dt / (3600.0 * nominal_capacity_ah)
        a1 = np.exp(-dt / tau1)
        a2 = np.exp(-dt / tau2)
        v1[k] = a1 * v1[k - 1] + r1 * (1.0 - a1) * interval_current
        v2[k] = a2 * v2[k - 1] + r2 * (1.0 - a2) * interval_current

    ocv_v = np.asarray(ocv_function(np.clip(soc, 0.0, 1.0)), dtype=float)
    terminal_voltage_v = ocv_v + r0 * current_a + v1 + v2
    # 不可逆损耗近似；静置时 I=0，所以不会人为继续产热。
    electrical_loss_w = np.maximum(current_a * (terminal_voltage_v - ocv_v), 0.0)
    return pd.DataFrame(
        {
            "soc_predicted": soc,
            "ocv_predicted_v": ocv_v,
            "polarization_fast_v": v1,
            "polarization_slow_v": v2,
            "terminal_voltage_predicted_v": terminal_voltage_v,
            "electrical_loss_predicted_w": electrical_loss_w,
        }
    )


def simulate_two_node_thermal(
    time_s: np.ndarray,
    heat_input_w: np.ndarray,
    initial_temperature_c: float,
    ambient_temperature_c: float,
    core_fraction: float,
    parameters: dict[str, float],
) -> pd.DataFrame:
    """模拟核心—表面双节点热网络。

    热源全部施加在核心节点。矩阵指数给出每个恒定热输入区间的精确离散化，
    因而不受显式欧拉法步长稳定性的影响。
    """
    time_s = np.asarray(time_s, dtype=float)
    heat_w = np.asarray(heat_input_w, dtype=float)
    if time_s.shape != heat_w.shape or time_s.ndim != 1:
        raise ValueError("时间和热输入必须是一维且长度相同。")
    if not 0.0 < core_fraction < 1.0:
        raise ValueError("核心热容量比例必须位于 0 和 1 之间。")

    c_total = parameters["total_heat_capacity_j_per_k"]
    c_core = core_fraction * c_total
    c_surface = (1.0 - core_fraction) * c_total
    r_core_surface = parameters["r_core_surface_k_per_w"]
    r_surface_ambient = parameters["r_surface_ambient_k_per_w"]
    heat_gain = parameters["heat_gain"]

    matrix_a = np.array(
        [
            [
                -1.0 / (c_core * r_core_surface),
                1.0 / (c_core * r_core_surface),
            ],
            [
                1.0 / (c_surface * r_core_surface),
                -(1.0 / r_core_surface + 1.0 / r_surface_ambient) / c_surface,
            ],
        ]
    )
    matrix_b = np.array([1.0 / c_core, 0.0])

    # 状态使用相对环境温升，初值允许不等于环境温度。
    state = np.array(
        [
            initial_temperature_c - ambient_temperature_c,
            initial_temperature_c - ambient_temperature_c,
        ],
        dtype=float,
    )
    states = np.empty((len(time_s), 2), dtype=float)
    states[0] = state
    discretization_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    for k in range(1, len(time_s)):
        dt = float(time_s[k] - time_s[k - 1])
        if dt not in discretization_cache:
            matrix_ad = expm(matrix_a * dt)
            matrix_bd = np.linalg.solve(matrix_a, (matrix_ad - np.eye(2)) @ matrix_b)
            discretization_cache[dt] = matrix_ad, matrix_bd
        matrix_ad, matrix_bd = discretization_cache[dt]
        state = matrix_ad @ state + matrix_bd * heat_gain * heat_w[k - 1]
        states[k] = state

    core_c = states[:, 0] + ambient_temperature_c
    surface_c = states[:, 1] + ambient_temperature_c
    average_c = core_fraction * core_c + (1.0 - core_fraction) * surface_c
    return pd.DataFrame(
        {
            "core_temperature_predicted_c": core_c,
            "surface_temperature_predicted_c": surface_c,
            "average_temperature_predicted_c": average_c,
        }
    )
