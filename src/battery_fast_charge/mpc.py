"""第三阶段的降阶预测模型和约束 MPC 教师控制器。

这里的“最优”始终是相对于第二阶段降阶模型、当前代价函数和约束而言，
并不表示已经得到真实电池在所有工况下的全局最优控制器。
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from .phase3_config import PhaseThreeConfig


@dataclass(frozen=True)
class ReducedState:
    """MPC 内部状态；充电电流的符号约定为正。"""

    soc: float
    polarization_fast_v: float
    polarization_slow_v: float
    core_temperature_c: float
    surface_temperature_c: float
    previous_current_a: float


@dataclass(frozen=True)
class StepOutput:
    """一个控制周期末端的可观测输出和约束评估值。"""

    terminal_voltage_v: float
    constraint_voltage_v: float
    average_temperature_c: float
    constraint_temperature_c: float
    electrical_loss_w: float


@dataclass(frozen=True)
class MPCResult:
    """一次在线优化的第一控制动作和诊断信息。"""

    current_a: float
    optimizer_success: bool
    prediction_feasible: bool
    used_fallback: bool
    status: str
    objective_value: float
    solve_time_s: float
    predicted_maximum_voltage_v: float
    predicted_maximum_temperature_c: float
    predicted_terminal_soc: float
    minimum_constraint_margin: float
    slack_voltage_v: float = 0.0
    slack_temperature_c: float = 0.0
    slack_soc: float = 0.0
    slack_current_change_a: float = 0.0
    braking_distance_steps: int = 0
    braking_current_deficit_a: float = 0.0


class ReducedBatteryModel:
    """将已辨识 2RC 和双节点热模型封装成 MPC 单步预测器。"""

    def __init__(
        self,
        config: PhaseThreeConfig,
        ocv_function: Callable[[np.ndarray], np.ndarray],
        identified_parameters: dict[str, Any],
    ) -> None:
        self.config = config
        self.ocv_function = ocv_function
        self.electrical = identified_parameters["electrical_2rc"]
        self.thermal = identified_parameters["thermal_two_node"]
        self.core_fraction = float(
            identified_parameters["core_heat_capacity_fraction"]
        )
        self.dt = config.control.control_interval_s
        self._build_thermal_discretization()

    def _build_thermal_discretization(self) -> None:
        """预先计算固定 5 s 周期下的精确热模型离散矩阵。"""
        c_total = self.thermal["total_heat_capacity_j_per_k"]
        c_core = self.core_fraction * c_total
        c_surface = (1.0 - self.core_fraction) * c_total
        r_cs = self.thermal["r_core_surface_k_per_w"]
        r_sa = self.thermal["r_surface_ambient_k_per_w"]
        matrix_a = np.array(
            [
                [-1.0 / (c_core * r_cs), 1.0 / (c_core * r_cs)],
                [
                    1.0 / (c_surface * r_cs),
                    -(1.0 / r_cs + 1.0 / r_sa) / c_surface,
                ],
            ],
            dtype=float,
        )
        matrix_b = np.array([1.0 / c_core, 0.0], dtype=float)
        self.thermal_ad = expm(matrix_a * self.dt)
        self.thermal_bd = np.linalg.solve(
            matrix_a, (self.thermal_ad - np.eye(2)) @ matrix_b
        )

    def ocv(self, soc: float) -> float:
        """读取 OCV；只在本阶段已经验证的 10%–80% SOC 内使用。"""
        return float(np.asarray(self.ocv_function(float(soc))))

    def average_temperature(self, state: ReducedState) -> float:
        """第二阶段真正得到验证的是该加权平均温度，而非两个节点本身。"""
        return (
            self.core_fraction * state.core_temperature_c
            + (1.0 - self.core_fraction) * state.surface_temperature_c
        )

    def step(self, state: ReducedState, current_a: float) -> tuple[ReducedState, StepOutput]:
        """在一个恒流控制周期内推进电、热状态。

        电压约束同时检查施加新电流后的周期起点和周期末端，避免在电流切换瞬间
        只检查末值而漏掉欧姆压升。
        """
        current_a = float(current_a)
        electrical = self.electrical
        capacity_ah = self.config.battery.nominal_capacity_ah

        ocv_start = self.ocv(state.soc)
        voltage_start = (
            ocv_start
            + electrical["r0_ohm"] * current_a
            + state.polarization_fast_v
            + state.polarization_slow_v
        )

        a1 = np.exp(-self.dt / electrical["tau1_s"])
        a2 = np.exp(-self.dt / electrical["tau2_s"])
        v1_next = (
            a1 * state.polarization_fast_v
            + electrical["r1_ohm"] * (1.0 - a1) * current_a
        )
        v2_next = (
            a2 * state.polarization_slow_v
            + electrical["r2_ohm"] * (1.0 - a2) * current_a
        )
        soc_next = state.soc + current_a * self.dt / (3600.0 * capacity_ah)
        ocv_end = self.ocv(soc_next)
        voltage_end = (
            ocv_end
            + electrical["r0_ohm"] * current_a
            + v1_next
            + v2_next
        )

        # 使用控制周期起末的平均不可逆压降估算发热，避免只取单端点造成偏差。
        resistive_overpotential_v = max(
            0.5
            * ((voltage_start - ocv_start) + (voltage_end - ocv_end)),
            0.0,
        )
        electrical_loss_w = current_a * resistive_overpotential_v

        ambient = self.config.battery.ambient_temperature_c
        relative_temperature = np.array(
            [
                state.core_temperature_c - ambient,
                state.surface_temperature_c - ambient,
            ]
        )
        next_relative_temperature = (
            self.thermal_ad @ relative_temperature
            + self.thermal_bd
            * self.thermal["heat_gain"]
            * electrical_loss_w
        )
        next_state = ReducedState(
            soc=soc_next,
            polarization_fast_v=float(v1_next),
            polarization_slow_v=float(v2_next),
            core_temperature_c=float(next_relative_temperature[0] + ambient),
            surface_temperature_c=float(next_relative_temperature[1] + ambient),
            previous_current_a=current_a,
        )
        average_end = self.average_temperature(next_state)
        output = StepOutput(
            terminal_voltage_v=float(voltage_end),
            constraint_voltage_v=float(max(voltage_start, voltage_end)),
            average_temperature_c=float(average_end),
            # 当前温度 T_k 是已经发生、无法被本次控制改变的状态；离散 MPC
            # 应约束可由当前动作影响的下一状态 T_{k+1}。若 T_k 因模型误差略高于
            # 收紧边界，允许零/低电流在下一步把它带回，而不是宣布整个问题无解。
            constraint_temperature_c=float(average_end),
            electrical_loss_w=float(electrical_loss_w),
        )
        return next_state, output

    def predict(self, state: ReducedState, currents_a: np.ndarray) -> dict[str, np.ndarray]:
        """沿给定电流序列预测全部状态和约束输出。"""
        currents_a = np.asarray(currents_a, dtype=float)
        soc = np.empty(currents_a.size)
        voltage = np.empty(currents_a.size)
        temperature = np.empty(currents_a.size)
        running_state = state
        for index, current_a in enumerate(currents_a):
            running_state, output = self.step(running_state, float(current_a))
            soc[index] = running_state.soc
            voltage[index] = output.constraint_voltage_v
            temperature[index] = output.constraint_temperature_c
        return {"soc": soc, "voltage_v": voltage, "temperature_c": temperature}


class ConstrainedMPC:
    """用 SLSQP 求解带电压、电流、平均温度和电流变化率约束的 MPC。"""

    def __init__(self, model: ReducedBatteryModel, config: PhaseThreeConfig) -> None:
        self.model = model
        self.config = config
        self.number_of_blocks = config.control.number_of_control_blocks
        self._warm_start: np.ndarray | None = None

    @property
    def last_optimal_block_currents_a(self) -> np.ndarray | None:
        """返回上一求解保存的控制块副本，供只读控制状态审计使用。"""
        return None if self._warm_start is None else self._warm_start.copy()

    def _expand_blocks(self, block_currents_a: np.ndarray) -> np.ndarray:
        """把较少的分块决策变量展开为每 5 s 一个预测值。"""
        return np.repeat(
            np.asarray(block_currents_a, dtype=float),
            self.config.control.control_block_steps,
        )[: self.config.control.prediction_horizon_steps]

    def _predict(
        self, state: ReducedState, expanded_currents_a: np.ndarray
    ) -> dict[str, np.ndarray]:
        """预测候选电流；子类可覆盖终端处理而复用同一求解器。"""
        return self.model.predict(state, expanded_currents_a)

    def _initial_guess(self, state: ReducedState) -> np.ndarray:
        """使用上次解热启动；首次求解则按允许的变化率逐块升流。"""
        maximum = self.config.constraints.maximum_current_a
        delta = self.config.constraints.maximum_current_change_a_per_step
        if self._warm_start is not None:
            guess = self._warm_start.copy()
            guess[0] = np.clip(guess[0], state.previous_current_a - delta, state.previous_current_a + delta)
            return np.clip(guess, 0.0, maximum)

        guess = np.empty(self.number_of_blocks, dtype=float)
        previous = state.previous_current_a
        for index in range(self.number_of_blocks):
            previous = min(previous + delta, maximum)
            guess[index] = previous
        return guess

    def _constraint_margins(
        self,
        state: ReducedState,
        block_currents_a: np.ndarray,
        prediction: dict[str, np.ndarray],
    ) -> np.ndarray:
        """统一返回“正数表示满足”的全部不等式余量。"""
        constraints = self.config.constraints
        current_differences = np.diff(
            np.concatenate([[state.previous_current_a], block_currents_a])
        )
        return np.concatenate(
            [
                constraints.mpc_maximum_voltage_v - prediction["voltage_v"],
                constraints.mpc_maximum_temperature_c
                - prediction["temperature_c"],
                self.config.battery.target_soc - prediction["soc"],
                constraints.maximum_current_change_a_per_step - current_differences,
                constraints.maximum_current_change_a_per_step + current_differences,
            ]
        )

    def _constraint_slacks(
        self, state: ReducedState, block_currents_a: np.ndarray,
        prediction: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """Return minimum relaxation required by each prediction-domain constraint."""
        constraints = self.config.constraints
        differences = np.diff(np.concatenate([[state.previous_current_a], block_currents_a]))
        return {
            "slack_voltage_v": float(np.maximum(prediction["voltage_v"] - constraints.mpc_maximum_voltage_v, 0.0).max(initial=0.0)),
            "slack_temperature_c": float(np.maximum(prediction["temperature_c"] - constraints.mpc_maximum_temperature_c, 0.0).max(initial=0.0)),
            "slack_soc": float(np.maximum(prediction["soc"] - self.config.battery.target_soc, 0.0).max(initial=0.0)),
            "slack_current_change_a": float(np.maximum(np.abs(differences) - constraints.maximum_current_change_a_per_step, 0.0).max(initial=0.0)),
        }

    def _braking_demand(self, state: ReducedState) -> tuple[int, float]:
        """Estimate steps and current deficit needed to enter the one-step hard-safe set."""
        limit = self.config.constraints.physical_maximum_voltage_v
        temperature_limit = self.config.constraints.physical_maximum_temperature_c
        safe_currents = np.linspace(0.0, self.config.constraints.maximum_current_a, 201)
        safe = [
            current for current in safe_currents
            if (lambda output: output.terminal_voltage_v <= limit + self.config.validation.physical_constraint_tolerance
                and output.average_temperature_c <= temperature_limit + self.config.validation.physical_constraint_tolerance)(self.model.step(state, float(current))[1])
        ]
        safe_current = max(safe, default=0.0)
        deficit = max(0.0, state.previous_current_a - safe_current)
        delta = self.config.constraints.maximum_current_change_a_per_step
        return int(np.ceil(deficit / delta)) if delta > 0 else 0, max(0.0, deficit - delta)

    def _objective_value(
        self,
        state: ReducedState,
        block_currents_a: np.ndarray,
        prediction: dict[str, np.ndarray],
    ) -> float:
        """计算基础SOC推进、终端SOC和电流平滑代价。

        单独保留该方法，使后续教师可以增加有物理含义的终端/参考代价，
        而不复制求解器、约束和回退逻辑。
        """
        config = self.config
        maximum_current = config.constraints.maximum_current_a
        soc_gap = np.maximum(
            config.battery.target_soc - prediction["soc"], 0.0
        )
        current_differences = np.diff(
            np.concatenate([[state.previous_current_a], block_currents_a])
        )
        return float(
            config.objective.soc_tracking_weight * np.mean(soc_gap)
            + config.objective.terminal_soc_weight * soc_gap[-1] ** 2
            + config.objective.current_change_weight
            * np.mean((current_differences / maximum_current) ** 2)
        )

    def _safe_one_step_current(self, state: ReducedState) -> float:
        """优化预测不可行时使用的保守单步回退。

        回退时物理安全优先于电流变化率，所以必要时允许比正常约束更快地降流。
        """
        upper = min(
            self.config.constraints.maximum_current_a,
            state.previous_current_a
            + self.config.constraints.maximum_current_change_a_per_step,
        )
        target = self.config.battery.target_soc
        voltage_limit = self.config.constraints.mpc_maximum_voltage_v
        temperature_limit = self.config.constraints.mpc_maximum_temperature_c
        for current_a in np.linspace(upper, 0.0, 101):
            next_state, output = self.model.step(state, float(current_a))
            if (
                next_state.soc <= target + self.config.optimizer.constraint_tolerance
                and output.constraint_voltage_v
                <= voltage_limit + self.config.optimizer.constraint_tolerance
                and output.constraint_temperature_c
                <= temperature_limit + self.config.optimizer.constraint_tolerance
            ):
                return float(current_a)
        return 0.0

    def solve(self, state: ReducedState) -> MPCResult:
        """求解当前状态下的 MPC，并只返回第一控制动作。"""
        config = self.config
        guess = self._initial_guess(state)
        maximum_current = config.constraints.maximum_current_a
        cache_x: np.ndarray | None = None
        cache_prediction: dict[str, np.ndarray] | None = None

        def evaluate(block_currents_a: np.ndarray) -> dict[str, np.ndarray]:
            nonlocal cache_x, cache_prediction
            values = np.asarray(block_currents_a, dtype=float)
            if cache_x is None or not np.array_equal(values, cache_x):
                cache_x = values.copy()
                cache_prediction = self._predict(
                    state, self._expand_blocks(values)
                )
            assert cache_prediction is not None
            return cache_prediction

        def objective(block_currents_a: np.ndarray) -> float:
            prediction = evaluate(block_currents_a)
            return self._objective_value(
                state,
                np.asarray(block_currents_a, dtype=float),
                prediction,
            )

        def inequality(block_currents_a: np.ndarray) -> np.ndarray:
            prediction = evaluate(block_currents_a)
            return self._constraint_margins(
                state, np.asarray(block_currents_a), prediction
            )

        start = perf_counter()
        result = minimize(
            objective,
            guess,
            method="SLSQP",
            bounds=[(0.0, maximum_current)] * self.number_of_blocks,
            constraints=[{"type": "ineq", "fun": inequality}],
            options={
                "maxiter": config.optimizer.maximum_iterations,
                "ftol": config.optimizer.function_tolerance,
                "disp": False,
            },
        )
        solve_time_s = perf_counter() - start
        block_currents = np.asarray(result.x, dtype=float)
        prediction = self._predict(state, self._expand_blocks(block_currents))
        margins = self._constraint_margins(state, block_currents, prediction)
        minimum_margin = float(np.min(margins))
        feasible = minimum_margin >= -config.optimizer.constraint_tolerance
        slacks = self._constraint_slacks(state, block_currents, prediction)
        braking_steps, braking_deficit = self._braking_demand(state)

        used_fallback = not feasible
        if used_fallback:
            current_a = self._safe_one_step_current(state)
            self._warm_start = np.full(self.number_of_blocks, current_a)
            status = f"fallback_after_{result.status}: {result.message}"
        else:
            current_a = float(block_currents[0])
            self._warm_start = block_currents.copy()
            status = str(result.message)

        return MPCResult(
            current_a=current_a,
            optimizer_success=bool(result.success),
            prediction_feasible=bool(feasible),
            used_fallback=used_fallback,
            status=status,
            objective_value=float(result.fun),
            solve_time_s=float(solve_time_s),
            predicted_maximum_voltage_v=float(np.max(prediction["voltage_v"])),
            predicted_maximum_temperature_c=float(
                np.max(prediction["temperature_c"])
            ),
            predicted_terminal_soc=float(prediction["soc"][-1]),
            minimum_constraint_margin=minimum_margin,
            **slacks,
            braking_distance_steps=braking_steps,
            braking_current_deficit_a=braking_deficit,
        )
