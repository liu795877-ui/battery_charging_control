"""论文 NDC 模型、混合初态设计与滚动时域 MPC 教师。"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize

from .paper_method import hammersley_points
from .phase6p0_config import PhaseSixPZeroConfig


@dataclass(frozen=True)
class MPCStepResult:
    current_a: float
    planned_currents_a: np.ndarray
    success: bool
    feasible: bool
    objective: float
    minimum_margin: float
    solve_time_s: float
    message: str


class NDCModel:
    """论文式 nonlinear double-capacitor 模型的 60 s 精确离散化。"""

    def __init__(self, config: PhaseSixPZeroConfig) -> None:
        self.config = config
        p = config.ndc
        denominator = p.bulk_resistance_ohm + p.surface_resistance_ohm
        if denominator <= 0.0:
            raise ValueError("Rb+Rs 必须为正。")
        self.a = np.array(
            [
                [-1.0 / (p.bulk_capacitance_f * denominator), 1.0 / (p.bulk_capacitance_f * denominator)],
                [1.0 / (p.surface_capacitance_f * denominator), -1.0 / (p.surface_capacitance_f * denominator)],
            ],
            dtype=float,
        )
        self.b = np.array(
            [
                p.surface_resistance_ohm / (p.bulk_capacitance_f * denominator),
                p.bulk_resistance_ohm / (p.surface_capacitance_f * denominator),
            ],
            dtype=float,
        )
        augmented = np.zeros((3, 3), dtype=float)
        augmented[:2, :2] = self.a
        augmented[:2, 2] = self.b
        discrete = expm(augmented * config.mpc.sample_period_s)
        self.ad = discrete[:2, :2]
        self.bd = discrete[:2, 2]

    def soc(self, state: np.ndarray) -> float:
        p = self.config.ndc
        vb, vs = np.asarray(state, dtype=float)
        return float((p.bulk_capacitance_f * vb + p.surface_capacitance_f * vs) / (p.bulk_capacitance_f + p.surface_capacitance_f))

    def open_circuit_voltage(self, surface_voltage_v: float) -> float:
        return float(np.polynomial.polynomial.polyval(surface_voltage_v, self.config.ndc.alpha))

    def series_resistance(self, soc: float) -> float:
        beta0, beta1, beta2 = self.config.ndc.beta
        # 表 2 给出 beta2=-10；采用 exp(beta2*(1-SOC)) 才与论文 3 A 轨迹和 4.2 V 约束量级一致。
        return float(beta0 + beta1 * np.exp(beta2 * (1.0 - soc)))

    def terminal_voltage(self, state: np.ndarray, current_a: float) -> float:
        return self.open_circuit_voltage(float(state[1])) + self.series_resistance(self.soc(state)) * current_a

    def step(self, state: np.ndarray, current_a: float) -> np.ndarray:
        return self.ad @ np.asarray(state, dtype=float) + self.bd * float(current_a)

    def rollout(self, state: np.ndarray, currents_a: np.ndarray) -> dict[str, np.ndarray]:
        running = np.asarray(state, dtype=float)
        states: list[np.ndarray] = []
        socs: list[float] = []
        terminals: list[float] = []
        for current in np.asarray(currents_a, dtype=float):
            running = self.step(running, float(current))
            states.append(running.copy())
            socs.append(self.soc(running))
            terminals.append(self.terminal_voltage(running, float(current)))
        return {"states": np.asarray(states), "soc": np.asarray(socs), "terminal_voltage_v": np.asarray(terminals)}

    def health_margin(self, state: np.ndarray) -> float:
        soc = self.soc(state)
        vb, vs = np.asarray(state, dtype=float)
        limit = self.config.mpc.health_slope * soc + self.config.mpc.health_intercept_v
        return float(limit - (vs - vb))


def _candidate_pool(count: int, offset: int = 0) -> np.ndarray:
    """构造足量 Hammersley 候选；offset 防止补点时重复。"""
    unit = hammersley_points(count + offset, 2)
    return unit[offset:]


def generate_training_initial_states(model: NDCModel) -> pd.DataFrame:
    """生成 324 个低差异可行点和 76 个全因子边界增强可行点。"""
    config = model.config
    low, high = config.data.state_bounds_v
    hamm_candidates = low + _candidate_pool(3000) * (high - low)
    hamm_feasible = [point for point in hamm_candidates if model.health_margin(point) >= 0.0]
    hammersley = np.asarray(hamm_feasible[: config.data.hammersley_count])
    if len(hammersley) != config.data.hammersley_count:
        raise RuntimeError("无法生成足量 Hammersley 可行初态。")

    # 论文未报告 76 个全因子点的各因子水平。使用 20x20 全因子候选，并优先保留最靠近
    # 可行域边界/顶点的 76 点，复现其“补偿 Hammersley 边界覆盖不足”的原始目的。
    axis = np.linspace(low, high, 20)
    factorial_pool = np.asarray([(vb, vs) for vb in axis for vs in axis], dtype=float)
    feasible_pool = np.asarray([point for point in factorial_pool if model.health_margin(point) >= 0.0])
    boundary_distance = np.minimum.reduce(
        [
            feasible_pool[:, 0] - low,
            high - feasible_pool[:, 0],
            feasible_pool[:, 1] - low,
            high - feasible_pool[:, 1],
            np.maximum([model.health_margin(point) for point in feasible_pool], 0.0),
        ]
    )
    order = np.lexsort((feasible_pool[:, 1], feasible_pool[:, 0], boundary_distance))
    factorial = feasible_pool[order[: config.data.factorial_count]]
    points = np.vstack([hammersley, factorial])
    methods = ["hammersley"] * len(hammersley) + ["boundary_factorial"] * len(factorial)
    frame = pd.DataFrame(points, columns=["bulk_voltage_v", "surface_voltage_v"])
    frame.insert(0, "sampling_method", methods)
    frame.insert(0, "initial_state_id", [f"ndc_train_{index:03d}" for index in range(len(frame))])
    frame["initial_soc"] = [model.soc(row) for row in points]
    frame["initial_health_margin_v"] = [model.health_margin(row) for row in points]
    return frame


def generate_frozen_test_initial_states(model: NDCModel) -> pd.DataFrame:
    """冻结 30 个与训练设计分离的可行测试初态。"""
    config = model.config
    low, high = config.data.state_bounds_v
    anchors = np.asarray([[0.2, 0.2], [0.4, 0.4], [0.6, 0.6]], dtype=float)
    random = np.random.default_rng(config.data.random_seed + 3000)
    candidates = random.uniform(low, high, size=(5000, 2))
    feasible = np.asarray([point for point in candidates if model.health_margin(point) >= 0.0])
    remaining = feasible[: config.data.independent_test_trajectories - len(anchors)]
    points = np.vstack([anchors, remaining])
    frame = pd.DataFrame(points, columns=["bulk_voltage_v", "surface_voltage_v"])
    frame.insert(0, "trajectory_id", [f"ndc_test_{index:02d}" for index in range(len(frame))])
    frame["initial_soc"] = [model.soc(row) for row in points]
    frame["initial_health_margin_v"] = [model.health_margin(row) for row in points]
    return frame


class NDCMPC:
    """Np=10、Nu=2、Nc=1 的论文 NDC 滚动时域控制器。"""

    def __init__(self, model: NDCModel) -> None:
        self.model = model
        self.config = model.config

    def expand_controls(self, moves: np.ndarray) -> np.ndarray:
        moves = np.asarray(moves, dtype=float)
        # 论文采用 CVP 且 Nu=2：把 Np=10 的预测窗划成两个等长常值控制块。
        block_length = int(np.ceil(self.config.mpc.prediction_horizon / len(moves)))
        return np.repeat(moves, block_length)[: self.config.mpc.prediction_horizon]

    def _objective(self, state: np.ndarray, previous_current_a: float, moves: np.ndarray) -> float:
        prediction = self.model.rollout(state, self.expand_controls(moves))
        tracking = prediction["soc"] - self.config.mpc.target_soc
        # 论文显式控制律只有 (Vs,Vb) 两个输入。若把历史电流并入首个增量，标签将不再是
        # 这两个状态的单值函数；因此 R 项只惩罚 CVP 预测块之间的变化。
        increments = np.diff(np.asarray(moves, dtype=float))
        return float(self.config.mpc.soc_tracking_weight * np.sum(tracking**2) + self.config.mpc.current_increment_weight * np.sum(increments**2))

    def _margins(self, state: np.ndarray, moves: np.ndarray) -> np.ndarray:
        first_current = float(moves[0])
        next_state = self.model.step(state, first_current)
        return np.asarray(
            [
                self.config.mpc.surface_voltage_max_v - next_state[1],
                self.config.mpc.terminal_voltage_max_v - self.model.terminal_voltage(next_state, first_current),
                self.model.health_margin(next_state),
            ],
            dtype=float,
        )

    def solve(self, state: np.ndarray, previous_current_a: float, warm_start: np.ndarray | None = None) -> MPCStepResult:
        lower, upper = self.config.mpc.current_bounds_a
        starts = []
        if warm_start is not None:
            starts.append(np.clip(np.asarray(warm_start, dtype=float), lower, upper))
        starts.extend([np.full(2, upper), np.full(2, previous_current_a), np.full(2, 0.5 * (lower + upper)), np.full(2, lower)])
        best = None
        start_time = perf_counter()
        for guess in starts:
            result = minimize(
                lambda moves: self._objective(state, previous_current_a, moves),
                guess,
                method="SLSQP",
                bounds=[(lower, upper)] * self.config.mpc.control_horizon,
                constraints=[{"type": "ineq", "fun": lambda moves: self._margins(state, moves)}],
                options={"maxiter": self.config.mpc.optimizer_max_iterations, "ftol": self.config.mpc.optimizer_ftol, "disp": False},
            )
            moves = np.asarray(result.x, dtype=float)
            margin = float(np.min(self._margins(state, moves)))
            feasible = margin >= -self.config.mpc.constraint_tolerance
            candidate = (not feasible, float(result.fun), result, moves, margin)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
            if result.success and feasible:
                break
        assert best is not None
        _, _, result, moves, margin = best
        feasible = margin >= -self.config.mpc.constraint_tolerance
        return MPCStepResult(
            current_a=float(moves[0]),
            planned_currents_a=self.expand_controls(moves),
            success=bool(result.success),
            feasible=bool(feasible),
            objective=float(result.fun),
            minimum_margin=margin,
            solve_time_s=perf_counter() - start_time,
            message=str(result.message),
        )


def rollout_mpc_trajectory(
    controller: NDCMPC,
    initial_state: np.ndarray,
    steps: int,
    trajectory_id: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """逐周期重新求解 MPC；每一行保存当前状态与下一动作标签。"""
    model = controller.model
    state = np.asarray(initial_state, dtype=float)
    previous = controller.config.mpc.initial_previous_current_a
    warm_start = None
    rows: list[dict[str, object]] = []
    statuses: list[bool] = []
    for step_index in range(steps):
        solve = controller.solve(state, previous, warm_start)
        statuses.append(solve.success and solve.feasible)
        if not solve.feasible:
            break
        current = solve.current_a
        next_state = model.step(state, current)
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "step_index": step_index,
                "bulk_voltage_v": state[0],
                "surface_voltage_v": state[1],
                "soc": model.soc(state),
                "previous_current_a": previous,
                "teacher_current_a": current,
                "next_bulk_voltage_v": next_state[0],
                "next_surface_voltage_v": next_state[1],
                "next_soc": model.soc(next_state),
                "next_terminal_voltage_v": model.terminal_voltage(next_state, current),
                "next_health_margin_v": model.health_margin(next_state),
                "mpc_objective": solve.objective,
                "mpc_minimum_margin": solve.minimum_margin,
                "mpc_solve_time_s": solve.solve_time_s,
                "mpc_success": solve.success,
            }
        )
        state = next_state
        previous = current
        warm_start = np.asarray([solve.planned_currents_a[1], solve.planned_currents_a[-1]])
    frame = pd.DataFrame.from_records(rows)
    audit = {
        "trajectory_id": trajectory_id,
        "requested_steps": steps,
        "completed_steps": len(frame),
        "complete": len(frame) == steps,
        "all_solver_steps_feasible": bool(statuses and all(statuses)),
    }
    return frame, audit
