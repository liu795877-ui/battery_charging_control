"""Phase 7A Level 1：项目参数 1RC 模型与无斜率/温度约束 MPC。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .phase7a_level1_config import Phase7ALevel1Config


@dataclass(frozen=True)
class Level1State:
    soc: float
    polarization_v: float


@dataclass(frozen=True)
class Level1MPCResult:
    current_a: float
    plan_a: np.ndarray
    objective_value: float
    optimizer_success: bool
    prediction_feasible: bool
    used_fallback: bool
    status: str
    solve_time_s: float
    maximum_voltage_v: float
    minimum_constraint_margin: float


class Level1Model:
    def __init__(self, config: Phase7ALevel1Config, project_root: str | Path):
        self.config = config
        curve = pd.read_csv(Path(project_root) / config.model.ocv_curve).sort_values("soc")
        self._soc = curve["soc"].to_numpy(float)
        self._ocv = curve["ocv_v"].to_numpy(float)
        self.decay = float(np.exp(-config.model.sample_period_s / config.model.tau1_s))

    def ocv(self, soc: float | np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(soc, dtype=float), self._soc, self._ocv)

    def terminal_voltage(self, state: Level1State, current_a: float) -> float:
        return float(self.ocv(state.soc) + self.config.model.r0_ohm * current_a + state.polarization_v)

    def step(self, state: Level1State, current_a: float) -> Level1State:
        cfg = self.config.model
        return Level1State(
            soc=state.soc + current_a * cfg.sample_period_s / (3600.0 * cfg.nominal_capacity_ah),
            polarization_v=self.decay * state.polarization_v + cfg.r1_ohm * (1.0 - self.decay) * current_a,
        )

    def rollout(self, state: Level1State, block_currents_a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        currents = np.repeat(np.asarray(block_currents_a, dtype=float), self.config.mpc.control_block_steps)
        soc, voltage = [], []
        current_state = state
        for current in currents:
            current_state = self.step(current_state, float(current))
            soc.append(current_state.soc)
            voltage.append(self.terminal_voltage(current_state, float(current)))
        return np.asarray(soc), np.asarray(voltage), currents


class Level1MPC:
    def __init__(self, model: Level1Model):
        self.model = model
        self.config = model.config
        self._warm_start: np.ndarray | None = None

    @property
    def number_of_blocks(self) -> int:
        return self.config.mpc.prediction_horizon_steps // self.config.mpc.control_block_steps

    def set_warm_start(self, values: np.ndarray | None) -> None:
        self._warm_start = None if values is None else np.asarray(values, dtype=float).copy()

    def _default_start(self, state: Level1State) -> np.ndarray:
        lower, upper = self.config.mpc.current_bounds_a
        voltage_headroom = self.config.mpc.terminal_voltage_max_v - float(self.model.ocv(state.soc)) - state.polarization_v
        current = float(np.clip(voltage_headroom / (self.config.model.r0_ohm + self.config.model.r1_ohm), lower, upper))
        return np.full(self.number_of_blocks, current)

    def solve(self, state: Level1State) -> Level1MPCResult:
        cfg = self.config.mpc
        initial = self._default_start(state) if self._warm_start is None else self._warm_start
        if initial.shape != (self.number_of_blocks,):
            raise ValueError("warm start 控制块维数不匹配。")
        initial = np.clip(initial, *cfg.current_bounds_a)

        def objective(values: np.ndarray) -> float:
            soc, _, _ = self.model.rollout(state, values)
            tracking = cfg.soc_tracking_weight * np.sum((cfg.target_soc - soc) ** 2)
            terminal = cfg.terminal_soc_weight * (cfg.target_soc - soc[-1]) ** 2
            smoothness = cfg.current_smoothness_weight * np.sum(np.diff(values) ** 2)
            return float(tracking + terminal + smoothness)

        def constraints(values: np.ndarray) -> np.ndarray:
            _, voltage, _ = self.model.rollout(state, values)
            return cfg.terminal_voltage_max_v - voltage

        def optimize(start: np.ndarray):
            return minimize(
                objective, start, method="SLSQP",
                bounds=[cfg.current_bounds_a] * self.number_of_blocks,
                constraints={"type": "ineq", "fun": constraints},
                options={"maxiter": cfg.optimizer_max_iterations, "ftol": cfg.optimizer_ftol, "disp": False},
            )

        started = perf_counter()
        result = optimize(initial)
        # SLSQP 偶尔会在已接近可行边界时误报约束不相容；仍由同一目标和约束
        # 从当前有限解重启一次。该操作不是控制 fallback，失败仍按失败记录。
        if not result.success and np.all(np.isfinite(result.x)):
            retry = optimize(np.clip(np.asarray(result.x, dtype=float), *cfg.current_bounds_a))
            if retry.success or (not result.success and retry.fun <= result.fun):
                result = retry
        elapsed = perf_counter() - started
        plan = np.asarray(result.x, dtype=float)
        _, voltage, _ = self.model.rollout(state, plan)
        margin = float(np.min(cfg.terminal_voltage_max_v - voltage))
        feasible = bool(
            np.all(plan >= cfg.current_bounds_a[0] - cfg.constraint_tolerance)
            and np.all(plan <= cfg.current_bounds_a[1] + cfg.constraint_tolerance)
            and margin >= -cfg.constraint_tolerance
        )
        success = bool(result.success and feasible and np.isfinite(result.fun))
        return Level1MPCResult(
            current_a=float(plan[0]), plan_a=plan, objective_value=float(result.fun),
            optimizer_success=bool(result.success), prediction_feasible=feasible,
            used_fallback=False, status=str(result.message), solve_time_s=elapsed,
            maximum_voltage_v=float(np.max(voltage)), minimum_constraint_margin=margin,
        )
