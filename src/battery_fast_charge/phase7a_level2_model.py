"""Phase 7A Level 2 的 2RC 三状态模型和电流/电压约束 MPC。"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .phase7a_level2_config import Phase7ALevel2Config


@dataclass(frozen=True)
class Level2State:
    soc: float
    polarization_1_v: float
    polarization_2_v: float


@dataclass(frozen=True)
class Level2MPCResult:
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


class Level2Model:
    def __init__(self, config: Phase7ALevel2Config, inherited: Any, project_root: str | Path):
        self.config = config
        self.inherited = inherited
        curve = pd.read_csv(Path(project_root) / config.model.ocv_curve).sort_values("soc")
        self._soc = curve.soc.to_numpy(float); self._ocv = curve.ocv_v.to_numpy(float)
        self.decay_1 = float(np.exp(-config.model.sample_period_s / config.model.tau1_s))
        self.decay_2 = float(np.exp(-config.model.sample_period_s / config.model.tau2_s))

    def ocv(self, soc: float | np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(soc, dtype=float), self._soc, self._ocv)

    def terminal_voltage(self, state: Level2State, current_a: float) -> float:
        return float(self.ocv(state.soc) + self.config.model.r0_ohm * current_a + state.polarization_1_v + state.polarization_2_v)

    def step(self, state: Level2State, current_a: float) -> Level2State:
        m = self.config.model
        return Level2State(
            soc=state.soc + current_a * m.sample_period_s / (3600.0 * m.nominal_capacity_ah),
            polarization_1_v=self.decay_1 * state.polarization_1_v + m.r1_ohm * (1.0 - self.decay_1) * current_a,
            polarization_2_v=self.decay_2 * state.polarization_2_v + m.r2_ohm * (1.0 - self.decay_2) * current_a,
        )

    def rollout(self, state: Level2State, block_currents_a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        currents = np.repeat(np.asarray(block_currents_a, dtype=float), self.inherited.mpc.control_block_steps)
        soc, voltage = [], []
        current_state = state
        for current in currents:
            current_state = self.step(current_state, float(current))
            soc.append(current_state.soc); voltage.append(self.terminal_voltage(current_state, float(current)))
        return np.asarray(soc), np.asarray(voltage), currents


class Level2MPC:
    def __init__(self, model: Level2Model):
        self.model = model; self.config = model.inherited; self._warm_start: np.ndarray | None = None

    @property
    def number_of_blocks(self) -> int:
        return self.config.mpc.prediction_horizon_steps // self.config.mpc.control_block_steps

    def set_warm_start(self, values: np.ndarray | None) -> None:
        self._warm_start = None if values is None else np.asarray(values, dtype=float).copy()

    def _default_start(self, state: Level2State) -> np.ndarray:
        lower, upper = self.config.mpc.current_bounds_a
        headroom = self.config.mpc.terminal_voltage_max_v - float(self.model.ocv(state.soc)) - state.polarization_1_v - state.polarization_2_v
        resistance = self.model.config.model.r0_ohm + self.model.config.model.r1_ohm + self.model.config.model.r2_ohm
        return np.full(self.number_of_blocks, float(np.clip(headroom / resistance, lower, upper)))

    def solve(self, state: Level2State) -> Level2MPCResult:
        cfg = self.config.mpc
        initial = self._default_start(state) if self._warm_start is None else self._warm_start
        if initial.shape != (self.number_of_blocks,): raise ValueError("warm start 控制块维数不匹配。")
        initial = np.clip(initial, *cfg.current_bounds_a)

        def objective(values: np.ndarray) -> float:
            soc, _, _ = self.model.rollout(state, values)
            return float(cfg.soc_tracking_weight * np.sum((cfg.target_soc - soc) ** 2)
                         + cfg.terminal_soc_weight * (cfg.target_soc - soc[-1]) ** 2
                         + cfg.current_smoothness_weight * np.sum(np.diff(values) ** 2))

        def constraints(values: np.ndarray) -> np.ndarray:
            return cfg.terminal_voltage_max_v - self.model.rollout(state, values)[1]

        def optimize(start: np.ndarray):
            return minimize(objective, start, method="SLSQP", bounds=[cfg.current_bounds_a] * self.number_of_blocks,
                            constraints={"type": "ineq", "fun": constraints},
                            options={"maxiter": cfg.optimizer_max_iterations, "ftol": cfg.optimizer_ftol, "disp": False})
        started = perf_counter(); result = optimize(initial)
        if not result.success and np.all(np.isfinite(result.x)):
            retry = optimize(np.clip(np.asarray(result.x, dtype=float), *cfg.current_bounds_a))
            if retry.success or retry.fun <= result.fun: result = retry
        elapsed = perf_counter() - started
        plan = np.asarray(result.x, dtype=float); voltage = self.model.rollout(state, plan)[1]
        margin = float(np.min(cfg.terminal_voltage_max_v - voltage))
        feasible = bool(np.all(plan >= cfg.current_bounds_a[0] - cfg.constraint_tolerance)
                        and np.all(plan <= cfg.current_bounds_a[1] + cfg.constraint_tolerance)
                        and margin >= -cfg.constraint_tolerance)
        return Level2MPCResult(float(plan[0]), plan, float(result.fun), bool(result.success), feasible, False,
                               str(result.message), elapsed, float(np.max(voltage)), margin)
