"""Recoverable MPC with slew-safe alternatives and explicit failure taxonomy."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .ann_model import TinyANN
from .mpc import ConstrainedMPC, ReducedBatteryModel, ReducedState
from .phase3_config import PhaseThreeConfig
from .phase6_closed_loop import pure_dnn_features


FAILURE_NONE = "none"
FAILURE_NUMERICAL_RECOVERED = "numerical_optimization_failure_feasible_alternative"
FAILURE_PREDICTION_INFEASIBLE = "prediction_domain_infeasible_under_candidate_audit"
FAILURE_HARD_SAFETY_SLEW_CONFLICT = "hard_safety_slew_conflict"


@dataclass(frozen=True)
class RecoveryMPCResult:
    current_a: float
    source: str
    failure_type: str
    optimizer_success: bool
    optimizer_prediction_feasible: bool
    selected_sequence_feasible: bool
    used_emergency_fallback: bool
    hard_safety_slew_conflict: bool
    status: str
    solve_time_s: float
    predicted_maximum_voltage_v: float
    predicted_maximum_temperature_c: float
    predicted_terminal_soc: float
    minimum_constraint_margin: float
    candidate_feasibility: dict[str, bool]


def slew_safe_interval(
    previous_current_a: float, maximum_current_a: float = 10.0, maximum_change_a: float = 2.0
) -> tuple[float, float]:
    return (
        max(0.0, float(previous_current_a) - maximum_change_a),
        min(maximum_current_a, float(previous_current_a) + maximum_change_a),
    )


def project_current_to_slew_interval(
    requested_current_a: float, previous_current_a: float, maximum_current_a: float = 10.0,
    maximum_change_a: float = 2.0,
) -> float:
    lower, upper = slew_safe_interval(previous_current_a, maximum_current_a, maximum_change_a)
    return float(np.clip(requested_current_a, lower, upper))


class RecoverableConstrainedMPC(ConstrainedMPC):
    """SLSQP MPC that retains feasible sequences before using emergency fallback."""

    def __init__(
        self,
        model: ReducedBatteryModel,
        config: PhaseThreeConfig,
        ann: TinyANN,
        scan_points: int = 101,
    ) -> None:
        super().__init__(model, config)
        self.ann = ann
        self.scan_points = scan_points
        self._last_feasible_expanded: np.ndarray | None = None

    def _audit_sequence(
        self, state: ReducedState, blocks: np.ndarray
    ) -> tuple[bool, dict[str, np.ndarray], float]:
        prediction = self._predict(state, self._expand_blocks(blocks))
        margins = self._constraint_margins(state, blocks, prediction)
        minimum = float(np.min(margins))
        return minimum >= -self.config.optimizer.constraint_tolerance, prediction, minimum

    def _shifted_previous(self, elapsed_steps: int) -> np.ndarray | None:
        if self._last_feasible_expanded is None:
            return None
        shift = max(0, min(int(elapsed_steps), len(self._last_feasible_expanded)))
        expanded = np.concatenate(
            [self._last_feasible_expanded[shift:], np.repeat(self._last_feasible_expanded[-1], shift)]
        )
        return expanded[:: self.config.control.control_block_steps][: self.number_of_blocks]

    def _projected_ann_blocks(self, state: ReducedState) -> np.ndarray:
        running = state
        blocks: list[float] = []
        for _ in range(self.number_of_blocks):
            raw = float(self.ann.predict_unclipped(pure_dnn_features(self.model, running)))
            current = project_current_to_slew_interval(
                raw,
                running.previous_current_a,
                self.config.constraints.maximum_current_a,
                self.config.constraints.maximum_current_change_a_per_step,
            )
            blocks.append(current)
            for _ in range(self.config.control.control_block_steps):
                running, _ = self.model.step(running, current)
        return np.asarray(blocks, dtype=float)

    def _conservative_slew_down(self, state: ReducedState) -> np.ndarray:
        previous = state.previous_current_a
        blocks = []
        for _ in range(self.number_of_blocks):
            previous = max(0.0, previous - self.config.constraints.maximum_current_change_a_per_step)
            blocks.append(previous)
        return np.asarray(blocks, dtype=float)

    def _one_step_hard_safe(self, state: ReducedState, current_a: float) -> bool:
        _, output = self.model.step(state, current_a)
        tolerance = self.config.validation.physical_constraint_tolerance
        return bool(
            output.terminal_voltage_v <= self.config.constraints.physical_maximum_voltage_v + tolerance
            and output.average_temperature_c <= self.config.constraints.physical_maximum_temperature_c + tolerance
        )

    def _emergency_current(self, state: ReducedState) -> tuple[float, bool, str]:
        lower, upper = slew_safe_interval(
            state.previous_current_a,
            self.config.constraints.maximum_current_a,
            self.config.constraints.maximum_current_change_a_per_step,
        )
        for current in np.linspace(upper, lower, self.scan_points):
            if self._one_step_hard_safe(state, float(current)):
                return float(current), False, "slope_safe_emergency"
        if lower > 0.0:
            for current in np.linspace(lower, 0.0, self.scan_points)[1:]:
                if self._one_step_hard_safe(state, float(current)):
                    return float(current), True, "hard_safety_emergency"
        return 0.0, True, "hard_safety_emergency"

    def solve_with_recovery(self, state: ReducedState, elapsed_steps: int = 0) -> RecoveryMPCResult:
        shifted = self._shifted_previous(elapsed_steps)
        guess = shifted.copy() if shifted is not None else self._initial_guess(state)
        cache_x: np.ndarray | None = None
        cache_prediction: dict[str, np.ndarray] | None = None

        def evaluate(blocks: np.ndarray) -> dict[str, np.ndarray]:
            nonlocal cache_x, cache_prediction
            values = np.asarray(blocks, dtype=float)
            if cache_x is None or not np.array_equal(values, cache_x):
                cache_x = values.copy()
                cache_prediction = self._predict(state, self._expand_blocks(values))
            assert cache_prediction is not None
            return cache_prediction

        def objective(blocks: np.ndarray) -> float:
            values = np.asarray(blocks, dtype=float)
            return self._objective_value(state, values, evaluate(values))

        def inequality(blocks: np.ndarray) -> np.ndarray:
            values = np.asarray(blocks, dtype=float)
            return self._constraint_margins(state, values, evaluate(values))

        started = perf_counter()
        optimization = minimize(
            objective,
            guess,
            method="SLSQP",
            bounds=[(0.0, self.config.constraints.maximum_current_a)] * self.number_of_blocks,
            constraints=[{"type": "ineq", "fun": inequality}],
            options={
                "maxiter": self.config.optimizer.maximum_iterations,
                "ftol": self.config.optimizer.function_tolerance,
                "disp": False,
            },
        )
        solve_time = perf_counter() - started
        optimized = np.asarray(optimization.x, dtype=float)
        optimized_feasible, optimized_prediction, optimized_margin = self._audit_sequence(state, optimized)
        candidates: list[tuple[str, np.ndarray | None]] = [
            ("shifted_previous_feasible", shifted),
            ("projected_ann_sequence", self._projected_ann_blocks(state)),
            ("conservative_slew_down", self._conservative_slew_down(state)),
        ]
        audits: dict[str, bool] = {}
        audited: dict[str, tuple[np.ndarray, dict[str, np.ndarray], float]] = {}
        for name, blocks in candidates:
            if blocks is None:
                audits[name] = False
                continue
            feasible, prediction, margin = self._audit_sequence(state, blocks)
            audits[name] = feasible
            audited[name] = (blocks, prediction, margin)

        if bool(optimization.success) and optimized_feasible:
            source = "slsqp"
            failure = FAILURE_NONE
            selected = optimized
            prediction = optimized_prediction
            margin = optimized_margin
            selected_feasible = True
            conflict = False
            emergency = False
            status = str(optimization.message)
        else:
            available = next((name for name, _ in candidates if audits.get(name, False)), None)
            if available is not None:
                source = available
                failure = FAILURE_NUMERICAL_RECOVERED
                selected, prediction, margin = audited[available]
                selected_feasible = True
                conflict = False
                emergency = False
                status = f"{optimization.message}; recovered_with_{available}"
            else:
                current, conflict, source = self._emergency_current(state)
                next_state, output = self.model.step(state, current)
                prediction = {
                    "voltage_v": np.asarray([output.terminal_voltage_v]),
                    "temperature_c": np.asarray([output.average_temperature_c]),
                    "soc": np.asarray([next_state.soc]),
                }
                selected = np.full(self.number_of_blocks, current)
                margin = float("nan")
                selected_feasible = False
                emergency = True
                failure = (
                    FAILURE_HARD_SAFETY_SLEW_CONFLICT
                    if conflict
                    else FAILURE_PREDICTION_INFEASIBLE
                )
                status = f"{optimization.message}; {failure}"

        current_a = float(selected[0])
        if selected_feasible:
            self._last_feasible_expanded = self._expand_blocks(selected).copy()
            self._warm_start = selected.copy()
        else:
            self._warm_start = np.full(self.number_of_blocks, current_a)
        audits["slsqp"] = bool(optimization.success and optimized_feasible)
        return RecoveryMPCResult(
            current_a=current_a,
            source=source,
            failure_type=failure,
            optimizer_success=bool(optimization.success),
            optimizer_prediction_feasible=optimized_feasible,
            selected_sequence_feasible=selected_feasible,
            used_emergency_fallback=emergency,
            hard_safety_slew_conflict=conflict,
            status=status,
            solve_time_s=solve_time,
            predicted_maximum_voltage_v=float(np.max(prediction["voltage_v"])),
            predicted_maximum_temperature_c=float(np.max(prediction["temperature_c"])),
            predicted_terminal_soc=float(prediction["soc"][-1]),
            minimum_constraint_margin=margin,
            candidate_feasibility=audits,
        )
