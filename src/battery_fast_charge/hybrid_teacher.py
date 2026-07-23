"""热预算MPC与终端参考调节器组成的混合最短时间教师。"""

from __future__ import annotations

from dataclasses import dataclass

from .mpc import ReducedBatteryModel, ReducedState
from .phase3_config import PhaseThreeConfig
from .phase4b_config import PhaseFourBConfig
from .teacher_data import filter_feasible_current
from .thermal_budget_mpc import ThermalBudgetMPC


@dataclass(frozen=True)
class HybridTeacherDecision:
    """统一记录MPC模式和终端参考调节模式的动作诊断。"""

    current_a: float
    mode: str
    optimizer_success: bool
    prediction_feasible: bool
    used_fallback: bool
    safety_override: bool
    solve_time_s: float
    predicted_maximum_voltage_v: float
    predicted_maximum_temperature_c: float


class HybridMinimumTimeTeacher:
    """低SOC用热预算MPC，高SOC用一步可行终端策略。"""

    def __init__(
        self,
        model: ReducedBatteryModel,
        phase3: PhaseThreeConfig,
        phase4b: PhaseFourBConfig,
    ) -> None:
        self.model = model
        self.phase3 = phase3
        self.phase4b = phase4b
        self.mpc = ThermalBudgetMPC(model, phase3, phase4b)

    def decide(self, state: ReducedState) -> HybridTeacherDecision:
        """根据当前SOC选择MPC或可行终端控制，不使用固定时间切换。"""
        reference = self.phase4b.thermal_budget_reference
        if (
            state.soc < reference.switch_soc
            and state.previous_current_a
            < reference.peak_current_a - 1.0e-8
        ):
            # 0 A初态与8 A参考之间相差四个正常变化步。这里明确使用同一
            # 一步可行过滤器完成2→4→6→8 A启动，不把首步SLSQP回退伪装
            # 成教师优化成功。
            filtered = filter_feasible_current(
                self.model,
                state,
                reference.peak_current_a,
                self.phase3,
            )
            return HybridTeacherDecision(
                current_a=filtered.current_a,
                mode="startup_reference_governor",
                optimizer_success=True,
                prediction_feasible=True,
                used_fallback=False,
                safety_override=filtered.safety_override,
                solve_time_s=0.0,
                predicted_maximum_voltage_v=filtered.next_voltage_v,
                predicted_maximum_temperature_c=filtered.next_temperature_c,
            )
        if state.soc < reference.reference_release_soc:
            result = self.mpc.solve(state)
            return HybridTeacherDecision(
                current_a=result.current_a,
                mode="thermal_budget_mpc",
                optimizer_success=result.optimizer_success,
                prediction_feasible=result.prediction_feasible,
                used_fallback=result.used_fallback,
                safety_override=False,
                solve_time_s=result.solve_time_s,
                predicted_maximum_voltage_v=result.predicted_maximum_voltage_v,
                predicted_maximum_temperature_c=result.predicted_maximum_temperature_c,
            )

        filtered = filter_feasible_current(
            self.model,
            state,
            reference.sustainable_current_a,
            self.phase3,
        )
        return HybridTeacherDecision(
            current_a=filtered.current_a,
            mode="terminal_reference_governor",
            optimizer_success=True,
            prediction_feasible=True,
            used_fallback=False,
            safety_override=filtered.safety_override,
            solve_time_s=0.0,
            predicted_maximum_voltage_v=filtered.next_voltage_v,
            predicted_maximum_temperature_c=filtered.next_temperature_c,
        )
