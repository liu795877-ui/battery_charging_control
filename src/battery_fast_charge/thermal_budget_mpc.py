"""用状态触发热预算参考改进短视野MPC教师。"""

from __future__ import annotations

import numpy as np

from .mpc import ConstrainedMPC, ReducedState
from .phase4b_config import PhaseFourBConfig


class ThermalBudgetMPC(ConstrainedMPC):
    """在原约束MPC目标中加入可行最短时间参考跟踪项。

    参考不是固定时刻切换：预测SOC或平均温度任一达到预算阈值后，
    参考从峰值电流切换到可持续电流，因此控制器仍是状态反馈。
    """

    def __init__(self, model, phase3_config, phase4b_config: PhaseFourBConfig) -> None:
        super().__init__(model, phase3_config)
        self.phase4b_config = phase4b_config

    def _constraint_margins(
        self,
        state: ReducedState,
        block_currents_a: np.ndarray,
        prediction: dict[str, np.ndarray],
    ) -> np.ndarray:
        """保留电压、温度和变化率约束，移除预测路径SOC上限。

        实际执行仍由 ``_cap_current_at_target`` 精确封顶。预测允许在到达目标后
        进入零电流段，避免为了让整个5分钟路径都低于80%而提前渐近降流。
        """
        margins = super()._constraint_margins(state, block_currents_a, prediction)
        horizon = self.config.control.prediction_horizon_steps
        return np.concatenate([margins[: 2 * horizon], margins[3 * horizon :]])

    def _predict(
        self, state: ReducedState, expanded_currents_a: np.ndarray
    ) -> dict[str, np.ndarray]:
        """到达80%后把预测有效电流置零，避免虚构目标后的约束。

        候选分块电流仍由优化器决定；每个预测步只施加补足80%所需的部分，
        到达后后续步为零电流。这样预测终止条件与实际闭环封顶规则一致。
        """
        currents = np.asarray(expanded_currents_a, dtype=float)
        soc = np.empty(currents.size)
        voltage = np.empty(currents.size)
        temperature = np.empty(currents.size)
        running_state = state
        capacity_ah = self.config.battery.nominal_capacity_ah
        dt = self.config.control.control_interval_s
        target = self.config.battery.target_soc
        for index, requested_current in enumerate(currents):
            remaining_ah = max(target - running_state.soc, 0.0) * capacity_ah
            target_cap = remaining_ah * 3600.0 / dt
            applied_current = min(float(requested_current), target_cap)
            running_state, output = self.model.step(
                running_state, applied_current
            )
            soc[index] = running_state.soc
            voltage[index] = output.constraint_voltage_v
            temperature[index] = output.constraint_temperature_c
        return {"soc": soc, "voltage_v": voltage, "temperature_c": temperature}

    def _objective_value(
        self,
        state: ReducedState,
        block_currents_a: np.ndarray,
        prediction: dict[str, np.ndarray],
    ) -> float:
        """基础快速充电目标加归一化电流参考跟踪代价。"""
        base = super()._objective_value(state, block_currents_a, prediction)
        reference = self.phase4b_config.thermal_budget_reference
        predicted_temperature = prediction["temperature_c"]
        high_budget = (
            (prediction["soc"] < reference.switch_soc)
            & (
                predicted_temperature
                < reference.switch_average_temperature_c
            )
        )
        reference_current = np.where(
            high_budget,
            reference.peak_current_a,
            reference.sustainable_current_a,
        )
        expanded_current = self._expand_blocks(block_currents_a)
        normalized_error = (
            expanded_current - reference_current
        ) / self.config.constraints.maximum_current_a
        # 参考只用于早期热预算分配；释放后由一步可行参考调节器负责中后段，
        # 避免5 min短视野优化持续低于已经验证可行的5 A参考。
        reference_active = prediction["soc"] < reference.reference_release_soc
        return float(
            base
            + reference.reference_tracking_weight
            * np.mean(np.where(reference_active, normalized_error**2, 0.0))
        )
