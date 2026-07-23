"""从物理可达轨迹生成 MPC 教师候选状态和监督学习标签。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .mpc import ConstrainedMPC, ReducedBatteryModel, ReducedState
from .phase3_config import PhaseThreeConfig
from .phase3b_config import PhaseThreeBConfig, RolloutConfig


@dataclass(frozen=True)
class FilteredCurrent:
    """一步可行性过滤后的电流及安全诊断。"""

    current_a: float
    desired_current_a: float
    safety_override: bool
    next_voltage_v: float
    next_temperature_c: float
    next_soc: float


def state_average_temperature(model: ReducedBatteryModel, state: ReducedState) -> float:
    """数据特征只使用第二阶段已经验证的加权平均温度。"""
    return model.average_temperature(state)


def filter_feasible_current(
    model: ReducedBatteryModel,
    state: ReducedState,
    desired_current_a: float,
    config: PhaseThreeConfig,
) -> FilteredCurrent:
    """选择不超过期望值的一步最大可行电流。

    正常情况下先满足每5 s不超过2 A的电流变化；若最小正常降流仍会违反
    电压、温度或终端SOC，则允许安全优先地更快降流，并记录 safety_override。
    电压和温度关于当前充电电流在本研究范围内单调，因此可用二分搜索。
    """
    constraints = config.constraints
    desired = float(np.clip(desired_current_a, 0.0, constraints.maximum_current_a))
    slew_lower = max(
        0.0, state.previous_current_a - constraints.maximum_current_change_a_per_step
    )
    slew_upper = min(
        constraints.maximum_current_a,
        state.previous_current_a + constraints.maximum_current_change_a_per_step,
    )
    candidate_upper = float(np.clip(desired, slew_lower, slew_upper))

    def feasible(current_a: float) -> tuple[bool, ReducedState, Any]:
        next_state, output = model.step(state, current_a)
        is_feasible = bool(
            output.constraint_voltage_v
            <= constraints.mpc_maximum_voltage_v
            + config.optimizer.constraint_tolerance
            and output.constraint_temperature_c
            <= constraints.mpc_maximum_temperature_c
            + config.optimizer.constraint_tolerance
            and next_state.soc
            <= config.battery.target_soc + config.optimizer.constraint_tolerance
        )
        return is_feasible, next_state, output

    upper_feasible, upper_state, upper_output = feasible(candidate_upper)
    safety_override = False
    if upper_feasible:
        selected = candidate_upper
        next_state, output = upper_state, upper_output
    else:
        zero_feasible, zero_state, zero_output = feasible(0.0)
        if not zero_feasible:
            # 当前状态可能因收紧约束和模型误差略在边界外；零电流仍是最安全动作。
            selected = 0.0
            next_state, output = zero_state, zero_output
            safety_override = True
        else:
            low, high = 0.0, candidate_upper
            for _ in range(35):
                middle = 0.5 * (low + high)
                middle_feasible, _, _ = feasible(middle)
                if middle_feasible:
                    low = middle
                else:
                    high = middle
            selected = low
            _, next_state, output = feasible(selected)
            safety_override = selected < slew_lower - 1.0e-8

    return FilteredCurrent(
        current_a=float(selected),
        desired_current_a=desired,
        safety_override=safety_override,
        next_voltage_v=float(output.terminal_voltage_v),
        next_temperature_c=float(output.average_temperature_c),
        next_soc=float(next_state.soc),
    )


class ExplorationPolicy:
    """把配置中的五类期望电流策略转换为确定性时序信号。"""

    def __init__(self, rollout: RolloutConfig) -> None:
        self.rollout = rollout
        self.random = np.random.default_rng(
            int(rollout.parameters.get("seed", 0))
        )
        self.random_target_a: float | None = None

    def desired_current(self, step_index: int, time_s: float, soc: float) -> float:
        """返回尚未经过约束过滤的期望充电电流，单位A。"""
        parameters = self.rollout.parameters
        kind = self.rollout.kind
        if kind == "constant":
            return float(parameters["current_a"])
        if kind == "soc_switch":
            return float(
                parameters["low_current_a"]
                if soc < float(parameters["switch_soc"])
                else parameters["high_current_a"]
            )
        if kind == "pulse":
            half_period = float(parameters["period_s"]) / 2.0
            high = int(time_s // half_period) % 2 == 1
            return float(
                parameters["high_current_a"] if high else parameters["low_current_a"]
            )
        if kind == "sine":
            angle = 2.0 * np.pi * time_s / float(parameters["period_s"])
            return float(
                parameters["center_current_a"]
                + parameters["amplitude_a"] * np.sin(angle)
            )
        if kind == "random_blocks":
            block_steps = int(parameters["block_steps"])
            if self.random_target_a is None or step_index % block_steps == 0:
                self.random_target_a = float(
                    self.random.uniform(
                        float(parameters["minimum_current_a"]),
                        float(parameters["maximum_current_a"]),
                    )
                )
            return self.random_target_a
        raise ValueError(f"不支持的探索策略：{kind}")


def generate_reachable_rollout(
    model: ReducedBatteryModel,
    phase3: PhaseThreeConfig,
    rollout: RolloutConfig,
) -> pd.DataFrame:
    """从平衡初态逐步推进一条受约束轨迹，保存每个动作之前的状态。"""
    policy = ExplorationPolicy(rollout)
    state = ReducedState(
        soc=phase3.battery.initial_soc,
        polarization_fast_v=0.0,
        polarization_slow_v=0.0,
        core_temperature_c=phase3.battery.initial_temperature_c,
        surface_temperature_c=phase3.battery.initial_temperature_c,
        previous_current_a=0.0,
    )
    maximum_steps = int(
        np.ceil(
            phase3.control.maximum_simulation_time_s
            / phase3.control.control_interval_s
        )
    )
    records: list[dict[str, Any]] = []
    for step_index in range(maximum_steps):
        time_s = step_index * phase3.control.control_interval_s
        desired = policy.desired_current(step_index, time_s, state.soc)
        filtered = filter_feasible_current(model, state, desired, phase3)
        records.append(
            {
                "trajectory_id": rollout.name,
                "policy_kind": rollout.kind,
                "step_index": step_index,
                "time_s": time_s,
                "state_soc": state.soc,
                "state_polarization_fast_v": state.polarization_fast_v,
                "state_polarization_slow_v": state.polarization_slow_v,
                "state_average_temperature_c": state_average_temperature(
                    model, state
                ),
                # 两个潜在热状态仅用于复现教师内部状态，不作为第一版DNN特征。
                "audit_core_temperature_c": state.core_temperature_c,
                "audit_surface_temperature_c": state.surface_temperature_c,
                "state_previous_current_a": state.previous_current_a,
                "exploration_desired_current_a": filtered.desired_current_a,
                "exploration_applied_current_a": filtered.current_a,
                "exploration_safety_override": filtered.safety_override,
                "next_voltage_v": filtered.next_voltage_v,
                "next_average_temperature_c": filtered.next_temperature_c,
                "next_soc": filtered.next_soc,
            }
        )
        state, _ = model.step(state, filtered.current_a)
        if state.soc >= phase3.battery.target_soc - phase3.validation.target_soc_tolerance:
            break
    return pd.DataFrame.from_records(records)


def assign_trajectory_splits(
    trajectory_ids: list[str], config: PhaseThreeBConfig
) -> dict[str, str]:
    """按整条轨迹划分，禁止同一轨迹的相邻状态跨越数据集边界。"""
    ids = np.asarray(sorted(trajectory_ids), dtype=object)
    random = np.random.default_rng(config.random_seed)
    random.shuffle(ids)
    n_train = config.dataset.train_trajectory_count
    n_validation = config.dataset.validation_trajectory_count
    split: dict[str, str] = {}
    for trajectory_id in ids[:n_train]:
        split[str(trajectory_id)] = "train"
    for trajectory_id in ids[n_train : n_train + n_validation]:
        split[str(trajectory_id)] = "validation"
    for trajectory_id in ids[n_train + n_validation :]:
        split[str(trajectory_id)] = "test"
    return split


def sample_reachable_states(
    rollouts: pd.DataFrame, config: PhaseThreeBConfig
) -> pd.DataFrame:
    """在每条轨迹、每个SOC分箱内取内部等距点，避免只抽到轨迹端点。"""
    edges = np.asarray(config.dataset.soc_bin_edges)
    frames = []
    for trajectory_id, trajectory in rollouts.groupby("trajectory_id", sort=True):
        trajectory = trajectory.sort_values("time_s").copy()
        trajectory["soc_bin_index"] = pd.cut(
            trajectory["state_soc"],
            bins=edges,
            labels=False,
            include_lowest=True,
            right=False,
        )
        for bin_index, group in trajectory.groupby("soc_bin_index", dropna=True):
            count = min(
                config.dataset.samples_per_soc_bin_per_trajectory, len(group)
            )
            positions = np.linspace(0, len(group) - 1, count + 2)[1:-1]
            selected = group.iloc[np.unique(np.rint(positions).astype(int))].copy()
            selected["soc_bin_index"] = int(bin_index)
            frames.append(selected)
    candidates = pd.concat(frames, ignore_index=True)
    split_map = assign_trajectory_splits(
        candidates["trajectory_id"].unique().tolist(), config
    )
    candidates["split"] = candidates["trajectory_id"].map(split_map)
    candidates["reachable_source"] = True
    return candidates.sort_values(
        ["split", "trajectory_id", "time_s"]
    ).reset_index(drop=True)


def label_teacher_states(
    candidates: pd.DataFrame,
    model: ReducedBatteryModel,
    phase3: PhaseThreeConfig,
    phase3b: PhaseThreeBConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐轨迹求解MPC，将全部尝试与可用于训练的通过样本分别返回。"""
    tolerance = phase3b.active_constraint_tolerances
    records: list[dict[str, Any]] = []
    for trajectory_id, group in candidates.groupby("trajectory_id", sort=True):
        controller = ConstrainedMPC(model, phase3)
        for _, row in group.sort_values("time_s").iterrows():
            state = ReducedState(
                soc=float(row["state_soc"]),
                polarization_fast_v=float(row["state_polarization_fast_v"]),
                polarization_slow_v=float(row["state_polarization_slow_v"]),
                core_temperature_c=float(row["audit_core_temperature_c"]),
                surface_temperature_c=float(row["audit_surface_temperature_c"]),
                previous_current_a=float(row["state_previous_current_a"]),
            )
            result = controller.solve(state)
            accepted = bool(
                result.optimizer_success
                and result.prediction_feasible
                and not result.used_fallback
            )
            current_change = result.current_a - state.previous_current_a
            record = row.to_dict()
            record.update(
                {
                    "teacher_current_a": result.current_a,
                    "teacher_optimizer_success": result.optimizer_success,
                    "teacher_prediction_feasible": result.prediction_feasible,
                    "teacher_used_fallback": result.used_fallback,
                    "teacher_accepted": accepted,
                    "teacher_status": result.status,
                    "teacher_objective": result.objective_value,
                    "teacher_solve_time_s": result.solve_time_s,
                    "teacher_predicted_maximum_voltage_v": result.predicted_maximum_voltage_v,
                    "teacher_predicted_maximum_temperature_c": result.predicted_maximum_temperature_c,
                    "teacher_predicted_terminal_soc": result.predicted_terminal_soc,
                    "teacher_minimum_constraint_margin": result.minimum_constraint_margin,
                    "active_voltage_constraint": (
                        result.predicted_maximum_voltage_v
                        >= phase3.constraints.mpc_maximum_voltage_v
                        - tolerance.voltage_v
                    ),
                    "active_temperature_constraint": (
                        result.predicted_maximum_temperature_c
                        >= phase3.constraints.mpc_maximum_temperature_c
                        - tolerance.temperature_c
                    ),
                    "active_current_upper_constraint": (
                        result.current_a
                        >= phase3.constraints.maximum_current_a
                        - tolerance.current_a
                    ),
                    "active_current_change_constraint": (
                        abs(current_change)
                        >= phase3.constraints.maximum_current_change_a_per_step
                        - tolerance.current_change_a
                    ),
                }
            )
            records.append(record)
    attempts = pd.DataFrame.from_records(records)
    accepted = attempts[attempts["teacher_accepted"]].copy().reset_index(drop=True)
    return attempts, accepted


def dataset_quality_metrics(
    attempts: pd.DataFrame,
    accepted: pd.DataFrame,
    config: PhaseThreeBConfig,
) -> dict[str, Any]:
    """计算数据规模、覆盖、划分隔离和活跃约束覆盖率。"""
    split_counts = accepted["split"].value_counts().to_dict()
    soc_bin_counts = (
        accepted["soc_bin_index"].astype(int).value_counts().sort_index().to_dict()
    )
    feature_columns = [
        "state_soc",
        "state_polarization_fast_v",
        "state_polarization_slow_v",
        "state_average_temperature_c",
        "state_previous_current_a",
    ]
    duplicate_count = int(accepted.duplicated(feature_columns).sum())
    trajectory_split_counts = (
        accepted[["trajectory_id", "split"]]
        .drop_duplicates()["split"]
        .value_counts()
        .to_dict()
    )
    acceptance_fraction = float(len(accepted) / len(attempts)) if len(attempts) else 0.0
    criteria = config.success_criteria
    checks = {
        "candidate_count": len(attempts) >= criteria.minimum_candidate_count,
        "teacher_acceptance_fraction": acceptance_fraction
        >= criteria.minimum_teacher_acceptance_fraction,
        "minimum_samples_per_split": all(
            split_counts.get(name, 0) >= criteria.minimum_samples_per_split
            for name in ("train", "validation", "test")
        ),
        "minimum_samples_per_soc_bin": bool(soc_bin_counts)
        and min(soc_bin_counts.values()) >= criteria.minimum_samples_per_soc_bin,
        "trajectory_split_isolation": bool(
            accepted.groupby("trajectory_id")["split"].nunique().max() == 1
        ),
        "reachable_sources_only": bool(accepted["reachable_source"].all()),
        "label_current_bounds": bool(
            accepted["teacher_current_a"].between(0.0, 10.0).all()
        ),
        "voltage_active_coverage": (
            bool(accepted["active_voltage_constraint"].any())
            if criteria.require_voltage_active_samples
            else True
        ),
        "temperature_active_coverage": (
            bool(accepted["active_temperature_constraint"].any())
            if criteria.require_temperature_active_samples
            else True
        ),
    }
    return {
        "candidate_count": int(len(attempts)),
        "accepted_count": int(len(accepted)),
        "rejected_count": int(len(attempts) - len(accepted)),
        "teacher_acceptance_fraction": acceptance_fraction,
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
        "trajectory_split_counts": {
            str(k): int(v) for k, v in trajectory_split_counts.items()
        },
        "soc_bin_counts": {str(k): int(v) for k, v in soc_bin_counts.items()},
        "duplicate_feature_row_count": duplicate_count,
        "active_constraint_counts": {
            "voltage": int(accepted["active_voltage_constraint"].sum()),
            "temperature": int(accepted["active_temperature_constraint"].sum()),
            "current_upper": int(accepted["active_current_upper_constraint"].sum()),
            "current_change": int(accepted["active_current_change_constraint"].sum()),
        },
        "mean_teacher_solve_time_ms": float(
            accepted["teacher_solve_time_s"].mean() * 1000.0
        ),
        "maximum_teacher_solve_time_ms": float(
            accepted["teacher_solve_time_s"].max() * 1000.0
        ),
        "checks": checks,
        "success": bool(all(checks.values())),
    }
