"""围绕旧ANN闭环采集可达状态，并由阶段4B混合教师重新标注。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .ann_closed_loop import ann_features
from .ann_model import TinyANN
from .closed_loop import (
    Chen2020DFNPlant,
    _cap_current_at_target,
    _correct_reduced_state_from_dfn,
    initial_reduced_state,
)
from .hybrid_teacher import HybridMinimumTimeTeacher
from .mpc import ReducedBatteryModel, ReducedState
from .phase3_config import PhaseThreeConfig
from .phase4b_config import PhaseFourBConfig
from .phase4b2_config import ActiveRolloutConfig, PhaseFourB2Config
from .teacher_data import filter_feasible_current


FEATURE_COLUMNS = [
    "state_soc",
    "state_polarization_fast_v",
    "state_polarization_slow_v",
    "state_average_temperature_c",
    "state_previous_current_a",
]


def _exploration_offset(
    rollout: ActiveRolloutConfig, step_index: int, time_s: float
) -> float:
    """生成围绕旧ANN请求的确定性偏移，单位为A。"""
    values = rollout.parameters
    if rollout.kind == "constant_offset":
        return float(values["offset_a"])
    amplitude = float(values["amplitude_a"])
    period_s = float(values.get("period_s", 1.0))
    if rollout.kind == "sine_offset":
        phase = float(values.get("phase_rad", 0.0))
        return amplitude * np.sin(2.0 * np.pi * time_s / period_s + phase)
    if rollout.kind == "pulse_offset":
        return amplitude if int(time_s // (period_s / 2.0)) % 2 == 0 else -amplitude
    if rollout.kind == "random_block_offset":
        block = step_index // int(values["block_steps"])
        random = np.random.default_rng(int(values["seed"]) + block)
        return float(random.uniform(-amplitude, amplitude))
    raise ValueError(f"未知主动探索策略：{rollout.kind}")


def generate_ann_centered_rollout(
    ann: TinyANN,
    model: ReducedBatteryModel,
    phase3: PhaseThreeConfig,
    rollout: ActiveRolloutConfig,
) -> pd.DataFrame:
    """从统一初态推进一条ANN中心探索轨迹，并保存动作前状态。"""
    state = initial_reduced_state(phase3)
    records: list[dict[str, Any]] = []
    maximum_steps = int(
        np.ceil(
            phase3.control.maximum_simulation_time_s
            / phase3.control.control_interval_s
        )
    )
    for step_index in range(maximum_steps):
        time_s = step_index * phase3.control.control_interval_s
        seed_request = float(ann.predict(ann_features(model, state)))
        offset = _exploration_offset(rollout, step_index, time_s)
        desired = float(
            np.clip(
                seed_request + offset,
                0.0,
                phase3.constraints.maximum_current_a,
            )
        )
        filtered = filter_feasible_current(model, state, desired, phase3)
        applied, target_cap = _cap_current_at_target(
            filtered.current_a, state.soc, phase3
        )
        next_state, output = model.step(state, applied)
        records.append(
            {
                "trajectory_id": f"active_{rollout.name}",
                "policy_kind": f"ann_{rollout.kind}",
                "step_index": step_index,
                "time_s": time_s,
                "state_soc": state.soc,
                "state_polarization_fast_v": state.polarization_fast_v,
                "state_polarization_slow_v": state.polarization_slow_v,
                "state_average_temperature_c": model.average_temperature(state),
                "audit_core_temperature_c": state.core_temperature_c,
                "audit_surface_temperature_c": state.surface_temperature_c,
                "state_previous_current_a": state.previous_current_a,
                "seed_ann_requested_current_a": seed_request,
                "exploration_offset_a": offset,
                "exploration_desired_current_a": desired,
                "exploration_applied_current_a": applied,
                "exploration_safety_override": filtered.safety_override,
                "target_current_cap_active": target_cap,
                "next_voltage_v": output.terminal_voltage_v,
                "next_average_temperature_c": output.average_temperature_c,
                "next_soc": next_state.soc,
                "reachable_source": True,
                "source_dataset": "active_ann_rollout",
            }
        )
        state = next_state
        if state.soc >= phase3.battery.target_soc - phase3.validation.target_soc_tolerance:
            break
    return pd.DataFrame.from_records(records)


def generate_active_rollouts(
    ann: TinyANN,
    model: ReducedBatteryModel,
    phase3: PhaseThreeConfig,
    config: PhaseFourB2Config,
) -> pd.DataFrame:
    """生成全部12条独立ANN中心探索轨迹。"""
    frames = [
        generate_ann_centered_rollout(ann, model, phase3, rollout)
        for rollout in config.active_data.rollouts
    ]
    return pd.concat(frames, ignore_index=True)


def generate_ann_dfn_rollout(
    ann: TinyANN,
    model: ReducedBatteryModel,
    phase3: PhaseThreeConfig,
) -> pd.DataFrame:
    """在Chen2020 DFN上收集安全包装ANN实际看到的动作前状态。

    这里只生成一条名义轨迹，目的不是扩大工况范围，而是修补降阶模型推进
    与DFN反馈校正造成的状态分布偏差。
    """
    plant = Chen2020DFNPlant(phase3)
    state = initial_reduced_state(phase3)
    records: list[dict[str, Any]] = []
    maximum_steps = int(
        np.ceil(
            phase3.control.maximum_simulation_time_s
            / phase3.control.control_interval_s
        )
    )
    for step_index in range(maximum_steps):
        time_s = step_index * phase3.control.control_interval_s
        requested = float(ann.predict(ann_features(model, state)))
        filtered = filter_feasible_current(model, state, requested, phase3)
        applied, target_cap = _cap_current_at_target(
            filtered.current_a, state.soc, phase3
        )
        predicted_state, _ = model.step(state, applied)
        measurement = plant.step(applied)
        records.append(
            {
                "trajectory_id": "dagger_round_3_dfn_nominal",
                "policy_kind": "ann_dfn_nominal",
                "step_index": step_index,
                "time_s": time_s,
                "state_soc": state.soc,
                "state_polarization_fast_v": state.polarization_fast_v,
                "state_polarization_slow_v": state.polarization_slow_v,
                "state_average_temperature_c": model.average_temperature(state),
                "audit_core_temperature_c": state.core_temperature_c,
                "audit_surface_temperature_c": state.surface_temperature_c,
                "state_previous_current_a": state.previous_current_a,
                "seed_ann_requested_current_a": requested,
                "exploration_offset_a": 0.0,
                "exploration_desired_current_a": requested,
                "exploration_applied_current_a": applied,
                "exploration_safety_override": filtered.safety_override,
                "target_current_cap_active": target_cap,
                "next_voltage_v": measurement["terminal_voltage_v"],
                "next_average_temperature_c": measurement[
                    "average_temperature_c"
                ],
                "next_soc": measurement["soc"],
                "reachable_source": True,
                "source_dataset": "dagger_round_3_dfn",
            }
        )
        state = _correct_reduced_state_from_dfn(
            predicted_state, measurement, model, applied
        )
        if state.soc >= phase3.battery.target_soc - phase3.validation.target_soc_tolerance:
            break
    return pd.DataFrame.from_records(records)


def _active_split_map(config: PhaseFourB2Config) -> dict[str, str]:
    """按整轨迹随机划分，固定种子保证可重复。"""
    ids = np.asarray(
        sorted(f"active_{item.name}" for item in config.active_data.rollouts),
        dtype=object,
    )
    random = np.random.default_rng(config.random_seed)
    random.shuffle(ids)
    active = config.active_data
    train_end = active.train_trajectory_count
    validation_end = train_end + active.validation_trajectory_count
    mapping = {str(value): "train" for value in ids[:train_end]}
    mapping.update(
        {str(value): "validation" for value in ids[train_end:validation_end]}
    )
    mapping.update({str(value): "test" for value in ids[validation_end:]})
    return mapping


def sample_active_states(
    rollouts: pd.DataFrame, config: PhaseFourB2Config
) -> pd.DataFrame:
    """中间SOC每轨迹取2点，启动段和70%–80%段各取4点。"""
    edges = np.asarray(config.active_data.soc_bin_edges, dtype=float)
    frames = []
    last_bin = len(edges) - 2
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
            count = (
                config.active_data.samples_per_edge_soc_bin_per_trajectory
                if int(bin_index) in (0, last_bin)
                else config.active_data.samples_per_middle_soc_bin_per_trajectory
            )
            group = group.sort_values("time_s")
            if len(group) <= count:
                selected = group
            else:
                # 避开端点并在该SOC段内部均匀取样，减少相邻状态冗余。
                indices = np.linspace(0, len(group) - 1, count + 2)[1:-1]
                selected = group.iloc[np.unique(np.rint(indices).astype(int))]
            frames.append(selected)
    sampled = pd.concat(frames, ignore_index=True)
    sampled["split"] = sampled["trajectory_id"].map(_active_split_map(config))
    return sampled.sort_values(
        ["split", "trajectory_id", "time_s"]
    ).reset_index(drop=True)


def sample_dense_on_policy_states(
    rollout: pd.DataFrame,
    config: PhaseFourB2Config,
    trajectory_id: str = "dagger_round_2_nominal",
    source_dataset: str = "dagger_round_2",
    samples_per_soc_bin: int | None = None,
) -> pd.DataFrame:
    """沿第一轮网络名义轨迹在每个SOC分箱内部加密取样。"""
    edges = np.asarray(config.active_data.soc_bin_edges, dtype=float)
    frame = rollout.sort_values("time_s").copy()
    frame["soc_bin_index"] = pd.cut(
        frame["state_soc"],
        bins=edges,
        labels=False,
        include_lowest=True,
        right=False,
    )
    frames = []
    count = (
        config.dagger_refinement.samples_per_soc_bin
        if samples_per_soc_bin is None
        else samples_per_soc_bin
    )
    for _, group in frame.groupby("soc_bin_index", dropna=True):
        group = group.sort_values("time_s")
        if len(group) <= count:
            selected = group
        else:
            indices = np.linspace(0, len(group) - 1, count + 2)[1:-1]
            selected = group.iloc[np.unique(np.rint(indices).astype(int))]
        frames.append(selected)
    sampled = pd.concat(frames, ignore_index=True)
    sampled["trajectory_id"] = trajectory_id
    sampled["split"] = config.dagger_refinement.split
    sampled["source_dataset"] = source_dataset
    return sampled.sort_values("time_s").reset_index(drop=True)


def prepare_legacy_states(legacy: pd.DataFrame) -> pd.DataFrame:
    """保留原可达状态和整轨迹划分，但删除旧MPC标签的语义影响。"""
    frame = legacy.copy()
    frame["trajectory_id"] = "legacy_" + frame["trajectory_id"].astype(str)
    frame["source_dataset"] = "legacy_relabelled"
    frame["reachable_source"] = True
    return frame


def combine_candidate_states(
    legacy: pd.DataFrame, active: pd.DataFrame
) -> pd.DataFrame:
    """合并后按五维状态去重，防止同一状态跨集合泄漏。"""
    combined = pd.concat(
        [active.assign(_priority=0), legacy.assign(_priority=1)],
        ignore_index=True,
        sort=False,
    )
    combined = combined.sort_values(
        ["_priority", "split", "trajectory_id", "time_s"]
    )
    combined = combined.drop_duplicates(FEATURE_COLUMNS, keep="first")
    return combined.drop(columns="_priority").reset_index(drop=True)


def _state_from_row(row: pd.Series) -> ReducedState:
    """从审计表恢复完整降阶状态。"""
    return ReducedState(
        soc=float(row["state_soc"]),
        polarization_fast_v=float(row["state_polarization_fast_v"]),
        polarization_slow_v=float(row["state_polarization_slow_v"]),
        core_temperature_c=float(row["audit_core_temperature_c"]),
        surface_temperature_c=float(row["audit_surface_temperature_c"]),
        previous_current_a=float(row["state_previous_current_a"]),
    )


def label_with_hybrid_teacher(
    candidates: pd.DataFrame,
    model: ReducedBatteryModel,
    phase3: PhaseThreeConfig,
    phase4b: PhaseFourBConfig,
    config: PhaseFourB2Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """用新混合教师重标全部旧状态和主动状态，并保留模式来源。"""
    tolerance = config.active_constraint_tolerances
    records: list[dict[str, Any]] = []
    for _, group in candidates.groupby("trajectory_id", sort=True):
        teacher = HybridMinimumTimeTeacher(model, phase3, phase4b)
        for _, row in group.sort_values("time_s").iterrows():
            state = _state_from_row(row)
            decision = teacher.decide(state)
            accepted = bool(
                decision.optimizer_success
                and decision.prediction_feasible
                and not decision.used_fallback
            )
            current_change = decision.current_a - state.previous_current_a
            record = row.to_dict()
            record.update(
                {
                    "teacher_current_a": decision.current_a,
                    "teacher_control_mode": decision.mode,
                    "teacher_optimizer_success": decision.optimizer_success,
                    "teacher_prediction_feasible": decision.prediction_feasible,
                    "teacher_used_fallback": decision.used_fallback,
                    "teacher_safety_override": decision.safety_override,
                    "teacher_accepted": accepted,
                    "teacher_status": decision.mode,
                    "teacher_objective": np.nan,
                    "teacher_solve_time_s": decision.solve_time_s,
                    "teacher_predicted_maximum_voltage_v": (
                        decision.predicted_maximum_voltage_v
                    ),
                    "teacher_predicted_maximum_temperature_c": (
                        decision.predicted_maximum_temperature_c
                    ),
                    "teacher_predicted_terminal_soc": np.nan,
                    "teacher_minimum_constraint_margin": np.nan,
                    "active_voltage_constraint": (
                        decision.predicted_maximum_voltage_v
                        >= phase3.constraints.mpc_maximum_voltage_v
                        - tolerance.voltage_v
                    ),
                    "active_temperature_constraint": (
                        decision.predicted_maximum_temperature_c
                        >= phase3.constraints.mpc_maximum_temperature_c
                        - tolerance.temperature_c
                    ),
                    "active_current_upper_constraint": (
                        decision.current_a
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


def active_dataset_metrics(
    attempts: pd.DataFrame,
    accepted: pd.DataFrame,
    config: PhaseFourB2Config,
) -> dict[str, Any]:
    """计算数据规模、模式覆盖、去重和整轨迹隔离闸门。"""
    criteria = config.success_criteria
    acceptance = float(len(accepted) / len(attempts)) if len(attempts) else 0.0
    split_counts = accepted["split"].value_counts().to_dict()
    mode_counts = accepted["teacher_control_mode"].value_counts().to_dict()
    source_counts = accepted["source_dataset"].value_counts().to_dict()
    duplicate_count = int(accepted.duplicated(FEATURE_COLUMNS).sum())
    checks = {
        "minimum_accepted_label_count": len(accepted)
        >= criteria.minimum_accepted_label_count,
        "minimum_teacher_acceptance_fraction": acceptance
        >= criteria.minimum_teacher_acceptance_fraction,
        "all_splits_present": all(
            split_counts.get(name, 0) > 0
            for name in ("train", "validation", "test")
        ),
        "trajectory_split_isolation": bool(
            accepted.groupby("trajectory_id")["split"].nunique().max() == 1
        ),
        "no_duplicate_feature_rows": duplicate_count == 0,
        "reachable_sources_only": bool(accepted["reachable_source"].all()),
        "all_teacher_modes_present": all(
            mode_counts.get(name, 0) > 0
            for name in (
                "startup_reference_governor",
                "thermal_budget_mpc",
                "terminal_reference_governor",
            )
        ),
    }
    return {
        "candidate_count": int(len(attempts)),
        "accepted_count": int(len(accepted)),
        "rejected_count": int(len(attempts) - len(accepted)),
        "teacher_acceptance_fraction": acceptance,
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
        "teacher_mode_counts": {str(k): int(v) for k, v in mode_counts.items()},
        "source_counts": {str(k): int(v) for k, v in source_counts.items()},
        "duplicate_feature_row_count": duplicate_count,
        "active_constraint_counts": {
            "voltage": int(accepted["active_voltage_constraint"].sum()),
            "temperature": int(accepted["active_temperature_constraint"].sum()),
            "current_upper": int(accepted["active_current_upper_constraint"].sum()),
            "current_change": int(accepted["active_current_change_constraint"].sum()),
        },
        "checks": checks,
        "success": bool(all(checks.values())),
    }
