"""执行 Phase 7A Level 1 的教师、审计、pure DNN 与同模型闭环流水线。"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
from typing import Any
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ann_model import TinyANN
from .phase7a_level1_config import Phase7ALevel1Config
from .phase7a_level1_model import Level1MPC, Level1Model, Level1State

FEATURES = ("state_soc", "state_polarization_v")


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _van_der_corput(index: int, base: int) -> float:
    value, denominator = 0.0, 1.0
    while index:
        index, remainder = divmod(index, base)
        denominator *= base
        value += remainder / denominator
    return value


def design_initial_states(config: Phase7ALevel1Config) -> pd.DataFrame:
    count = config.data.trajectory_count
    unit = np.asarray([[(i + 0.5) / count, _van_der_corput(i + 1, 2)] for i in range(count)])
    soc = config.data.soc_bounds[0] + np.ptp(config.data.soc_bounds) * unit[:, 0]
    polarization = config.data.polarization_bounds_v[0] + np.ptp(config.data.polarization_bounds_v) * unit[:, 1]
    frame = pd.DataFrame({
        "trajectory_id": [f"level1_{i:03d}" for i in range(count)],
        "initial_soc": soc,
        "initial_polarization_v": polarization,
    })
    ids = frame.trajectory_id.to_numpy(object).copy()
    np.random.default_rng(config.data.random_seed).shuffle(ids)
    train_end = round(config.data.train_fraction * count)
    validation_end = train_end + round(config.data.validation_fraction * count)
    split = {str(value): "train" for value in ids[:train_end]}
    split.update({str(value): "validation" for value in ids[train_end:validation_end]})
    split.update({str(value): "test" for value in ids[validation_end:]})
    frame["split"] = frame.trajectory_id.map(split)
    return frame


def _teacher_trajectory(row: pd.Series, model: Level1Model) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    controller = Level1MPC(model)
    state = Level1State(float(row.initial_soc), float(row.initial_polarization_v))
    records: list[dict[str, Any]] = []
    rejection = ""
    for step in range(model.config.data.trajectory_steps):
        result = controller.solve(state)
        if not result.optimizer_success or not result.prediction_feasible or result.used_fallback:
            rejection = result.status
            break
        next_state = model.step(state, result.current_a)
        records.append({
            "trajectory_id": row.trajectory_id, "split": row.split, "step_index": step,
            "state_soc": state.soc, "state_polarization_v": state.polarization_v,
            "teacher_current_a": result.current_a,
            "terminal_voltage_v": model.terminal_voltage(state, result.current_a),
            "next_soc": next_state.soc, "next_polarization_v": next_state.polarization_v,
            "teacher_objective": result.objective_value, "teacher_solve_time_s": result.solve_time_s,
            "teacher_optimizer_success": result.optimizer_success,
            "teacher_prediction_feasible": result.prediction_feasible,
            "teacher_used_fallback": result.used_fallback,
            "minimum_prediction_margin": result.minimum_constraint_margin,
            **{f"plan_block_{i:02d}_a": float(value) for i, value in enumerate(result.plan_a)},
        })
        state = next_state
    accepted = len(records) == model.config.data.trajectory_steps
    return (records if accepted else []), {
        "trajectory_id": row.trajectory_id, "split": row.split, "teacher_accepted": accepted,
        "completed_step_count": len(records), "rejection_reason": rejection,
    }


def _generate_teacher(config: Phase7ALevel1Config, model: Level1Model, data_dir: Path, resume: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_path, attempts_path = data_dir / "teacher_dataset.csv", data_dir / "teacher_trajectory_audit.csv"
    design = design_initial_states(config)
    design.to_csv(data_dir / "initial_state_design.csv", index=False)
    if resume and dataset_path.exists() and attempts_path.exists():
        attempts = pd.read_csv(attempts_path)
        if len(attempts) == len(design):
            return pd.read_csv(dataset_path), attempts
    rows: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    for index, (_, initial) in enumerate(design.iterrows(), start=1):
        trajectory, audit = _teacher_trajectory(initial, model)
        rows.extend(trajectory); attempts.append(audit)
        if index % 10 == 0:
            pd.DataFrame(rows).to_csv(dataset_path, index=False)
            pd.DataFrame(attempts).to_csv(attempts_path, index=False)
            print(f"Level 1 teacher {index}/{len(design)}", flush=True)
    dataset, audit = pd.DataFrame(rows), pd.DataFrame(attempts)
    dataset.to_csv(dataset_path, index=False); audit.to_csv(attempts_path, index=False)
    return dataset, audit


def _representative_states(dataset: pd.DataFrame, count: int) -> pd.DataFrame:
    ordered = dataset.sort_values(["state_soc", "state_polarization_v"]).reset_index(drop=True)
    indices = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    selected = ordered.iloc[indices][["trajectory_id", "step_index", *FEATURES]].copy().reset_index(drop=True)
    selected.insert(0, "state_id", [f"audit_{i:03d}" for i in range(len(selected))])
    return selected


def _warm_starts(controller: Level1MPC, state_index: int, count: int, seed: int) -> list[tuple[str, np.ndarray | None]]:
    blocks = controller.number_of_blocks
    upper = controller.config.mpc.current_bounds_a[1]
    starts: list[tuple[str, np.ndarray | None]] = [("default", None)]
    for level in np.linspace(0.0, upper, 5):
        starts.append((f"constant_{level:.1f}", np.full(blocks, level)))
    starts.extend([
        ("ramp_up", np.linspace(0.0, upper, blocks)),
        ("ramp_down", np.linspace(upper, 0.0, blocks)),
        ("alternating", np.resize(np.asarray([0.0, upper]), blocks)),
    ])
    rng = np.random.default_rng(seed + state_index)
    while len(starts) < count:
        starts.append((f"random_{len(starts):02d}", rng.uniform(0.0, upper, blocks)))
    return starts[:count]


def _teacher_audit(config: Phase7ALevel1Config, model: Level1Model, dataset: pd.DataFrame, data_dir: Path, resume: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    states_path = data_dir / "representative_states_100.csv"
    solutions_path = data_dir / "multistart_solutions.csv"
    summary_path = data_dir / "multistart_state_summary.csv"
    selected = pd.read_csv(states_path) if resume and states_path.exists() else _representative_states(dataset, config.data.audit_state_count)
    selected.to_csv(states_path, index=False)
    if resume and solutions_path.exists():
        candidate = pd.read_csv(solutions_path)
        counts = candidate.groupby("state_id").size()
        complete = (
            len(counts) == config.data.audit_state_count
            and bool((counts == config.data.warm_starts_per_state).all())
        )
        solutions = candidate if complete else pd.DataFrame()
    else:
        solutions = pd.DataFrame()
    if solutions.empty:
        records: list[dict[str, Any]] = []
        for state_index, row in selected.iterrows():
            state = Level1State(float(row.state_soc), float(row.state_polarization_v))
            controller = Level1MPC(model)
            for warm_index, (kind, warm) in enumerate(_warm_starts(controller, state_index, config.data.warm_starts_per_state, config.data.random_seed)):
                controller.set_warm_start(warm)
                result = controller.solve(state)
                records.append({
                    "state_id": row.state_id, "warm_start_index": warm_index, "warm_start_kind": kind,
                    "first_action_a": result.current_a, "objective_value": result.objective_value,
                    "optimizer_success": result.optimizer_success, "prediction_feasible": result.prediction_feasible,
                    "used_fallback": result.used_fallback, "status": result.status,
                    "minimum_constraint_margin": result.minimum_constraint_margin,
                })
            if (state_index + 1) % 10 == 0:
                pd.DataFrame(records).to_csv(solutions_path, index=False)
                print(f"Level 1 audit {state_index + 1}/{len(selected)}", flush=True)
        solutions = pd.DataFrame(records); solutions.to_csv(solutions_path, index=False)
    records = []
    for state_id, group in solutions.groupby("state_id", sort=True):
        valid = group[group.optimizer_success.astype(bool) & group.prediction_feasible.astype(bool) & ~group.used_fallback.astype(bool)]
        if valid.empty:
            near_range, near_count = np.nan, 0
        else:
            best = float(valid.objective_value.min())
            tolerance = max(config.gates.absolute_objective_tolerance, abs(best) * config.gates.relative_objective_tolerance)
            near = valid[valid.objective_value <= best + tolerance]
            near_range, near_count = float(near.first_action_a.max() - near.first_action_a.min()), len(near)
        records.append({
            "state_id": state_id, "successful_feasible_count": len(valid),
            "optimizer_success_count": int(group.optimizer_success.astype(bool).sum()),
            "prediction_feasible_count": int(group.prediction_feasible.astype(bool).sum()),
            "fallback_count": int(group.used_fallback.astype(bool).sum()),
            "near_optimal_solution_count": near_count,
            "near_optimal_first_action_range_a": near_range,
            "near_optimal_multivalued": bool(np.isfinite(near_range) and near_range > config.gates.maximum_near_optimal_action_range_p95_a),
        })
    summary = pd.DataFrame(records); summary.to_csv(summary_path, index=False)
    p95 = float(summary.near_optimal_first_action_range_a.quantile(0.95))
    multivalued_fraction = float(summary.near_optimal_multivalued.mean())
    checks = {
        "exact_audit_contract": bool(
            len(summary) == config.data.audit_state_count
            and len(solutions) == config.data.audit_state_count * config.data.warm_starts_per_state
        ),
        "all_warm_starts_optimizer_successful": bool((summary.optimizer_success_count == config.data.warm_starts_per_state).all()),
        "all_warm_starts_prediction_feasible": bool((summary.prediction_feasible_count == config.data.warm_starts_per_state).all()),
        "zero_fallback": bool((summary.fallback_count == 0).all()),
        "multivalued_fraction_within_limit": bool(multivalued_fraction <= config.gates.maximum_multivalued_state_fraction),
        "action_range_p95_within_limit": bool(p95 <= config.gates.maximum_near_optimal_action_range_p95_a),
    }
    metrics = {
        "state_count": len(summary), "warm_starts_per_state": config.data.warm_starts_per_state,
        "near_optimal_multivalued_state_count": int(summary.near_optimal_multivalued.sum()),
        "near_optimal_multivalued_fraction": multivalued_fraction,
        "near_optimal_first_action_range_p95_a": p95,
        "maximum_near_optimal_first_action_range_a": float(summary.near_optimal_first_action_range_a.max()),
        "checks": checks, "success": bool(all(checks.values())),
    }
    return solutions, summary, metrics


def _regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction) - np.asarray(target)
    denominator = float(np.sum((target - np.mean(target)) ** 2))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    return {"mae_a": float(np.mean(np.abs(error))), "rmse_a": rmse, "nrmse": rmse / 10.0,
            "maximum_absolute_error_a": float(np.max(np.abs(error))), "bias_a": float(np.mean(error)),
            "r2": float(1.0 - np.sum(error ** 2) / denominator) if denominator > 0 else 0.0}


def _fit_network(config: Phase7ALevel1Config, train: pd.DataFrame, seed: int) -> tuple[TinyANN, dict[str, Any]]:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    x_scaler = StandardScaler().fit(train[list(FEATURES)])
    y_scaler = StandardScaler().fit(train[["teacher_current_a"]])
    estimator = MLPRegressor(
        hidden_layer_sizes=config.network.hidden_layer_sizes, activation=config.network.activation,
        solver="adam", alpha=config.network.regularization_alpha, max_iter=config.network.maximum_iterations,
        tol=config.network.convergence_tolerance, learning_rate_init=config.network.learning_rate_init,
        n_iter_no_change=config.network.no_improvement_iterations, random_state=seed, shuffle=True,
    )
    started = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(x_scaler.transform(train[list(FEATURES)]), y_scaler.transform(train[["teacher_current_a"]]).ravel())
    model = TinyANN(
        feature_names=FEATURES, feature_mean=x_scaler.mean_, feature_scale=x_scaler.scale_,
        target_mean=float(y_scaler.mean_[0]), target_scale=float(y_scaler.scale_[0]),
        weights=tuple(np.asarray(v) for v in estimator.coefs_), biases=tuple(np.asarray(v) for v in estimator.intercepts_),
        minimum_current_a=config.mpc.current_bounds_a[0], maximum_current_a=config.mpc.current_bounds_a[1],
    )
    return model, {"fit_time_s": perf_counter() - started, "optimization_iterations": int(estimator.n_iter_),
                   "warning_count": len(caught), "warnings": sorted({str(v.message) for v in caught})}


def _train_networks(config: Phase7ALevel1Config, dataset: pd.DataFrame, output: Path, resume: bool) -> tuple[pd.DataFrame, dict[int, TinyANN]]:
    model_dir = output / "models"; model_dir.mkdir(exist_ok=True)
    metrics_path = output / "dnn_offline_metrics.csv"
    existing = pd.read_csv(metrics_path).to_dict("records") if resume and metrics_path.exists() else []
    records = {int(row["seed"]): row for row in existing}
    models: dict[int, TinyANN] = {}
    train = dataset[dataset.split == "train"]
    for seed in config.network.initialization_seeds:
        model_path = model_dir / f"level1_seed_{seed}.npz"
        if seed in records and model_path.exists():
            models[seed] = TinyANN.load(model_path); continue
        network, optimization = _fit_network(config, train, seed)
        network.save(model_path); models[seed] = network
        record: dict[str, Any] = {"seed": seed, "parameter_count": network.parameter_count, **optimization}
        for split in ("train", "validation", "test"):
            frame = dataset[dataset.split == split]
            prediction = np.asarray(network.predict(frame[list(FEATURES)].to_numpy(float)))
            record.update({f"{split}_{key}": value for key, value in _regression_metrics(frame.teacher_current_a.to_numpy(float), prediction).items()})
        records[seed] = record
        pd.DataFrame(records.values()).sort_values("seed").to_csv(metrics_path, index=False)
        print(f"Level 1 DNN seed {seed}: test NRMSE={100 * record['test_nrmse']:.3f}%", flush=True)
    return pd.DataFrame(records.values()).sort_values("seed"), models


def _closed_loop_initial_states(config: Phase7ALevel1Config) -> pd.DataFrame:
    count = config.data.closed_loop_trajectory_count
    unit = np.asarray([[(i + 0.5) / count, _van_der_corput(i + 17, 2)] for i in range(count)])
    return pd.DataFrame({
        "trajectory_id": [f"closed_{i:02d}" for i in range(count)],
        "initial_soc": config.data.closed_loop_soc_bounds[0] + np.ptp(config.data.closed_loop_soc_bounds) * unit[:, 0],
        "initial_polarization_v": config.data.polarization_bounds_v[0] + np.ptp(config.data.polarization_bounds_v) * unit[:, 1],
    })


def _rollout_closed_loop(config: Phase7ALevel1Config, model: Level1Model, controller: Level1MPC | TinyANN, initial: pd.Series, kind: str, seed: int | None = None) -> list[dict[str, Any]]:
    state = Level1State(float(initial.initial_soc), float(initial.initial_polarization_v))
    rows = []
    for step in range(config.data.maximum_closed_loop_steps):
        started = perf_counter()
        if kind == "mpc":
            result = controller.solve(state)  # type: ignore[union-attr]
            elapsed, current = result.solve_time_s, result.current_a
            if not result.optimizer_success or not result.prediction_feasible: break
        else:
            current = float(controller.predict(np.asarray([state.soc, state.polarization_v])))  # type: ignore[union-attr]
            elapsed = perf_counter() - started
        voltage = model.terminal_voltage(state, current)
        next_state = model.step(state, current)
        rows.append({"controller": kind, "seed": seed, "trajectory_id": initial.trajectory_id, "step_index": step,
                     "soc": state.soc, "polarization_v": state.polarization_v, "current_a": current,
                     "terminal_voltage_v": voltage, "next_soc": next_state.soc, "elapsed_s": elapsed})
        state = next_state
        if state.soc >= config.mpc.target_soc - 5e-4: break
    return rows


def _closed_loop(config: Phase7ALevel1Config, model: Level1Model, networks: dict[int, TinyANN], data_dir: Path, resume: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    path, metrics_path = data_dir / "closed_loop_trajectories.csv", data_dir / "closed_loop_metrics.csv"
    if resume and path.exists() and metrics_path.exists(): return pd.read_csv(path), pd.read_csv(metrics_path)
    initial_states = _closed_loop_initial_states(config); initial_states.to_csv(data_dir / "closed_loop_initial_states.csv", index=False)
    rows: list[dict[str, Any]] = []
    for index, (_, initial) in enumerate(initial_states.iterrows(), start=1):
        rows.extend(_rollout_closed_loop(config, model, Level1MPC(model), initial, "mpc"))
        print(f"Level 1 closed-loop MPC {index}/{len(initial_states)}", flush=True)
    for seed, network in networks.items():
        for _, initial in initial_states.iterrows(): rows.extend(_rollout_closed_loop(config, model, network, initial, "dnn", seed))
    trajectories = pd.DataFrame(rows); trajectories.to_csv(path, index=False)
    teacher = trajectories[trajectories.controller == "mpc"]
    records = []
    for seed in networks:
        dnn = trajectories[(trajectories.controller == "dnn") & (trajectories.seed == seed)]
        per_trajectory = []
        for trajectory_id, teacher_group in teacher.groupby("trajectory_id"):
            dnn_group = dnn[dnn.trajectory_id == trajectory_id]
            paired = teacher_group[["step_index", "current_a"]].merge(dnn_group[["step_index", "current_a"]], on="step_index", suffixes=("_mpc", "_dnn"))
            nrmse = _regression_metrics(paired.current_a_mpc.to_numpy(), paired.current_a_dnn.to_numpy())["nrmse"]
            mpc_reached = float(teacher_group.next_soc.iloc[-1]) >= config.mpc.target_soc - 5e-4
            dnn_reached = float(dnn_group.next_soc.iloc[-1]) >= config.mpc.target_soc - 5e-4
            time_gap = abs(len(dnn_group) - len(teacher_group)) / len(teacher_group)
            per_trajectory.append((nrmse, mpc_reached, dnn_reached, time_gap))
        violation = np.maximum(dnn.terminal_voltage_v.to_numpy() - config.mpc.terminal_voltage_max_v, 0.0)
        current_violation = np.maximum(dnn.current_a.to_numpy() - config.mpc.current_bounds_a[1], 0.0)
        teacher_time = float(teacher.elapsed_s.sum()); dnn_time = float(dnn.elapsed_s.sum())
        records.append({"seed": seed, "mean_current_nrmse": float(np.mean([v[0] for v in per_trajectory])),
                        "maximum_current_nrmse": float(np.max([v[0] for v in per_trajectory])),
                        "target_reach_fraction": float(np.mean([v[2] for v in per_trajectory])),
                        "mpc_target_reach_fraction": float(np.mean([v[1] for v in per_trajectory])),
                        "mean_charge_time_gap_fraction": float(np.mean([v[3] for v in per_trajectory])),
                        "maximum_voltage_violation_v": float(np.max(violation)),
                        "maximum_current_violation_a": float(np.max(current_violation)),
                        "mpc_time_s": teacher_time, "dnn_time_s": dnn_time,
                        "speedup": teacher_time / dnn_time})
    metrics = pd.DataFrame(records); metrics.to_csv(metrics_path, index=False)
    return trajectories, metrics


def _plots(output: Path, dataset: pd.DataFrame, audit: pd.DataFrame, offline: pd.DataFrame, closed: pd.DataFrame) -> list[str]:
    figure_dir = output / "figures"; figure_dir.mkdir(exist_ok=True)
    paths = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    scatter = axes[0].scatter(dataset.state_soc, dataset.state_polarization_v, c=dataset.teacher_current_a, s=8, cmap="viridis")
    axes[0].set(xlabel="SOC", ylabel="Polarization voltage [V]", title="Accepted teacher samples"); fig.colorbar(scatter, ax=axes[0], label="Current [A]")
    axes[1].hist(audit.near_optimal_first_action_range_a, bins=25); axes[1].axvline(0.05, color="red", linestyle="--")
    axes[1].set(xlabel="Near-optimal first-action range [A]", ylabel="States", title="100-state multistart audit")
    fig.tight_layout(); path = figure_dir / "teacher_and_audit.png"; fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(offline.seed.astype(str), 100 * offline.test_nrmse); axes[0].axhline(1.0, color="red", linestyle="--")
    axes[0].set(xlabel="Seed", ylabel="Test NRMSE [%]", title="Frozen offline test")
    axes[1].bar(closed.seed.astype(str), 100 * closed.mean_current_nrmse); axes[1].axhline(1.0, color="red", linestyle="--")
    axes[1].set(xlabel="Seed", ylabel="Closed-loop NRMSE [%]", title="Same-model closed loop")
    fig.tight_layout(); path = figure_dir / "dnn_seed_stability.png"; fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))
    return paths


def _write_report(path: Path, metrics: dict[str, Any]) -> None:
    teacher, audit, offline, closed, decision = metrics["teacher"], metrics["teacher_audit"], metrics.get("offline", {}), metrics.get("closed_loop", {}), metrics["decision"]
    text = f"""# Phase 7A Level 1：项目参数 1RC pure DNN 消融报告

## 结论

Level 1 判定：**{decision['conclusion']}**。是否允许进入 Level 2：**{'是' if decision['proceed_to_level2'] else '否'}**。

## 实验边界

- 状态仅为 `SOC` 与主导极化电压 `Vp`；使用项目 OCV、5 Ah 容量、`R0={metrics['configuration']['model']['r0_ohm']:.8f} Ω`、`R1={metrics['configuration']['model']['r1_ohm']:.8f} Ω`、`τ1={metrics['configuration']['model']['tau1_s']:.6f} s`。
- 仅保留 `0–10 A` 电流边界与 `Vtr≤4.20 V`；未加入斜率、温度、扰动、DFN 或压力场。
- 候选教师轨迹 240 条、每条 8 步；训练/验证/测试按轨迹冻结隔离。

## 教师数据与确定性审计

- 接受轨迹：{teacher['accepted_trajectories']}/{teacher['attempted_trajectories']}（{100*teacher['acceptance_fraction']:.2f}%），有效样本 {teacher['sample_count']}。
- 100 个代表状态各运行 15 个 warm start；fallback 总数为 0 的检查：{audit['checks']['zero_fallback']}。
- 近最优多值状态比例：{100*audit['near_optimal_multivalued_fraction']:.2f}%（门槛 5%）。
- 近最优第一动作极差 P95：{audit['near_optimal_first_action_range_p95_a']:.6f} A（门槛 0.05 A）。
- 教师审计总判定：{'通过' if audit['success'] else '未通过'}。

## pure DNN 与同模型闭环

{('- 五种子冻结测试 NRMSE 范围：' + f"{100*offline['minimum_test_nrmse']:.4f}%–{100*offline['maximum_test_nrmse']:.4f}%（门槛 <1%）。" + chr(10) + '- 五种子同模型闭环平均电流 NRMSE 范围：' + f"{100*closed['minimum_mean_current_nrmse']:.4f}%–{100*closed['maximum_mean_current_nrmse']:.4f}%（门槛 <1%）。" + chr(10) + '- 最低在线加速：' + f"{closed['minimum_speedup']:.1f}×。") if offline else '- 教师审计未通过，依据预注册停止条件未训练 DNN。'}

## 阶段门槛

```json
{json.dumps(decision['checks'], ensure_ascii=False, indent=2)}
```

## 失败定位

{('冻结离线测试通过但同模型闭环失败，属于预注册的情况 C。教师状态最大 SOC 为 ' + f"{metrics['coverage_diagnostic']['teacher_state_soc_max']:.4f}" + '，而闭环运行到 ' + f"{metrics['coverage_diagnostic']['closed_loop_state_soc_max']:.4f}" + '；原始 8 步教师轨迹没有覆盖目标 SOC 0.80 附近的末端降流区域。DNN 虽保持安全并到达目标，但在该分布外区域不能复现 MPC。') if metrics.get('coverage_diagnostic', {}).get('terminal_soc_coverage_gap') else '未检测到明确的 SOC 末端覆盖缺口。'}

最终判定严格要求“教师确定性、冻结离线测试、同模型闭环”三者同时通过；本报告不包含任何 Level 2 或更高层级实验。
"""
    path.write_text(text, encoding="utf-8")


def run_phase7a_level1(config: Phase7ALevel1Config, project_root: str | Path, resume: bool = False) -> dict[str, Any]:
    root = Path(project_root).resolve(); data_dir = root / "data" / "phase7a_level1_1rc"; output = root / "outputs" / "phase7a_level1_1rc"
    data_dir.mkdir(parents=True, exist_ok=True); output.mkdir(parents=True, exist_ok=True)
    model = Level1Model(config, root)
    dataset, attempts = _generate_teacher(config, model, data_dir, resume)
    if dataset.groupby("trajectory_id").split.nunique().max() != 1: raise RuntimeError("检测到轨迹跨集合泄漏。")
    acceptance = float(attempts.teacher_accepted.astype(bool).mean())
    teacher_metrics = {"attempted_trajectories": len(attempts), "accepted_trajectories": int(attempts.teacher_accepted.astype(bool).sum()),
                       "acceptance_fraction": acceptance, "sample_count": len(dataset),
                       "split_trajectory_counts": {str(k): int(v) for k, v in dataset.groupby("split").trajectory_id.nunique().items()},
                       "success": bool(acceptance >= config.gates.minimum_teacher_acceptance_fraction)}
    _, audit_table, audit_metrics = _teacher_audit(config, model, dataset, data_dir, resume)
    teacher_gate = bool(teacher_metrics["success"] and audit_metrics["success"])
    payload: dict[str, Any] = {"study_name": config.study_name, "configuration": asdict(config), "teacher": teacher_metrics, "teacher_audit": audit_metrics}
    if teacher_gate:
        offline_runs, networks = _train_networks(config, dataset, output, resume)
        offline_success = bool((offline_runs.test_nrmse < config.gates.offline_nrmse_max).all())
        payload["offline"] = {"seed_count": len(offline_runs), "passing_seed_count": int((offline_runs.test_nrmse < config.gates.offline_nrmse_max).sum()),
                              "minimum_test_nrmse": float(offline_runs.test_nrmse.min()), "maximum_test_nrmse": float(offline_runs.test_nrmse.max()), "success": offline_success}
        closed_trajectories, closed_runs = _closed_loop(config, model, networks, data_dir, resume)
        closed_checks = {
            "all_seed_current_nrmse": bool((closed_runs.mean_current_nrmse < config.gates.closed_loop_current_nrmse_max).all()),
            "all_seed_charge_time_gap": bool((closed_runs.mean_charge_time_gap_fraction < config.gates.charge_time_gap_fraction_max).all()),
            "all_seed_target_reach": bool((closed_runs.target_reach_fraction >= config.gates.minimum_target_reach_fraction).all()),
            "all_seed_voltage_safe": bool((closed_runs.maximum_voltage_violation_v <= config.gates.maximum_constraint_violation).all()),
            "all_seed_current_safe": bool((closed_runs.maximum_current_violation_a <= config.gates.maximum_constraint_violation).all()),
            "all_seed_speedup": bool((closed_runs.speedup > config.gates.minimum_speedup).all()),
        }
        payload["closed_loop"] = {"seed_count": len(closed_runs), "minimum_mean_current_nrmse": float(closed_runs.mean_current_nrmse.min()),
                                  "maximum_mean_current_nrmse": float(closed_runs.mean_current_nrmse.max()),
                                  "maximum_mean_charge_time_gap_fraction": float(closed_runs.mean_charge_time_gap_fraction.max()),
                                  "minimum_target_reach_fraction": float(closed_runs.target_reach_fraction.min()),
                                  "maximum_voltage_violation_v": float(closed_runs.maximum_voltage_violation_v.max()),
                                  "minimum_speedup": float(closed_runs.speedup.min()), "checks": closed_checks, "success": bool(all(closed_checks.values()))}
        payload["coverage_diagnostic"] = {
            "teacher_state_soc_max": float(dataset.state_soc.max()),
            "closed_loop_state_soc_max": float(closed_trajectories.soc.max()),
            "teacher_polarization_max_v": float(dataset.state_polarization_v.max()),
            "closed_loop_polarization_max_v": float(closed_trajectories.polarization_v.max()),
            "terminal_soc_coverage_gap": bool(
                dataset.state_soc.max() < config.mpc.target_soc - 5e-4
                and closed_trajectories.soc.max() >= config.mpc.target_soc - 5e-4
            ),
        }
        _plots(output, dataset, audit_table, offline_runs, closed_runs)
    checks = {"teacher_determinism_passed": teacher_gate,
              "offline_test_passed": bool(payload.get("offline", {}).get("success", False)),
              "same_model_closed_loop_passed": bool(payload.get("closed_loop", {}).get("success", False))}
    success = bool(all(checks.values()))
    payload["decision"] = {"checks": checks, "level1_success": success, "proceed_to_level2": success,
                           "conclusion": "Level 1 通过" if success else ("教师审计未通过，按计划停止 DNN 训练" if not teacher_gate else "Level 1 未通过")}
    payload["status"] = "completed"; payload["success"] = success
    (output / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    _write_report(output / "PHASE7A_LEVEL1_中文实验报告.md", payload)
    print(json.dumps(payload["decision"], ensure_ascii=False), flush=True)
    return payload
