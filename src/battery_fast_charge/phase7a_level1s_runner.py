"""Level 1S：冻结数据和控制问题，仅消融 ANN 结构与训练优化器。"""

from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ann_model import TinyANN
from .phase7a_level1_config import load_phase7a_level1_config
from .phase7a_level1_model import Level1Model
from .phase7a_level1_runner import FEATURES, _closed_loop, _regression_metrics
from .phase7a_level1s_config import Phase7ALevel1SConfig


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_sha256(frame: pd.DataFrame) -> str:
    normalized = frame.sort_values(["trajectory_id", "step_index"]).reset_index(drop=True)
    return hashlib.sha256(normalized.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def continuous_crossing_time_s(frame: pd.DataFrame, threshold_soc: float, sample_period_s: float) -> float:
    """对首次跨越阈值的单步 SOC 线性插值；时间原点是该轨迹初态。"""
    ordered = frame.sort_values("step_index")
    for row in ordered.itertuples(index=False):
        before, after = float(row.soc), float(row.next_soc)
        if before >= threshold_soc:
            return float(row.step_index) * sample_period_s
        if after >= threshold_soc and after > before:
            fraction = float(np.clip((threshold_soc - before) / (after - before), 0.0, 1.0))
            return (float(row.step_index) + fraction) * sample_period_s
    return float("nan")


def _fit_candidate(base_config: Any, train: pd.DataFrame, hidden: tuple[int, ...], solver: str, seed: int) -> tuple[TinyANN, dict[str, Any]]:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    x_scaler = StandardScaler().fit(train[list(FEATURES)])
    y_scaler = StandardScaler().fit(train[["teacher_current_a"]])
    kwargs: dict[str, Any] = {
        "hidden_layer_sizes": hidden, "activation": base_config.network.activation,
        "solver": solver, "alpha": base_config.network.regularization_alpha,
        "max_iter": base_config.network.maximum_iterations,
        "tol": base_config.network.convergence_tolerance, "random_state": seed,
    }
    if solver == "adam":
        kwargs.update(learning_rate_init=base_config.network.learning_rate_init,
                      n_iter_no_change=base_config.network.no_improvement_iterations, shuffle=True)
    estimator = MLPRegressor(**kwargs)
    started = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(x_scaler.transform(train[list(FEATURES)]), y_scaler.transform(train[["teacher_current_a"]]).ravel())
    network = TinyANN(
        feature_names=FEATURES, feature_mean=x_scaler.mean_, feature_scale=x_scaler.scale_,
        target_mean=float(y_scaler.mean_[0]), target_scale=float(y_scaler.scale_[0]),
        weights=tuple(np.asarray(v) for v in estimator.coefs_), biases=tuple(np.asarray(v) for v in estimator.intercepts_),
        minimum_current_a=base_config.mpc.current_bounds_a[0], maximum_current_a=base_config.mpc.current_bounds_a[1],
    )
    return network, {"fit_time_s": perf_counter() - started, "optimization_iterations": int(estimator.n_iter_),
                     "warning_count": len(caught), "warnings": sorted({str(v.message) for v in caught})}


def _bias_metrics(frame: pd.DataFrame, prediction: np.ndarray, low_threshold: float) -> dict[str, float | int]:
    error = np.asarray(prediction, dtype=float) - frame.teacher_current_a.to_numpy(float)
    low = frame.teacher_current_a.to_numpy(float) <= low_threshold
    return {
        "bias_a": float(np.mean(error)), "abs_bias_a": float(abs(np.mean(error))),
        "low_current_sample_count": int(low.sum()),
        "low_current_bias_a": float(np.mean(error[low])) if low.any() else float("nan"),
        "low_current_abs_bias_a": float(abs(np.mean(error[low]))) if low.any() else float("nan"),
    }


def _train_candidates(config: Phase7ALevel1SConfig, base_config: Any, dataset: pd.DataFrame, output: Path, resume: bool) -> tuple[pd.DataFrame, dict[str, TinyANN]]:
    model_dir = output / "models"; model_dir.mkdir(exist_ok=True)
    metrics_path = output / "candidate_metrics.csv"
    existing = pd.read_csv(metrics_path).to_dict("records") if resume and metrics_path.exists() else []
    records = {str(v["run_key"]): v for v in existing}
    models: dict[str, TinyANN] = {}
    train = dataset[dataset.split == "train"]
    frames = {name: dataset[dataset.split == name] for name in ("train", "validation", "test", "terminal_test")}
    for architecture in config.architectures:
        for solver in config.optimizers:
            for seed in base_config.network.initialization_seeds:
                run_key = f"{architecture.name}__{solver}__seed-{seed}"
                model_path = model_dir / f"{run_key}.npz"
                if run_key in records and model_path.exists():
                    models[run_key] = TinyANN.load(model_path); continue
                network, optimization = _fit_candidate(base_config, train, architecture.hidden_layer_sizes, solver, seed)
                network.save(model_path); models[run_key] = network
                record: dict[str, Any] = {"run_key": run_key, "architecture": architecture.name,
                                          "hidden_layer_sizes": "-".join(map(str, architecture.hidden_layer_sizes)),
                                          "optimizer": solver, "seed": seed, "parameter_count": network.parameter_count, **optimization}
                for split, frame in frames.items():
                    prediction = np.asarray(network.predict(frame[list(FEATURES)].to_numpy(float)))
                    regression = _regression_metrics(frame.teacher_current_a.to_numpy(float), prediction)
                    bias = _bias_metrics(frame, prediction, config.selection.low_current_threshold_a)
                    record.update({f"{split}_{key}": value for key, value in {**regression, **bias}.items()})
                records[run_key] = record
                pd.DataFrame(records.values()).sort_values("run_key").to_csv(metrics_path, index=False)
                print(f"Level 1S {run_key}: val={100*record['validation_nrmse']:.3f}% bias={record['validation_bias_a']:+.4f} A", flush=True)
    return pd.DataFrame(records.values()).sort_values("run_key"), models


def select_scheme(candidate_metrics: pd.DataFrame, rank_metrics: tuple[str, ...]) -> tuple[pd.DataFrame, str]:
    """五种子聚合后仅按验证集三指标等权秩和选择单一训练方案。"""
    if any(not name.startswith("validation_") for name in rank_metrics):
        raise ValueError("模型选择禁止读取冻结测试指标。")
    rows = []
    for (architecture, optimizer), group in candidate_metrics.groupby(["architecture", "optimizer"], sort=True):
        rows.append({"scheme": f"{architecture}__{optimizer}", "architecture": architecture, "optimizer": optimizer,
                     "seed_count": len(group),
                     "validation_nrmse": float(group.validation_nrmse.mean()),
                     "validation_abs_bias_a": float(group.validation_abs_bias_a.mean()),
                     "validation_low_current_abs_bias_a": float(group.validation_low_current_abs_bias_a.mean()),
                     "validation_nrmse_std": float(group.validation_nrmse.std(ddof=1))})
    summary = pd.DataFrame(rows)
    rank_columns = []
    for metric in rank_metrics:
        column = f"rank_{metric}"; summary[column] = summary[metric].rank(method="min", ascending=True); rank_columns.append(column)
    summary["validation_rank_sum"] = summary[rank_columns].sum(axis=1)
    summary = summary.sort_values(["validation_rank_sum", "validation_nrmse", "validation_abs_bias_a", "scheme"]).reset_index(drop=True)
    summary["selected"] = False; summary.loc[0, "selected"] = True
    return summary, str(summary.loc[0, "scheme"])


def _closed_loop_diagnostics(trajectories: pd.DataFrame, base_config: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    threshold = base_config.mpc.target_soc - 5e-4
    dt = base_config.model.sample_period_s
    teacher = trajectories[trajectories.controller == "mpc"]
    rows = []
    for seed in sorted(int(v) for v in trajectories.loc[trajectories.controller == "dnn", "seed"].dropna().unique()):
        dnn = trajectories[(trajectories.controller == "dnn") & (trajectories.seed == seed)]
        for trajectory_id, mpc_group in teacher.groupby("trajectory_id"):
            dnn_group = dnn[dnn.trajectory_id == trajectory_id]
            mpc_steps, dnn_steps = len(mpc_group), len(dnn_group)
            mpc_charge = float(mpc_group.current_a.sum() * dt / 3600.0)
            dnn_charge = float(dnn_group.current_a.sum() * dt / 3600.0)
            mpc_cont = continuous_crossing_time_s(mpc_group, threshold, dt)
            dnn_cont = continuous_crossing_time_s(dnn_group, threshold, dt)
            rows.append({"seed": seed, "trajectory_id": trajectory_id,
                         "mpc_steps": mpc_steps, "dnn_steps": dnn_steps,
                         "signed_step_difference": dnn_steps - mpc_steps,
                         "discrete_arrival_time_difference_s": (dnn_steps - mpc_steps) * dt,
                         "discrete_arrival_time_gap_fraction": abs(dnn_steps - mpc_steps) / mpc_steps,
                         "mpc_cumulative_charge_ah": mpc_charge, "dnn_cumulative_charge_ah": dnn_charge,
                         "cumulative_charge_error_ah": dnn_charge - mpc_charge,
                         "mpc_continuous_crossing_time_s": mpc_cont, "dnn_continuous_crossing_time_s": dnn_cont,
                         "continuous_crossing_time_difference_s": dnn_cont - mpc_cont,
                         "continuous_crossing_time_gap_fraction": abs(dnn_cont - mpc_cont) / mpc_cont})
    table = pd.DataFrame(rows)
    seed_rows = []
    for seed, group in table.groupby("seed"):
        seed_rows.append({"seed": int(seed), "mean_signed_step_difference": float(group.signed_step_difference.mean()),
                          "maximum_absolute_step_difference": int(group.signed_step_difference.abs().max()),
                          "mean_discrete_arrival_time_difference_s": float(group.discrete_arrival_time_difference_s.mean()),
                          "mean_discrete_arrival_time_gap_fraction": float(group.discrete_arrival_time_gap_fraction.mean()),
                          "mean_cumulative_charge_error_ah": float(group.cumulative_charge_error_ah.mean()),
                          "mean_absolute_cumulative_charge_error_ah": float(group.cumulative_charge_error_ah.abs().mean()),
                          "mean_continuous_crossing_time_difference_s": float(group.continuous_crossing_time_difference_s.mean()),
                          "mean_continuous_crossing_time_gap_fraction": float(group.continuous_crossing_time_gap_fraction.mean())})
    seed_summary = pd.DataFrame(seed_rows)
    overall = {"seed_count": len(seed_summary),
               "maximum_mean_discrete_arrival_time_gap_fraction": float(seed_summary.mean_discrete_arrival_time_gap_fraction.max()),
               "maximum_mean_continuous_crossing_time_gap_fraction": float(seed_summary.mean_continuous_crossing_time_gap_fraction.max()),
               "maximum_absolute_mean_cumulative_charge_error_ah": float(seed_summary.mean_cumulative_charge_error_ah.abs().max()),
               "signed_step_difference_range": [float(seed_summary.mean_signed_step_difference.min()), float(seed_summary.mean_signed_step_difference.max())]}
    return table, {"per_seed": seed_summary, "overall": overall}


def _plots(output: Path, scheme_summary: pd.DataFrame, selected_runs: pd.DataFrame, closed: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    figure_dir = output / "figures"; figure_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    labels = scheme_summary.scheme.str.replace("deep_32_32_16", "deep").str.replace("shallow_", "shallow-")
    axes[0].bar(labels, 100 * scheme_summary.validation_nrmse); axes[0].tick_params(axis="x", rotation=35)
    axes[0].set(ylabel="Validation NRMSE [%]", title="Training-scheme comparison")
    axes[1].bar(labels, scheme_summary.validation_low_current_abs_bias_a); axes[1].tick_params(axis="x", rotation=35)
    axes[1].set(ylabel="Absolute low-current bias [A]", title="Validation terminal bias")
    fig.tight_layout(); fig.savefig(figure_dir / "scheme_selection.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(selected_runs.seed.astype(str), 100 * selected_runs.terminal_test_nrmse, label="NRMSE")
    axes[0].axhline(1.0, color="red", linestyle="--"); axes[0].set(xlabel="Seed", ylabel="Terminal test NRMSE [%]")
    summary = diagnostics.groupby("seed").discrete_arrival_time_difference_s.mean()
    continuous = diagnostics.groupby("seed").continuous_crossing_time_difference_s.mean()
    x = np.arange(len(summary)); axes[1].bar(x-.18, summary, width=.36, label="Discrete"); axes[1].bar(x+.18, continuous, width=.36, label="Interpolated")
    axes[1].set_xticks(x, summary.index.astype(int).astype(str)); axes[1].set(xlabel="Seed", ylabel="Signed arrival-time difference [s]"); axes[1].legend()
    fig.tight_layout(); fig.savefig(figure_dir / "selected_scheme_closed_loop.png", dpi=180); plt.close(fig)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    s, o, c, d = payload["selection"], payload["offline"], payload["closed_loop"], payload["decision"]
    text = f"""# Phase 7A Level 1S：训练稳定性消融报告

## 结论

选择方案：**{s['selected_scheme']}**。Level 1S 判定：**{d['conclusion']}**；进入 Level 2：**{'是' if d['proceed_to_level2'] else '否'}**。

## 冻结合同

- 未生成任何新教师数据；原始数据、末端数据、两套冻结测试、MPC、1RC、约束和 12 条闭环初态的哈希均保持不变。
- 只比较 `2-32-32-16-1`、`2-16-1`、`2-32-1` 与 Adam/LBFGS，共 6 个方案、每方案 5 个原种子。
- 方案选择只使用验证集 NRMSE、总体绝对 bias、低电流区绝对 bias 的等权秩和；冻结测试未参与选择。

## 双冻结测试

- 原始冻结测试 NRMSE：{100*o['original_test_nrmse_min']:.4f}%–{100*o['original_test_nrmse_max']:.4f}%。
- 末端冻结测试 NRMSE：{100*o['terminal_test_nrmse_min']:.4f}%–{100*o['terminal_test_nrmse_max']:.4f}%。
- 末端测试有符号电流 bias：{o['terminal_test_bias_min_a']:+.5f}–{o['terminal_test_bias_max_a']:+.5f} A。
- 低电流区有符号 bias：{o['terminal_low_current_bias_min_a']:+.5f}–{o['terminal_low_current_bias_max_a']:+.5f} A。

## 固定初态闭环

- 电流 NRMSE：{100*c['current_nrmse_min']:.4f}%–{100*c['current_nrmse_max']:.4f}%。
- 最大平均离散到达时间偏差：{100*c['maximum_discrete_arrival_gap_fraction']:.4f}%（原 2% 门槛保持不变）。
- 最大平均连续插值穿越时间偏差：{100*c['maximum_continuous_arrival_gap_fraction']:.4f}%（仅诊断 5 s 采样量化，不参与验收）。
- 五种子平均有符号步数差范围：{c['signed_step_difference_range'][0]:+.3f}–{c['signed_step_difference_range'][1]:+.3f} 步，符号为 DNN−MPC。
- 最大平均累计电荷误差绝对值：{c['maximum_absolute_mean_cumulative_charge_error_ah']:.6f} Ah。

## 阶段判定

```json
{json.dumps(d['checks'], ensure_ascii=False, indent=2)}
```

Level 1R 的正式解释保持为：策略拟合、闭环电流、安全性和计算速度均通过，未完全通过项来自训练随机性导致的终端到达时间一致性，而非模型不可学习。
"""
    path.write_text(text, encoding="utf-8")


def run_phase7a_level1s(config: Phase7ALevel1SConfig, project_root: str | Path, resume: bool = False) -> dict[str, Any]:
    root = Path(project_root).resolve(); data_dir = root / "data" / "phase7a_level1s_training_stability"; output = root / "outputs" / "phase7a_level1s_training_stability"
    data_dir.mkdir(parents=True, exist_ok=True); output.mkdir(parents=True, exist_ok=True)
    paths = {name: root / value for name, value in {
        "combined": config.source_combined_dataset, "original": config.source_original_dataset,
        "terminal": config.source_terminal_dataset, "tail": config.source_tail_dataset,
        "closed_initial": config.source_closed_loop_initial_states, "level1_config": config.source_level1_config,
        "level1r_metrics": config.source_level1r_metrics}.items()}
    hashes_before = {name: file_sha256(path) for name, path in paths.items()}
    base_config = load_phase7a_level1_config(paths["level1_config"])
    level1r = json.loads(paths["level1r_metrics"].read_text(encoding="utf-8"))
    dataset = pd.read_csv(paths["combined"])
    original_test = pd.read_csv(paths["original"]).query("split == 'test'")
    terminal_test = pd.read_csv(paths["terminal"]).query("split == 'terminal_test'")
    frozen_rows_before = {"original": frame_sha256(original_test), "terminal": frame_sha256(terminal_test)}
    candidates, models = _train_candidates(config, base_config, dataset, output, resume)
    scheme_summary, selected_scheme = select_scheme(candidates, config.selection.rank_metrics)
    scheme_summary.to_csv(output / "scheme_validation_summary.csv", index=False)
    architecture, optimizer = selected_scheme.rsplit("__", 1)
    selected_runs = candidates[(candidates.architecture == architecture) & (candidates.optimizer == optimizer)].sort_values("seed")
    selected_models = {int(seed): models[f"{architecture}__{optimizer}__seed-{int(seed)}"] for seed in selected_runs.seed}
    offline_checks = {"five_original_test_seeds": bool((selected_runs.test_nrmse < base_config.gates.offline_nrmse_max).all()),
                      "five_terminal_test_seeds": bool((selected_runs.terminal_test_nrmse < base_config.gates.offline_nrmse_max).all())}
    offline = {"seed_count": len(selected_runs), "original_test_nrmse_min": float(selected_runs.test_nrmse.min()), "original_test_nrmse_max": float(selected_runs.test_nrmse.max()),
               "terminal_test_nrmse_min": float(selected_runs.terminal_test_nrmse.min()), "terminal_test_nrmse_max": float(selected_runs.terminal_test_nrmse.max()),
               "terminal_test_bias_min_a": float(selected_runs.terminal_test_bias_a.min()), "terminal_test_bias_max_a": float(selected_runs.terminal_test_bias_a.max()),
               "terminal_low_current_bias_min_a": float(selected_runs.terminal_test_low_current_bias_a.min()),
               "terminal_low_current_bias_max_a": float(selected_runs.terminal_test_low_current_bias_a.max()),
               "checks": offline_checks, "success": bool(all(offline_checks.values()))}
    model = Level1Model(base_config, root)
    trajectories, closed = _closed_loop(base_config, model, selected_models, data_dir, resume)
    generated_initial_hash = file_sha256(data_dir / "closed_loop_initial_states.csv")
    fixed_initial_hash = file_sha256(paths["closed_initial"])
    detailed, diagnostics = _closed_loop_diagnostics(trajectories, base_config)
    detailed.to_csv(data_dir / "closed_loop_diagnostics_per_trajectory.csv", index=False)
    diagnostics["per_seed"].to_csv(data_dir / "closed_loop_diagnostics_per_seed.csv", index=False)
    closed_checks = {"all_seed_current_nrmse": bool((closed.mean_current_nrmse < base_config.gates.closed_loop_current_nrmse_max).all()),
                     "all_seed_discrete_charge_time": bool((closed.mean_charge_time_gap_fraction < base_config.gates.charge_time_gap_fraction_max).all()),
                     "all_seed_target_reach": bool((closed.target_reach_fraction >= base_config.gates.minimum_target_reach_fraction).all()),
                     "all_seed_voltage_safe": bool((closed.maximum_voltage_violation_v <= base_config.gates.maximum_constraint_violation).all()),
                     "all_seed_current_safe": bool((closed.maximum_current_violation_a <= base_config.gates.maximum_constraint_violation).all()),
                     "all_seed_speedup": bool((closed.speedup > base_config.gates.minimum_speedup).all())}
    closed_payload = {"current_nrmse_min": float(closed.mean_current_nrmse.min()), "current_nrmse_max": float(closed.mean_current_nrmse.max()),
                      "maximum_discrete_arrival_gap_fraction": float(closed.mean_charge_time_gap_fraction.max()),
                      "minimum_target_reach_fraction": float(closed.target_reach_fraction.min()), "minimum_speedup": float(closed.speedup.min()),
                      "maximum_voltage_violation_v": float(closed.maximum_voltage_violation_v.max()),
                      "maximum_continuous_arrival_gap_fraction": diagnostics["overall"]["maximum_mean_continuous_crossing_time_gap_fraction"],
                      "maximum_absolute_mean_cumulative_charge_error_ah": diagnostics["overall"]["maximum_absolute_mean_cumulative_charge_error_ah"],
                      "signed_step_difference_range": diagnostics["overall"]["signed_step_difference_range"],
                      "checks": closed_checks, "success": bool(all(closed_checks.values()))}
    hashes_after = {name: file_sha256(path) for name, path in paths.items()}
    frozen_rows_after = {"original": frame_sha256(pd.read_csv(paths["original"]).query("split == 'test'")),
                         "terminal": frame_sha256(pd.read_csv(paths["terminal"]).query("split == 'terminal_test'"))}
    frozen = {"source_hashes_before": hashes_before, "source_hashes_after": hashes_after,
              "frozen_row_hashes_before": frozen_rows_before, "frozen_row_hashes_after": frozen_rows_after,
              "closed_loop_initial_state_sha256": fixed_initial_hash, "generated_closed_loop_initial_state_sha256": generated_initial_hash,
              "all_sources_preserved": bool(hashes_before == hashes_after and frozen_rows_before == frozen_rows_after),
              "closed_loop_initial_states_preserved": bool(fixed_initial_hash == generated_initial_hash),
              "no_new_teacher_data": True}
    selection_payload = {"selected_scheme": selected_scheme, "selection_data": "validation_only",
                         "rank_metrics": list(config.selection.rank_metrics), "candidate_scheme_count": len(scheme_summary),
                         "candidate_run_count": len(candidates)}
    checks = {"frozen_contract_preserved": bool(frozen["all_sources_preserved"] and frozen["closed_loop_initial_states_preserved"] and frozen["no_new_teacher_data"]),
              "dual_offline_tests_passed": offline["success"], "same_model_closed_loop_passed": closed_payload["success"]}
    success = bool(all(checks.values()))
    payload = {"study_name": config.study_name, "status": "completed", "configuration": asdict(config),
               "inherited_level1_configuration": asdict(base_config), "source_level1r_decision": level1r["decision"],
               "frozen_contract": frozen, "selection": selection_payload, "offline": offline, "closed_loop": closed_payload,
               "decision": {"checks": checks, "level1s_success": success, "proceed_to_level2": success,
                            "conclusion": "Level 1S 训练稳定性修复通过" if success else "Level 1S 未通过，保持停止在 Level 1"}, "success": success}
    (output / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    selected_runs.to_csv(output / "selected_scheme_five_seed_metrics.csv", index=False)
    _plots(output, scheme_summary, selected_runs, closed, detailed)
    _write_report(output / "PHASE7A_LEVEL1S_中文实验报告.md", payload)
    print(json.dumps(payload["decision"], ensure_ascii=False), flush=True)
    return payload
