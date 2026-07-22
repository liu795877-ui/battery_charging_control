"""Level 1R：保持控制问题不变，只修复目标 SOC 附近的数据覆盖。"""

from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ann_model import TinyANN
from .phase7a_level1_config import load_phase7a_level1_config
from .phase7a_level1_model import Level1MPC, Level1Model, Level1State
from .phase7a_level1_runner import (
    FEATURES, _closed_loop, _fit_network, _plots, _regression_metrics,
    _teacher_audit, _van_der_corput,
)
from .phase7a_level1r_config import Phase7ALevel1RConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_sha256(frame: pd.DataFrame) -> str:
    normalized = frame.sort_values(["trajectory_id", "step_index"]).reset_index(drop=True)
    return hashlib.sha256(normalized.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def design_terminal_states(config: Phase7ALevel1RConfig) -> pd.DataFrame:
    c = config.coverage
    count = c.terminal_trajectory_count
    unit = np.asarray([[(i + 0.5) / count, _van_der_corput(i + 1, 2)] for i in range(count)])
    frame = pd.DataFrame({
        "trajectory_id": [f"level1r_terminal_{i:03d}" for i in range(count)],
        "initial_soc": c.soc_bounds[0] + np.ptp(c.soc_bounds) * unit[:, 0],
        "initial_polarization_v": c.polarization_bounds_v[0] + np.ptp(c.polarization_bounds_v) * unit[:, 1],
    })
    ids = frame.trajectory_id.to_numpy(object).copy()
    np.random.default_rng(c.random_seed).shuffle(ids)
    train_end = c.train_trajectory_count
    validation_end = train_end + c.validation_trajectory_count
    split = {str(v): "train" for v in ids[:train_end]}
    split.update({str(v): "validation" for v in ids[train_end:validation_end]})
    split.update({str(v): "terminal_test" for v in ids[validation_end:]})
    frame["split"] = frame.trajectory_id.map(split)
    return frame


def design_tail_training_states(config: Phase7ALevel1RConfig) -> pd.DataFrame:
    c = config.coverage
    count = c.tail_training_trajectory_count
    unit = np.asarray([[(i + 0.5) / count, _van_der_corput(i + 37, 2)] for i in range(count)])
    return pd.DataFrame({
        "trajectory_id": [f"level1r_tail_train_{i:03d}" for i in range(count)],
        "initial_soc": c.tail_soc_bounds[0] + np.ptp(c.tail_soc_bounds) * unit[:, 0],
        "initial_polarization_v": c.polarization_bounds_v[0] + np.ptp(c.polarization_bounds_v) * unit[:, 1],
        "split": "train",
    })


def _terminal_trajectory(row: pd.Series, config: Phase7ALevel1RConfig, model: Level1Model) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    state = Level1State(float(row.initial_soc), float(row.initial_polarization_v))
    controller = Level1MPC(model)
    rows: list[dict[str, Any]] = []
    rejection = ""
    for step in range(config.coverage.trajectory_steps):
        result = controller.solve(state)
        if not result.optimizer_success or not result.prediction_feasible or result.used_fallback:
            rejection = result.status
            break
        next_state = model.step(state, result.current_a)
        rows.append({
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
    accepted = len(rows) == config.coverage.trajectory_steps
    return (rows if accepted else []), {
        "trajectory_id": row.trajectory_id, "split": row.split, "teacher_accepted": accepted,
        "completed_step_count": len(rows), "rejection_reason": rejection,
    }


def _generate_terminal_teacher(config: Phase7ALevel1RConfig, model: Level1Model, data_dir: Path, resume: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_path = data_dir / "terminal_teacher_dataset.csv"
    attempts_path = data_dir / "terminal_teacher_trajectory_audit.csv"
    design = design_terminal_states(config); design.to_csv(data_dir / "terminal_initial_state_design.csv", index=False)
    if resume and dataset_path.exists() and attempts_path.exists():
        dataset, attempts = pd.read_csv(dataset_path), pd.read_csv(attempts_path)
        if len(attempts) == config.coverage.terminal_trajectory_count:
            return dataset, attempts
    rows, attempts = [], []
    for index, (_, initial) in enumerate(design.iterrows(), start=1):
        trajectory, attempt = _terminal_trajectory(initial, config, model)
        rows.extend(trajectory); attempts.append(attempt)
        if index % 10 == 0:
            pd.DataFrame(rows).to_csv(dataset_path, index=False)
            pd.DataFrame(attempts).to_csv(attempts_path, index=False)
            print(f"Level 1R terminal teacher {index}/{len(design)}", flush=True)
    dataset, audit = pd.DataFrame(rows), pd.DataFrame(attempts)
    dataset.to_csv(dataset_path, index=False); audit.to_csv(attempts_path, index=False)
    return dataset, audit


def _generate_tail_teacher(config: Phase7ALevel1RConfig, model: Level1Model, data_dir: Path, resume: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_path = data_dir / "tail_training_teacher_dataset.csv"
    attempts_path = data_dir / "tail_training_teacher_audit.csv"
    design = design_tail_training_states(config); design.to_csv(data_dir / "tail_training_initial_state_design.csv", index=False)
    if resume and dataset_path.exists() and attempts_path.exists():
        dataset, attempts = pd.read_csv(dataset_path), pd.read_csv(attempts_path)
        if len(attempts) == config.coverage.tail_training_trajectory_count:
            return dataset, attempts
    rows, attempts = [], []
    for index, (_, initial) in enumerate(design.iterrows(), start=1):
        trajectory, attempt = _terminal_trajectory(initial, config, model)
        rows.extend(trajectory); attempts.append(attempt)
        print(f"Level 1R tail teacher {index}/{len(design)}", flush=True)
    dataset, audit = pd.DataFrame(rows), pd.DataFrame(attempts)
    dataset.to_csv(dataset_path, index=False); audit.to_csv(attempts_path, index=False)
    return dataset, audit


def _combined_dataset(original: pd.DataFrame, terminal: pd.DataFrame) -> pd.DataFrame:
    base = original.copy()
    base["coverage_source"] = "original_level1"
    augmented = terminal.copy()
    augmented["coverage_source"] = "terminal_level1r"
    columns = sorted(set(base.columns) & set(augmented.columns))
    return pd.concat([base[columns], augmented[columns]], ignore_index=True)


def _train(config: Phase7ALevel1RConfig, base_config: Any, dataset: pd.DataFrame, output: Path, resume: bool) -> tuple[pd.DataFrame, dict[int, TinyANN]]:
    metrics_path = output / "dnn_offline_metrics.csv"
    model_dir = output / "models"; model_dir.mkdir(exist_ok=True)
    records = {int(v["seed"]): v for v in (pd.read_csv(metrics_path).to_dict("records") if resume and metrics_path.exists() else [])}
    models: dict[int, TinyANN] = {}
    train = dataset[dataset.split == "train"]
    for seed in base_config.network.initialization_seeds:
        model_path = model_dir / f"level1r_seed_{seed}.npz"
        if seed in records and model_path.exists():
            models[seed] = TinyANN.load(model_path); continue
        network, optimization = _fit_network(base_config, train, seed)
        network.save(model_path); models[seed] = network
        record: dict[str, Any] = {"seed": seed, "parameter_count": network.parameter_count, **optimization}
        for split in ("train", "validation", "test", "terminal_test"):
            frame = dataset[dataset.split == split]
            prediction = np.asarray(network.predict(frame[list(FEATURES)].to_numpy(float)))
            record.update({f"{split}_{key}": value for key, value in _regression_metrics(frame.teacher_current_a.to_numpy(float), prediction).items()})
        records[seed] = record
        pd.DataFrame(records.values()).sort_values("seed").to_csv(metrics_path, index=False)
        print(f"Level 1R DNN seed {seed}: original={100*record['test_nrmse']:.3f}% terminal={100*record['terminal_test_nrmse']:.3f}%", flush=True)
    return pd.DataFrame(records.values()).sort_values("seed"), models


def _coverage_plot(output: Path, original: pd.DataFrame, terminal: pd.DataFrame) -> str:
    figure_dir = output / "figures"; figure_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(original.state_soc, original.teacher_current_a, s=6, alpha=.3, label="Level 1")
    axes[0].scatter(terminal.state_soc, terminal.teacher_current_a, s=6, alpha=.3, label="Level 1R terminal")
    axes[0].axvspan(.76, .80, color="tab:orange", alpha=.1); axes[0].legend()
    axes[0].set(xlabel="SOC", ylabel="MPC current [A]", title="Terminal taper coverage")
    axes[1].scatter(terminal.state_soc, terminal.state_polarization_v, c=terminal.teacher_current_a, s=7, cmap="viridis")
    axes[1].set(xlabel="SOC", ylabel="Polarization voltage [V]", title="Level 1R terminal teacher")
    fig.tight_layout(); path = figure_dir / "terminal_coverage_repair.png"; fig.savefig(path, dpi=180); plt.close(fig)
    return str(path)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    t, a, o, c, d = payload["terminal_teacher"], payload["terminal_teacher_audit"], payload["offline"], payload["closed_loop"], payload["decision"]
    text = f"""# Phase 7A Level 1R：末端覆盖修复报告

## 结论

Level 1R 判定：**{d['conclusion']}**。进入 Level 2：**{'是' if d['proceed_to_level2'] else '否'}**。

## 不变量与覆盖修复

- 1RC 模型、MPC 目标与约束、DNN `2-32-32-16-1 tanh`、五个原随机种子及全部验收门槛均保持不变。
- 原始冻结测试集保持 {payload['frozen_test_contract']['trajectory_count']} 条，修复前后哈希一致：{payload['frozen_test_contract']['preserved']}。
- 新增末端轨迹 {t['attempted_trajectories']} 条，每条 24 步：训练 120（含 20 条 SOC 0.795–0.799 尾端加密）、验证 20、独立末端冻结测试 20。
- 末端初态覆盖 SOC 0.74–0.799、极化电压 0–0.10 V。

## 教师覆盖与确定性

- 末端教师接受率：{100*t['acceptance_fraction']:.2f}%（{t['accepted_trajectories']}/{t['attempted_trajectories']}）。
- 低电流标签（≤0.25 A）：{t['low_current_label_count']}；降流标签（0.25–5 A）：{t['taper_current_label_count']}。
- 末端 100×15 多起点审计：{a['success']}；多值状态比例 {100*a['near_optimal_multivalued_fraction']:.2f}%；第一动作极差 P95 {a['near_optimal_first_action_range_p95_a']:.3e} A。

## pure DNN

- 原始冻结测试五种子 NRMSE：{100*o['original_test_nrmse_min']:.4f}%–{100*o['original_test_nrmse_max']:.4f}%。
- 独立末端冻结测试五种子 NRMSE：{100*o['terminal_test_nrmse_min']:.4f}%–{100*o['terminal_test_nrmse_max']:.4f}%。

## 完整同模型闭环

- 五种子平均电流 NRMSE：{100*c['mean_current_nrmse_min']:.4f}%–{100*c['mean_current_nrmse_max']:.4f}%。
- 最大平均充电时间偏差：{100*c['maximum_mean_charge_time_gap_fraction']:.4f}%。
- 最低目标到达率：{100*c['minimum_target_reach_fraction']:.1f}%；最大电压违约：{c['maximum_voltage_violation_v']:.3e} V；最低加速：{c['minimum_speedup']:.1f}×。

## 阶段门槛

```json
{json.dumps(d['checks'], ensure_ascii=False, indent=2)}
```

Level 1 的正式记录保持为：严格门槛未通过，但教师确定性和 pure DNN 局部逼近能力已通过；失败由目标 SOC 附近末端降流标签覆盖不足造成。Level 1R 只验证该覆盖修复，不改变控制问题。
"""
    path.write_text(text, encoding="utf-8")


def run_phase7a_level1r(config: Phase7ALevel1RConfig, project_root: str | Path, resume: bool = False) -> dict[str, Any]:
    root = Path(project_root).resolve(); data_dir = root / "data" / "phase7a_level1r_terminal_coverage"; output = root / "outputs" / "phase7a_level1r_terminal_coverage"
    data_dir.mkdir(parents=True, exist_ok=True); output.mkdir(parents=True, exist_ok=True)
    base_config = load_phase7a_level1_config(root / config.source_level1_config)
    original_path = root / config.source_level1_dataset
    original_file_hash_before = _sha256(original_path)
    original = pd.read_csv(original_path)
    original_test = original[original.split == "test"].copy()
    frozen_hash_before = _frame_sha256(original_test)
    if original_test.trajectory_id.nunique() != config.coverage.original_frozen_test_trajectory_count:
        raise RuntimeError("原始 Level 1 冻结测试轨迹数不符合合同。")
    model = Level1Model(base_config, root)
    terminal, attempts = _generate_terminal_teacher(config, model, data_dir, resume)
    terminal_test_hash_before = _frame_sha256(terminal[terminal.split == "terminal_test"])
    tail, tail_attempts = _generate_tail_teacher(config, model, data_dir, resume)
    all_terminal = pd.concat([terminal, tail], ignore_index=True)
    all_attempts = pd.concat([attempts, tail_attempts], ignore_index=True)
    acceptance = float(all_attempts.teacher_accepted.astype(bool).mean())
    low_count = int((all_terminal.teacher_current_a <= config.coverage.low_current_threshold_a).sum())
    taper_low, taper_high = config.coverage.taper_current_bounds_a
    taper_count = int(all_terminal.teacher_current_a.between(taper_low, taper_high, inclusive="right").sum())
    teacher_checks = {
        "acceptance": acceptance >= base_config.gates.minimum_teacher_acceptance_fraction,
        "trajectory_length": bool((all_terminal.groupby("trajectory_id").size() == config.coverage.trajectory_steps).all()),
        "low_current_labels": low_count >= config.coverage.minimum_low_current_label_count,
        "taper_current_labels": taper_count >= config.coverage.minimum_taper_current_label_count,
    }
    terminal_metrics = {"attempted_trajectories": len(all_attempts), "accepted_trajectories": int(all_attempts.teacher_accepted.astype(bool).sum()),
                        "acceptance_fraction": acceptance, "sample_count": len(all_terminal), "low_current_label_count": low_count,
                        "taper_current_label_count": taper_count, "checks": teacher_checks, "success": bool(all(teacher_checks.values()))}
    audit_dir = data_dir / "terminal_tail_audit"; audit_dir.mkdir(exist_ok=True)
    _, audit_table, audit_metrics = _teacher_audit(base_config, model, all_terminal, audit_dir, resume)
    combined = _combined_dataset(original, all_terminal); combined.to_csv(data_dir / "combined_training_contract.csv", index=False)
    frozen_hash_after = _frame_sha256(pd.read_csv(original_path).query("split == 'test'"))
    original_file_hash_after = _sha256(original_path)
    terminal_test_hash_after = _frame_sha256(pd.read_csv(data_dir / "terminal_teacher_dataset.csv").query("split == 'terminal_test'"))
    frozen_contract = {"trajectory_count": int(original_test.trajectory_id.nunique()), "sample_count": len(original_test),
                       "frozen_rows_sha256_before": frozen_hash_before, "frozen_rows_sha256_after": frozen_hash_after,
                       "source_file_sha256_before": original_file_hash_before, "source_file_sha256_after": original_file_hash_after,
                       "terminal_test_rows_sha256_before": terminal_test_hash_before, "terminal_test_rows_sha256_after": terminal_test_hash_after,
                       "preserved": bool(frozen_hash_before == frozen_hash_after and original_file_hash_before == original_file_hash_after
                                         and terminal_test_hash_before == terminal_test_hash_after)}
    payload: dict[str, Any] = {"study_name": config.study_name, "configuration": asdict(config), "inherited_level1_configuration": asdict(base_config),
                               "frozen_test_contract": frozen_contract, "terminal_teacher": terminal_metrics, "terminal_teacher_audit": audit_metrics}
    teacher_gate = bool(frozen_contract["preserved"] and terminal_metrics["success"] and audit_metrics["success"])
    if teacher_gate:
        offline, networks = _train(config, base_config, combined, output, resume)
        offline_checks = {"original_frozen_test_all_seeds": bool((offline.test_nrmse < base_config.gates.offline_nrmse_max).all()),
                          "terminal_frozen_test_all_seeds": bool((offline.terminal_test_nrmse < base_config.gates.offline_nrmse_max).all())}
        payload["offline"] = {"seed_count": len(offline), "original_test_nrmse_min": float(offline.test_nrmse.min()),
                              "original_test_nrmse_max": float(offline.test_nrmse.max()), "terminal_test_nrmse_min": float(offline.terminal_test_nrmse.min()),
                              "terminal_test_nrmse_max": float(offline.terminal_test_nrmse.max()), "checks": offline_checks, "success": bool(all(offline_checks.values()))}
        _, closed = _closed_loop(base_config, model, networks, data_dir, resume)
        closed_checks = {
            "all_seed_current_nrmse": bool((closed.mean_current_nrmse < base_config.gates.closed_loop_current_nrmse_max).all()),
            "all_seed_charge_time_gap": bool((closed.mean_charge_time_gap_fraction < base_config.gates.charge_time_gap_fraction_max).all()),
            "all_seed_target_reach": bool((closed.target_reach_fraction >= base_config.gates.minimum_target_reach_fraction).all()),
            "all_seed_voltage_safe": bool((closed.maximum_voltage_violation_v <= base_config.gates.maximum_constraint_violation).all()),
            "all_seed_current_safe": bool((closed.maximum_current_violation_a <= base_config.gates.maximum_constraint_violation).all()),
            "all_seed_speedup": bool((closed.speedup > base_config.gates.minimum_speedup).all()),
        }
        payload["closed_loop"] = {"mean_current_nrmse_min": float(closed.mean_current_nrmse.min()), "mean_current_nrmse_max": float(closed.mean_current_nrmse.max()),
                                  "maximum_mean_charge_time_gap_fraction": float(closed.mean_charge_time_gap_fraction.max()),
                                  "minimum_target_reach_fraction": float(closed.target_reach_fraction.min()),
                                  "maximum_voltage_violation_v": float(closed.maximum_voltage_violation_v.max()), "minimum_speedup": float(closed.speedup.min()),
                                  "checks": closed_checks, "success": bool(all(closed_checks.values()))}
        _coverage_plot(output, original, all_terminal); _plots(output, combined, audit_table, offline, closed)
    checks = {"frozen_test_preserved": frozen_contract["preserved"], "terminal_teacher_passed": terminal_metrics["success"],
              "terminal_determinism_passed": audit_metrics["success"], "dual_offline_tests_passed": bool(payload.get("offline", {}).get("success", False)),
              "same_model_closed_loop_passed": bool(payload.get("closed_loop", {}).get("success", False))}
    success = bool(all(checks.values()))
    payload["decision"] = {"checks": checks, "level1r_success": success, "proceed_to_level2": success,
                           "conclusion": "Level 1R 覆盖修复通过" if success else "Level 1R 未通过，保持停止在 Level 1"}
    payload["status"] = "completed"; payload["success"] = success
    (output / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(output / "PHASE7A_LEVEL1R_中文实验报告.md", payload)
    print(json.dumps(payload["decision"], ensure_ascii=False), flush=True)
    return payload
