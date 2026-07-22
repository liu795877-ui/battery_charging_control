"""运行 Phase 6P-0 NDC 论文原位复现并生成可审计产物。"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .phase6p0_config import PhaseSixPZeroConfig
from .phase6p0_dnn import FEATURES, PaperNDCNetwork, train_paper_ndc_network
from .phase6p0_ndc import (
    NDCMPC,
    NDCModel,
    generate_frozen_test_initial_states,
    generate_training_initial_states,
    rollout_mpc_trajectory,
)


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"无法序列化 {type(value)!r}")


def _nrmse_percent(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=float)
    error = np.asarray(prediction, dtype=float) - target
    scale = float(np.max(target) - np.min(target))
    return float(100.0 * np.sqrt(np.mean(error**2)) / scale) if scale > 0.0 else float("nan")


def _build_training_dataset(model: NDCModel, controller: NDCMPC, design: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trajectories: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for row in design.itertuples(index=False):
        frame, audit = rollout_mpc_trajectory(
            controller,
            np.asarray([row.bulk_voltage_v, row.surface_voltage_v]),
            model.config.data.training_trajectory_steps,
            row.initial_state_id,
        )
        if len(frame):
            frame["sampling_method"] = row.sampling_method
            trajectories.append(frame)
        audits.append(audit)
    dataset = pd.concat(trajectories, ignore_index=True) if trajectories else pd.DataFrame()
    return dataset, pd.DataFrame.from_records(audits)


def _build_test_teacher(model: NDCModel, controller: NDCMPC, initial_states: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trajectories: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for row in initial_states.itertuples(index=False):
        frame, audit = rollout_mpc_trajectory(
            controller,
            np.asarray([row.bulk_voltage_v, row.surface_voltage_v]),
            model.config.mpc.maximum_closed_loop_steps,
            row.trajectory_id,
        )
        trajectories.append(frame)
        audits.append(audit)
    return pd.concat(trajectories, ignore_index=True), pd.DataFrame.from_records(audits)


def _rollout_dnn(model: NDCModel, network: PaperNDCNetwork, initial_states: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    rows: list[dict[str, object]] = []
    inference_time = 0.0
    for initial in initial_states.itertuples(index=False):
        state = np.asarray([initial.bulk_voltage_v, initial.surface_voltage_v], dtype=float)
        previous = model.config.mpc.initial_previous_current_a
        for step_index in range(model.config.mpc.maximum_closed_loop_steps):
            start = perf_counter()
            current = float(network.predict(np.asarray([state[1], state[0]])))
            inference_time += perf_counter() - start
            next_state = model.step(state, current)
            rows.append(
                {
                    "trajectory_id": initial.trajectory_id,
                    "step_index": step_index,
                    "bulk_voltage_v": state[0],
                    "surface_voltage_v": state[1],
                    "soc": model.soc(state),
                    "previous_current_a": previous,
                    "dnn_current_a": current,
                    "next_bulk_voltage_v": next_state[0],
                    "next_surface_voltage_v": next_state[1],
                    "next_soc": model.soc(next_state),
                    "next_terminal_voltage_v": model.terminal_voltage(next_state, current),
                    "next_health_margin_v": model.health_margin(next_state),
                }
            )
            state = next_state
            previous = current
    return pd.DataFrame.from_records(rows), inference_time


def _closed_loop_metrics(
    config: PhaseSixPZeroConfig,
    teacher: pd.DataFrame,
    dnn: pd.DataFrame,
    inference_time_s: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    paired = teacher[["trajectory_id", "step_index", "teacher_current_a", "mpc_solve_time_s"]].merge(
        dnn, on=["trajectory_id", "step_index"], how="inner", validate="one_to_one"
    )
    records: list[dict[str, object]] = []
    for trajectory_id, group in paired.groupby("trajectory_id", sort=True):
        records.append(
            {
                "trajectory_id": trajectory_id,
                "current_nrmse_percent": _nrmse_percent(group["teacher_current_a"].to_numpy(), group["dnn_current_a"].to_numpy()),
                "maximum_current_error_a": float(np.max(np.abs(group["dnn_current_a"] - group["teacher_current_a"]))),
                "final_soc": float(group["next_soc"].iloc[-1]),
                "target_reached": bool(group["next_soc"].iloc[-1] >= config.mpc.target_soc - 1.0e-3),
            }
        )
    trajectory_metrics = pd.DataFrame.from_records(records)
    upper_violation = np.maximum(dnn["dnn_current_a"].to_numpy() - config.mpc.current_bounds_a[1], 0.0)
    lower_violation = np.maximum(config.mpc.current_bounds_a[0] - dnn["dnn_current_a"].to_numpy(), 0.0)
    surface_violation = np.maximum(dnn["next_surface_voltage_v"].to_numpy() - config.mpc.surface_voltage_max_v, 0.0)
    terminal_violation = np.maximum(dnn["next_terminal_voltage_v"].to_numpy() - config.mpc.terminal_voltage_max_v, 0.0)
    health_violation = np.maximum(-dnn["next_health_margin_v"].to_numpy(), 0.0)
    mpc_time = float(teacher["mpc_solve_time_s"].sum())
    metrics = {
        "paired_sample_count": int(len(paired)),
        "mean_trajectory_current_nrmse_percent": float(trajectory_metrics["current_nrmse_percent"].mean()),
        "maximum_trajectory_current_nrmse_percent": float(trajectory_metrics["current_nrmse_percent"].max()),
        "target_reach_fraction": float(trajectory_metrics["target_reached"].mean()),
        "constraint_violations": {
            "current_upper_average_a": float(np.mean(upper_violation)),
            "current_upper_maximum_a": float(np.max(upper_violation)),
            "current_lower_average_a": float(np.mean(lower_violation)),
            "current_lower_maximum_a": float(np.max(lower_violation)),
            "surface_voltage_average_v": float(np.mean(surface_violation)),
            "surface_voltage_maximum_v": float(np.max(surface_violation)),
            "terminal_voltage_average_v": float(np.mean(terminal_violation)),
            "terminal_voltage_maximum_v": float(np.max(terminal_violation)),
            "health_average_v": float(np.mean(health_violation)),
            "health_maximum_v": float(np.max(health_violation)),
        },
        "timing": {
            "mpc_total_s": mpc_time,
            "dnn_total_s": inference_time_s,
            "speedup": float(mpc_time / inference_time_s) if inference_time_s > 0.0 else float("inf"),
        },
    }
    return metrics, trajectory_metrics


def _plot_results(output: Path, design: pd.DataFrame, training: pd.DataFrame, teacher: pd.DataFrame, dnn: pd.DataFrame, seed_metrics: pd.DataFrame) -> list[str]:
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(design["bulk_voltage_v"], design["surface_voltage_v"], s=10, c=np.where(design["sampling_method"] == "hammersley", "tab:blue", "tab:orange"), alpha=0.75)
    axes[0].scatter(training["bulk_voltage_v"], training["surface_voltage_v"], s=4, c="tab:red", alpha=0.25)
    axes[0].set(xlabel="Bulk voltage Vb [V]", ylabel="Surface voltage Vs [V]", title="Hybrid DOCE and unfolded MPC states")
    axes[1].hist(training["teacher_current_a"], bins=30, color="tab:blue", edgecolor="white")
    axes[1].set(xlabel="Optimal current [A]", ylabel="Samples", title="Training target distribution")
    fig.tight_layout()
    path = figure_dir / "training_design.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    paired = teacher.merge(
        dnn[["trajectory_id", "step_index", "dnn_current_a", "next_soc", "next_terminal_voltage_v"]].rename(
            columns={"next_soc": "dnn_next_soc", "next_terminal_voltage_v": "dnn_next_terminal_voltage_v"}
        ),
        on=["trajectory_id", "step_index"],
    )
    selected = paired[paired["trajectory_id"].isin(["ndc_test_00", "ndc_test_01", "ndc_test_02"])]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for trajectory_id, group in selected.groupby("trajectory_id"):
        time_min = group["step_index"].to_numpy() * 60.0 / 60.0
        axes[0].plot(time_min, group["teacher_current_a"], linewidth=1.5, label=f"MPC {trajectory_id}")
        axes[0].plot(time_min, group["dnn_current_a"], linestyle="--", linewidth=1.1, label=f"DNN {trajectory_id}")
        axes[1].plot(time_min, group["dnn_next_soc"], linewidth=1.3, label=trajectory_id)
    axes[0].axhline(3.0, color="black", linewidth=0.8)
    axes[0].set(ylabel="Current [A]", title="Frozen closed-loop trajectories")
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].axhline(0.9, color="black", linewidth=0.8)
    axes[1].set(xlabel="Time [min]", ylabel="DNN SOC")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = figure_dir / "closed_loop_examples.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(seed_metrics["seed"].astype(str), seed_metrics["internal_test_nrmse_percent"], color="tab:green")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    ax.set(xlabel="Initialization seed", ylabel="Internal test NRMSE [%]", title="BR-LM seed stability")
    fig.tight_layout()
    path = figure_dir / "seed_stability.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))
    return paths


def _write_report(path: Path, metrics: dict[str, Any]) -> None:
    gates = metrics["gates"]
    violations = metrics["closed_loop"]["constraint_violations"]
    text = f"""# Phase 6P-0：论文 NDC 原位复现报告

## 结论

本次阳性对照总体判定：**{'通过' if metrics['success'] else '未通过'}**。在该判定通过前，不进入 Chen2020 Phase 2R 审计，也不开展 A1–A7 加难度消融。

## 论文合同与证据边界

- 论文：Shokry 等，*Computers & Chemical Engineering* 199 (2025) 109096，DOI `{metrics['source_doi']}`。
- PDF 第 11 页、式 (12)–(24)：NDC 两状态、60 s、`Np=10`、`Nu=2`、`Nc=1`、电流 0–3 A、表面电压 0.95 V、端电压 4.2 V、健康约束。
- PDF 第 11–12 页、图 6–7：324+76 个混合 DOCE 初态、每个闭环展开 5 步、约 2000 样本、2-7-5-3-1 sigmoid、BR-LMB、90/10 随机划分。
- PDF 第 12–13 页、表 3–4：30 条独立闭环测试轨迹；论文报告开环 0.90%、闭环 0.38%，健康约束平均违约约 `1.5e-2`。
- 论文未报告随机种子、76 个全因子点的具体水平、归一化细节和 BR-LMB 超参数。本实现冻结种子，使用 20x20 全因子候选的 76 个可行边界点、MATLAB 风格 `mapminmax`，并用 Bayesian evidence 更新的正则化非线性最小二乘近似 BR-LMB；这些属于复现假设，不冒充论文原文。
- 论文网络输入仅有 `(Vs,Vb)`，但未说明首个电流增量如何引用历史动作。为保持教师标签是两状态的单值函数，本实现只对两个 CVP 预测块之间的变化施加 `R=0.1`，不把未进入网络输入的上一时刻电流加入目标函数。
- PDF 文本抽取会丢失表 2 中 OCV 多项式的负号；本实现按图 9 的端电压量级采用交替符号 `3.2, 3.041, -11.475, 24.457, -23.536, 8.513`。式 (17) 与表 2 的 `beta2=-10` 也存在排版歧义，本实现采用 `exp(beta2*(1-SOC))`；反向符号会令低 SOC 串联电阻达到不可能支持论文 3 A 轨迹的量级。
- 因上述未报告细节与复现假设，本结果证明的是“论文方法在当前代码框架中的功能性阳性对照”，不是对论文数值的逐比特复刻；本实现误差低于论文表 3，不应据此声称优于论文。

## 数据与优化审计

| 指标 | 结果 |
|---|---:|
| 训练初态 | {metrics['data']['training_initial_states']} |
| 完整训练轨迹 | {metrics['data']['complete_training_trajectories']} / 400 |
| 监督样本 | {metrics['data']['training_samples']} |
| 冻结测试轨迹 | {metrics['data']['complete_test_trajectories']} / 30 |
| 冻结测试样本 | {metrics['data']['test_samples']} |
| 选中随机种子 | {metrics['network']['selected_seed']} |
| 内部 10% NRMSE <1% 的种子 | {metrics['network']['passing_seed_count']} / {metrics['network']['seed_count']} |
| 五种子内部 NRMSE 均值 ± 标准差 | {metrics['network']['seed_internal_nrmse_mean_percent']:.4f}% ± {metrics['network']['seed_internal_nrmse_std_percent']:.4f}% |

## 验收指标

| 验收项 | 结果 | 门槛 | 判定 |
|---|---:|---:|---|
| 冻结测试离线 NRMSE | {metrics['offline']['nrmse_percent']:.4f}% | < 1% | {'通过' if gates['offline_nrmse'] else '未通过'} |
| 闭环电流平均 NRMSE | {metrics['closed_loop']['mean_trajectory_current_nrmse_percent']:.4f}% | < 1% | {'通过' if gates['closed_loop_current_nrmse'] else '未通过'} |
| DNN 到达目标比例 | {metrics['closed_loop']['target_reach_fraction']:.1%} | 100% | {'通过' if gates['target_reach'] else '未通过'} |
| 最大约束违约 | {metrics['maximum_constraint_violation']:.6g} A/V | <= 1e-2 数量级 | {'通过' if gates['constraint_violation'] else '未通过'} |
| 在线加速 | {metrics['closed_loop']['timing']['speedup']:.1f}x | > 2x | {'通过' if gates['speedup'] else '未通过'} |
| 随机种子多数重复通过 | {metrics['network']['passing_seed_count']} / {metrics['network']['seed_count']} | > 50% | {'通过' if gates['majority_seeds'] else '未通过'} |

约束违约明细：端电压最大 `{violations['terminal_voltage_maximum_v']:.6g} V`，健康约束最大 `{violations['health_maximum_v']:.6g} V`，电流上界最大 `{violations['current_upper_maximum_a']:.6g} A`，电流下界最大 `{violations['current_lower_maximum_a']:.6g} A`。

## 阶段决策

{'阳性对照已通过，可以开始第二阶段 Chen2020 Phase 2R 模型充分性审计；A1 及以后仍需等 Phase 2R 审计完成。' if metrics['success'] else '阳性对照尚未通过。下一步仅诊断 NDC 管线中的数据展开、标签时序、归一化、BR-LM 近似、闭环更新和 NRMSE 定义，不重建 Chen2020，也不进入 A1。'}
"""
    path.write_text(text, encoding="utf-8")


def run_phase_six_p_zero(
    config: PhaseSixPZeroConfig,
    project_root: str | Path = ".",
    resume: bool = False,
) -> dict[str, Any]:
    root = Path(project_root)
    output = root / "outputs" / "phase6p0_ndc_paper"
    output.mkdir(parents=True, exist_ok=True)
    model = NDCModel(config)
    controller = NDCMPC(model)
    resume_files = [
        "training_initial_states.csv", "training_teacher_2000.csv", "training_trajectory_audit.csv",
        "training_internal_predictions.csv", "seed_metrics.csv", "frozen_test_initial_states_30.csv",
        "frozen_test_mpc_trajectories.csv", "frozen_test_audit.csv", "frozen_test_dnn_closed_loop.csv",
        "ndc_dnn_model.npz",
    ]
    can_resume = resume and all((output / name).exists() for name in resume_files)
    if can_resume:
        design = pd.read_csv(output / "training_initial_states.csv")
        training = pd.read_csv(output / "training_teacher_2000.csv")
        training_audit = pd.read_csv(output / "training_trajectory_audit.csv")
        internal_predictions = pd.read_csv(output / "training_internal_predictions.csv")
        seed_metrics = pd.read_csv(output / "seed_metrics.csv")
        frozen_test = pd.read_csv(output / "frozen_test_initial_states_30.csv")
        test_teacher = pd.read_csv(output / "frozen_test_mpc_trajectories.csv")
        test_audit = pd.read_csv(output / "frozen_test_audit.csv")
        dnn_closed_loop = pd.read_csv(output / "frozen_test_dnn_closed_loop.csv")
        network = PaperNDCNetwork.load(output / "ndc_dnn_model.npz")
        selected_seed = int(seed_metrics.sort_values(["internal_test_rmse_a", "seed"]).iloc[0]["seed"])
        network_metrics = {
            "architecture": list(network.layer_sizes),
            "parameter_count": network.parameter_count,
            "selected_seed": selected_seed,
            "normalization": "MATLAB-style mapminmax to [-1,1] inferred because the paper does not report preprocessing",
            "training_algorithm": "Bayesian evidence updates around regularized nonlinear least squares (BR-LM approximation)",
            "split_metrics": {
                str(name): {
                    "sample_count": int(len(group)),
                    "rmse_a": float(np.sqrt(np.mean(group["dnn_error_a"].to_numpy(dtype=float) ** 2))),
                    "mae_a": float(np.mean(np.abs(group["dnn_error_a"].to_numpy(dtype=float)))),
                    "maximum_absolute_error_a": float(np.max(np.abs(group["dnn_error_a"].to_numpy(dtype=float)))),
                    "nrmse_percent": _nrmse_percent(group["teacher_current_a"].to_numpy(dtype=float), group["dnn_current_a"].to_numpy(dtype=float)),
                }
                for name, group in internal_predictions.groupby("split")
            },
            "resumed_from_saved_artifacts": True,
        }
    else:
        design = generate_training_initial_states(model)
        frozen_test = generate_frozen_test_initial_states(model)
        training, training_audit = _build_training_dataset(model, controller, design)
        if len(training) != 2000 or not bool(training_audit["complete"].all()):
            raise RuntimeError(f"论文训练合同未满足：得到 {len(training)} 个样本，完整轨迹 {int(training_audit['complete'].sum())}/400。")
        network, seed_metrics, internal_predictions, network_metrics = train_paper_ndc_network(training, config)
        test_teacher, test_audit = _build_test_teacher(model, controller, frozen_test)
        if len(test_teacher) != 4500 or not bool(test_audit["complete"].all()):
            raise RuntimeError(f"冻结测试合同未满足：得到 {len(test_teacher)} 个样本，完整轨迹 {int(test_audit['complete'].sum())}/30。")

    offline_prediction = network.predict(test_teacher[list(FEATURES)].to_numpy())
    offline = {
        "nrmse_percent": _nrmse_percent(test_teacher["teacher_current_a"].to_numpy(), offline_prediction),
        "rmse_a": float(np.sqrt(np.mean((offline_prediction - test_teacher["teacher_current_a"].to_numpy()) ** 2))),
        "maximum_absolute_error_a": float(np.max(np.abs(offline_prediction - test_teacher["teacher_current_a"].to_numpy()))),
    }
    test_offline = test_teacher.copy()
    test_offline["dnn_open_loop_current_a"] = offline_prediction
    test_offline["dnn_open_loop_error_a"] = offline_prediction - test_offline["teacher_current_a"]
    if can_resume:
        _, inference_time = _rollout_dnn(model, network, frozen_test)
    else:
        dnn_closed_loop, inference_time = _rollout_dnn(model, network, frozen_test)
    closed_loop, trajectory_metrics = _closed_loop_metrics(config, test_teacher, dnn_closed_loop, inference_time)
    maximum_violation = max(closed_loop["constraint_violations"].values())
    seed_pass_count = int((seed_metrics["internal_test_nrmse_percent"] < config.gates.offline_nrmse_percent_max).sum())
    seed_count = int(len(seed_metrics))
    network_metrics.update(
        {
            "seed_count": seed_count,
            "passing_seed_count": seed_pass_count,
            "seed_internal_nrmse_mean_percent": float(seed_metrics["internal_test_nrmse_percent"].mean()),
            "seed_internal_nrmse_std_percent": float(seed_metrics["internal_test_nrmse_percent"].std(ddof=1)),
            "seed_internal_nrmse_best_percent": float(seed_metrics["internal_test_nrmse_percent"].min()),
        }
    )
    gates = {
        "offline_nrmse": offline["nrmse_percent"] < config.gates.offline_nrmse_percent_max,
        "closed_loop_current_nrmse": closed_loop["mean_trajectory_current_nrmse_percent"] < config.gates.closed_loop_current_nrmse_percent_max,
        "target_reach": closed_loop["target_reach_fraction"] >= config.gates.minimum_target_reach_fraction,
        "constraint_violation": maximum_violation <= config.gates.maximum_constraint_violation_order_a_or_v,
        "speedup": closed_loop["timing"]["speedup"] > config.gates.minimum_speedup,
        "majority_seeds": seed_pass_count > seed_count / 2,
    }
    metrics: dict[str, Any] = {
        "study_name": config.study_name,
        "source_doi": config.source_doi,
        "data": {
            "training_initial_states": int(len(design)),
            "complete_training_trajectories": int(training_audit["complete"].sum()),
            "training_samples": int(len(training)),
            "complete_test_trajectories": int(test_audit["complete"].sum()),
            "test_samples": int(len(test_teacher)),
        },
        "network": network_metrics,
        "offline": offline,
        "closed_loop": closed_loop,
        "maximum_constraint_violation": maximum_violation,
        "gates": gates,
        "success": bool(all(gates.values())),
    }

    design.to_csv(output / "training_initial_states.csv", index=False)
    training.to_csv(output / "training_teacher_2000.csv", index=False)
    training_audit.to_csv(output / "training_trajectory_audit.csv", index=False)
    internal_predictions.to_csv(output / "training_internal_predictions.csv", index=False)
    seed_metrics.to_csv(output / "seed_metrics.csv", index=False)
    frozen_test.to_csv(output / "frozen_test_initial_states_30.csv", index=False)
    test_teacher.to_csv(output / "frozen_test_mpc_trajectories.csv", index=False)
    test_audit.to_csv(output / "frozen_test_audit.csv", index=False)
    test_offline.to_csv(output / "frozen_test_open_loop_predictions.csv", index=False)
    dnn_closed_loop.to_csv(output / "frozen_test_dnn_closed_loop.csv", index=False)
    trajectory_metrics.to_csv(output / "closed_loop_trajectory_metrics.csv", index=False)
    network.save(output / "ndc_dnn_model.npz")
    figure_paths = _plot_results(output, design, training, test_teacher, dnn_closed_loop, seed_metrics)
    metrics["artifacts"] = {"figures": figure_paths}
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    _write_report(output / "PHASE6P0_NDC论文原位复现报告.md", metrics)
    return metrics
