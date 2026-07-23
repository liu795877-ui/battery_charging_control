"""Phase 7B-0：冻结 Level 3P 控制器在 25 ℃ Chen2020 DFN 上的闭环审计。"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybamm

from .ann_model import TinyANN
from .phase7a_level1_config import load_phase7a_level1_config
from .phase7a_level1_runner import _regression_metrics
from .phase7a_level2_config import load_phase7a_level2_config
from .phase7a_level3_config import load_phase7a_level3_config
from .phase7a_level3_model import Level3MPC, Level3Model, Level3State
from .phase7a_level3p_runner import project_current
from .phase7b0_config import Phase7B0Config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.suffix.lower() in {".py", ".yaml", ".yml"}:
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        return digest.hexdigest()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_artifacts(
    config: Phase7B0Config, root: Path
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for relative, expected in config.frozen_artifacts.items():
        actual = _sha256(root / relative)
        records[relative] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matched": actual == expected,
        }
    mismatched = [key for key, value in records.items() if not value["matched"]]
    if mismatched:
        raise RuntimeError(f"Phase 7B-0 冻结工件哈希不匹配：{mismatched}")
    return records


class Chen2020IsothermalDFN:
    """固定 25 ℃、允许外部逐步输入充电电流的 Chen2020 DFN。"""

    def __init__(
        self, config: Phase7B0Config, initial_soc: float, sample_period_s: float
    ) -> None:
        os.environ.setdefault("PYBAMM_DISABLE_TELEMETRY", "true")
        pybamm.set_logging_level("ERROR")
        model = pybamm.lithium_ion.DFN(options={"thermal": "isothermal"})
        parameters = pybamm.ParameterValues(config.dfn.parameter_set)
        parameters.update(
            {
                "Ambient temperature [K]": config.dfn.temperature_c + 273.15,
                "Initial temperature [K]": config.dfn.temperature_c + 273.15,
                "Upper voltage cut-off [V]": config.dfn.upper_voltage_cutoff_v,
                "Current function [A]": pybamm.InputParameter(
                    "phase7b0_applied_current_a"
                ),
            }
        )
        self.initial_soc = float(initial_soc)
        self.nominal_capacity_ah = float(
            parameters["Nominal cell capacity [A.h]"]
        )
        self.sample_period_s = float(sample_period_s)
        self.simulation = pybamm.Simulation(
            model,
            parameter_values=parameters,
            solver=pybamm.CasadiSolver(mode="safe"),
        )
        self.simulation.build(initial_soc=self.initial_soc)

    def step(self, charge_current_a: float) -> dict[str, float]:
        solution = self.simulation.step(
            self.sample_period_s,
            inputs={"phase7b0_applied_current_a": -float(charge_current_a)},
            save=False,
        )

        def last(name: str) -> float:
            return float(np.asarray(solution[name].entries).reshape(-1)[-1])

        discharge_capacity_ah = last("Discharge capacity [A.h]")
        return {
            "time_s": last("Time [s]"),
            "soc": self.initial_soc
            - discharge_capacity_ah / self.nominal_capacity_ah,
            "terminal_voltage_v": last("Terminal voltage [V]"),
        }


def _load_context(config: Phase7B0Config, root: Path):
    level3 = load_phase7a_level3_config(root / config.source_level3_config)
    level2 = load_phase7a_level2_config(root / level3.source_level2_config)
    inherited = load_phase7a_level1_config(root / level2.source_level1_config)
    model = Level3Model(level3, inherited, root)
    networks = {
        seed: TinyANN.load(
            root
            / config.source_model_directory
            / f"level3_deep_lbfgs_seed_{seed}.npz"
        )
        for seed in inherited.network.initialization_seeds
    }
    return level3, inherited, model, networks


def _correct_controller_state(
    predicted: Level3State, measurement: dict[str, float], current_a: float
) -> Level3State:
    return Level3State(
        soc=float(measurement["soc"]),
        polarization_1_v=predicted.polarization_1_v,
        polarization_2_v=predicted.polarization_2_v,
        previous_current_a=float(current_a),
    )


def _run_rollout(
    config: Phase7B0Config,
    model: Level3Model,
    initial: pd.Series,
    controller: str,
    seed: int | None,
    network: TinyANN | None,
) -> pd.DataFrame:
    state = Level3State(
        float(initial.initial_soc),
        float(initial.initial_polarization_1_v),
        float(initial.initial_polarization_2_v),
        float(initial.initial_previous_current_a),
    )
    plant = Chen2020IsothermalDFN(
        config, state.soc, model.config.model.sample_period_s
    )
    mpc = Level3MPC(model) if controller == "mpc" else None
    lower_bound, upper_bound = model.inherited.mpc.current_bounds_a
    target = model.inherited.mpc.target_soc - config.dfn.target_soc_tolerance
    rows: list[dict[str, Any]] = []
    for step in range(config.dfn.maximum_steps):
        started = perf_counter()
        if mpc is not None:
            result = mpc.solve(state)
            raw = float(result.current_a)
            safe = raw
            feasible_lower = max(
                lower_bound,
                state.previous_current_a
                - model.config.constraint.maximum_current_step_a,
            )
            feasible_upper = min(
                upper_bound,
                state.previous_current_a
                + model.config.constraint.maximum_current_step_a,
            )
            optimizer_success = result.optimizer_success
            prediction_feasible = result.prediction_feasible
        else:
            assert network is not None
            raw = float(
                network.predict(
                    np.asarray(
                        [
                            state.soc,
                            state.polarization_1_v,
                            state.polarization_2_v,
                            state.previous_current_a,
                        ]
                    )
                )
            )
            safe, feasible_lower, feasible_upper = project_current(
                raw,
                state.previous_current_a,
                lower_bound,
                upper_bound,
                model.config.constraint.maximum_current_step_a,
            )
            optimizer_success = True
            prediction_feasible = True
        decision_time_s = perf_counter() - started
        previous_soc = state.soc
        previous_current = state.previous_current_a
        predicted = model.step(state, safe)
        measurement = plant.step(safe)
        state = _correct_controller_state(predicted, measurement, safe)
        rows.append(
            {
                "controller": controller,
                "seed": -1 if seed is None else seed,
                "trajectory_id": str(initial.trajectory_id),
                "step_index": step,
                "time_s": (step + 1) * model.config.model.sample_period_s,
                "soc": previous_soc,
                "next_soc": state.soc,
                "polarization_1_v": predicted.polarization_1_v,
                "polarization_2_v": predicted.polarization_2_v,
                "previous_current_a": previous_current,
                "raw_current_a": raw,
                "current_a": safe,
                "current_step_a": abs(safe - previous_current),
                "feasible_lower_a": feasible_lower,
                "feasible_upper_a": feasible_upper,
                "projection_intervened": abs(safe - raw) > 1.0e-12,
                "terminal_voltage_v": measurement["terminal_voltage_v"],
                "decision_time_s": decision_time_s,
                "optimizer_success": optimizer_success,
                "prediction_feasible": prediction_feasible,
            }
        )
        if state.soc >= target:
            break
    return pd.DataFrame(rows)


def _continuous_crossing_time(
    frame: pd.DataFrame, threshold: float, sample_period_s: float
) -> float:
    for row in frame.sort_values("step_index").itertuples():
        before, after = float(row.soc), float(row.next_soc)
        if before >= threshold:
            return float(row.step_index) * sample_period_s
        if after >= threshold and after > before:
            fraction = np.clip((threshold - before) / (after - before), 0.0, 1.0)
            return float((row.step_index + fraction) * sample_period_s)
    return float("nan")


def _direction_reversals(values: np.ndarray, threshold: float) -> int:
    changes = np.diff(values)
    signs = np.sign(changes[np.abs(changes) >= threshold])
    return int(np.sum(signs[1:] * signs[:-1] < 0)) if len(signs) > 1 else 0


def _unexpected_early_taper_count(
    frame: pd.DataFrame, soc_threshold: float, current_threshold: float
) -> int:
    currents = frame.current_a.to_numpy(float)
    soc = frame.soc.to_numpy(float)
    previously_high = np.maximum.accumulate(currents) >= current_threshold
    return int(
        np.sum(
            previously_high
            & (soc < soc_threshold)
            & (currents < current_threshold)
        )
    )


def _evaluate(
    config: Phase7B0Config,
    level3: Any,
    inherited: Any,
    trajectories: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    baseline = trajectories[trajectories.controller == "mpc"]
    threshold = inherited.mpc.target_soc - config.dfn.target_soc_tolerance
    dt = float(level3.model.sample_period_s)
    diagnostics: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for seed in inherited.network.initialization_seeds:
        ann = trajectories[
            (trajectories.controller == "ann_projection")
            & (trajectories.seed == seed)
        ]
        per = []
        for trajectory_id, mpc_group in baseline.groupby("trajectory_id"):
            ann_group = ann[ann.trajectory_id == trajectory_id]
            paired = mpc_group[["step_index", "current_a"]].merge(
                ann_group[["step_index", "current_a"]],
                on="step_index",
                suffixes=("_mpc", "_ann"),
            )
            regression = _regression_metrics(
                paired.current_a_mpc.to_numpy(float),
                paired.current_a_ann.to_numpy(float),
            )
            mpc_reached = float(mpc_group.next_soc.iloc[-1]) >= threshold
            ann_reached = float(ann_group.next_soc.iloc[-1]) >= threshold
            discrete_gap = abs(len(ann_group) - len(mpc_group)) / len(mpc_group)
            mpc_continuous = _continuous_crossing_time(mpc_group, threshold, dt)
            ann_continuous = _continuous_crossing_time(ann_group, threshold, dt)
            per.append((regression["nrmse"], discrete_gap, ann_reached))
            diagnostics.append(
                {
                    "seed": seed,
                    "trajectory_id": trajectory_id,
                    "current_nrmse": regression["nrmse"],
                    "current_bias_a": regression["bias_a"],
                    "mpc_reached_target": mpc_reached,
                    "ann_reached_target": ann_reached,
                    "signed_step_difference": len(ann_group) - len(mpc_group),
                    "discrete_time_gap_fraction": discrete_gap,
                    "continuous_crossing_time_difference_s": (
                        ann_continuous - mpc_continuous
                    ),
                    "maximum_ann_voltage_v": float(
                        ann_group.terminal_voltage_v.max()
                    ),
                    "maximum_mpc_voltage_v": float(
                        mpc_group.terminal_voltage_v.max()
                    ),
                    "ann_early_taper_count": _unexpected_early_taper_count(
                        ann_group,
                        config.diagnostics.early_taper_soc_threshold,
                        config.diagnostics.early_taper_current_threshold_a,
                    ),
                    "mpc_early_taper_count": _unexpected_early_taper_count(
                        mpc_group,
                        config.diagnostics.early_taper_soc_threshold,
                        config.diagnostics.early_taper_current_threshold_a,
                    ),
                    "ann_direction_reversals": _direction_reversals(
                        ann_group.current_a.to_numpy(float),
                        config.diagnostics.oscillation_delta_threshold_a,
                    ),
                    "mpc_direction_reversals": _direction_reversals(
                        mpc_group.current_a.to_numpy(float),
                        config.diagnostics.oscillation_delta_threshold_a,
                    ),
                }
            )
        voltage_violation = np.maximum(
            ann.terminal_voltage_v.to_numpy(float)
            - config.dfn.physical_voltage_limit_v,
            0.0,
        )
        current_violation = np.maximum(
            np.maximum(
                ann.current_a.to_numpy(float) - inherited.mpc.current_bounds_a[1],
                inherited.mpc.current_bounds_a[0]
                - ann.current_a.to_numpy(float),
            ),
            0.0,
        )
        slew_violation = np.maximum(
            ann.current_step_a.to_numpy(float)
            - level3.constraint.maximum_current_step_a,
            0.0,
        )
        metrics.append(
            {
                "seed": seed,
                "mean_current_nrmse": float(np.mean([value[0] for value in per])),
                "maximum_current_nrmse": float(np.max([value[0] for value in per])),
                "mean_charge_time_gap_fraction": float(
                    np.mean([value[1] for value in per])
                ),
                "target_reach_fraction": float(
                    np.mean([value[2] for value in per])
                ),
                "maximum_voltage_v": float(ann.terminal_voltage_v.max()),
                "maximum_voltage_violation_v": float(np.max(voltage_violation)),
                "maximum_current_violation_a": float(np.max(current_violation)),
                "maximum_slew_violation_a": float(np.max(slew_violation)),
                "maximum_current_step_a": float(ann.current_step_a.max()),
                "projection_intervention_count": int(
                    ann.projection_intervened.astype(bool).sum()
                ),
                "projection_intervention_fraction": float(
                    ann.projection_intervened.astype(bool).mean()
                ),
                "speedup": float(
                    baseline.decision_time_s.sum() / ann.decision_time_s.sum()
                ),
            }
        )
    metrics_frame = pd.DataFrame(metrics)
    diagnostics_frame = pd.DataFrame(diagnostics)
    baseline_summary = {
        "trajectory_count": int(baseline.trajectory_id.nunique()),
        "target_reach_fraction": float(
            baseline.groupby("trajectory_id").next_soc.last().ge(threshold).mean()
        ),
        "maximum_voltage_v": float(baseline.terminal_voltage_v.max()),
        "maximum_voltage_violation_v": float(
            max(
                baseline.terminal_voltage_v.max()
                - config.dfn.physical_voltage_limit_v,
                0.0,
            )
        ),
        "optimizer_success_fraction": float(
            baseline.optimizer_success.astype(bool).mean()
        ),
        "prediction_feasible_fraction": float(
            baseline.prediction_feasible.astype(bool).mean()
        ),
    }
    return metrics_frame, diagnostics_frame, baseline_summary


def _plot(
    output: Path, trajectories: pd.DataFrame, metrics: pd.DataFrame
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    representative = str(
        trajectories.loc[
            trajectories.terminal_voltage_v.idxmax(), "trajectory_id"
        ]
    )
    subset = trajectories[trajectories.trajectory_id == representative]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    mpc = subset[subset.controller == "mpc"]
    axes[0].plot(mpc.time_s, mpc.current_a, color="black", label="MPC")
    for seed, group in subset[subset.controller == "ann_projection"].groupby("seed"):
        axes[0].plot(group.time_s, group.current_a, alpha=0.7, label=f"ANN {seed}")
    axes[0].set(xlabel="Time [s]", ylabel="Current [A]", title=representative)
    axes[1].plot(mpc.time_s, mpc.terminal_voltage_v, color="black", label="MPC")
    for _, group in subset[subset.controller == "ann_projection"].groupby("seed"):
        axes[1].plot(group.time_s, group.terminal_voltage_v, alpha=0.7)
    axes[1].axhline(4.2, color="red", linestyle="--")
    axes[1].set(xlabel="Time [s]", ylabel="DFN voltage [V]", title="Voltage safety")
    axes[2].bar(metrics.seed.astype(str), 100 * metrics.mean_current_nrmse)
    axes[2].axhline(1.0, color="red", linestyle="--")
    axes[2].set(xlabel="Seed", ylabel="Current NRMSE [%]", title="Cross-model gate")
    axes[0].legend(fontsize=7, ncol=2)
    fig.savefig(output / "phase7b0_cross_model_audit.png", dpi=180)
    plt.close(fig)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    c = payload["cross_model"]
    b = payload["mpc_baseline"]
    d = payload["decision"]
    path.write_text(
        f"""# Phase 7B-0：Level 3P 控制器的 25 ℃ DFN 跨模型审计

## 结论

**{d['conclusion']}**

本阶段没有训练新 ANN、没有增加教师数据、没有加入温度状态或参数扰动。唯一变化是把闭环被控对象由 2RC 换成固定 25 ℃ 的 Chen2020 DFN；控制器仍使用冻结的 2RC 状态接口，其中 SOC 由 DFN 反馈校正，两个极化状态按冻结 2RC 模型传播。

## 冻结合同

- 冻结工件哈希：{payload['frozen_contract']['matched_count']}/{payload['frozen_contract']['artifact_count']} 匹配。
- 固定网络种子：{payload['frozen_contract']['network_seeds']}。
- 初始状态：{payload['frozen_contract']['initial_state_count']} 条。
- 采样周期：{payload['frozen_contract']['sample_period_s']:.1f} s。
- 电流/斜率投影未修改。

## DFN 上的 MPC 基线

- 目标到达率：{100*b['target_reach_fraction']:.1f}%。
- 最大端电压：{b['maximum_voltage_v']:.6f} V。
- 最大电压越界：{b['maximum_voltage_violation_v']:.6e} V。
- 优化器成功率：{100*b['optimizer_success_fraction']:.2f}%。

## 冻结 ANN＋投影五种子

- 平均电流 NRMSE 范围：{100*c['current_nrmse_min']:.4f}%–{100*c['current_nrmse_max']:.4f}%。
- 最大平均离散充电时间偏差：{100*c['maximum_charge_time_gap_fraction']:.4f}%。
- 最低目标到达率：{100*c['minimum_target_reach_fraction']:.1f}%。
- 最大 DFN 端电压：{c['maximum_voltage_v']:.6f} V。
- 最大电压越界：{c['maximum_voltage_violation_v']:.6e} V。
- 最大单步电流变化：{c['maximum_current_step_a']:.6f} A。
- 最大电流/斜率越界：{c['maximum_current_violation_a']:.3e} A / {c['maximum_slew_violation_a']:.3e} A。
- 最低在线控制器加速：{c['minimum_speedup']:.1f}×（不计 DFN 被控对象求解时间）。
- 投影介入率范围：{100*c['projection_fraction_min']:.4f}%–{100*c['projection_fraction_max']:.4f}%。
- 异常提前降流计数最大值：{c['maximum_early_taper_count']}。
- 相对 MPC 的提前降流计数最大增量：{c['maximum_early_taper_excess_vs_mpc']}（五种子均值差 {c['mean_early_taper_excess_vs_mpc']:.3f}）。
- 电流方向反转计数最大值：{c['maximum_direction_reversals']}。
- 连续目标穿越时间最大绝对偏差：{c['maximum_absolute_continuous_crossing_time_difference_s']:.3f} s。

## 严格门槛

```json
{json.dumps(d['checks'], ensure_ascii=False, indent=2)}
```

若仅电压门槛失败，应把原因判定为跨模型状态约束失配：电流/斜率投影只保证输入约束，不能保证 DFN 端电压。下一步应优先评估电压感知安全修正，而不是重新扩大 pure ANN。
""",
        encoding="utf-8",
    )


def _json_default(value: object) -> object:
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    raise TypeError(type(value).__name__)


def run_phase7b0(
    config: Phase7B0Config,
    project_root: str | Path,
    resume: bool = False,
    limit_trajectories: int | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    hashes = verify_frozen_artifacts(config, root)
    level3, inherited, model, networks = _load_context(config, root)
    initial = pd.read_csv(root / config.source_initial_states)
    if limit_trajectories is not None:
        initial = initial.head(limit_trajectories)
    data_dir = root / config.output.data_directory
    output = root / config.output.result_directory
    run_dir = data_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, int | None, TinyANN | None, pd.Series]] = []
    for _, state in initial.iterrows():
        jobs.append(("mpc", None, None, state))
    for seed, network in networks.items():
        for _, state in initial.iterrows():
            jobs.append(("ann_projection", seed, network, state))

    frames = []
    for index, (controller, seed, network, state) in enumerate(jobs, start=1):
        seed_label = "baseline" if seed is None else str(seed)
        path = run_dir / f"{controller}_{seed_label}_{state.trajectory_id}.csv"
        if resume and path.exists():
            frame = pd.read_csv(path)
        else:
            print(
                f"[Phase 7B-0] {index}/{len(jobs)} {controller} "
                f"seed={seed_label} trajectory={state.trajectory_id}",
                flush=True,
            )
            frame = _run_rollout(
                config, model, state, controller, seed, network
            )
            frame.to_csv(path, index=False)
        frames.append(frame)
    trajectories = pd.concat(frames, ignore_index=True)
    trajectories.to_csv(data_dir / "closed_loop_trajectories.csv", index=False)
    metrics, diagnostics, baseline = _evaluate(
        config, level3, inherited, trajectories
    )
    metrics.to_csv(data_dir / "five_seed_metrics.csv", index=False)
    diagnostics.to_csv(data_dir / "trajectory_diagnostics.csv", index=False)
    _plot(output, trajectories, metrics)

    cross = {
        "current_nrmse_min": float(metrics.mean_current_nrmse.min()),
        "current_nrmse_max": float(metrics.mean_current_nrmse.max()),
        "maximum_charge_time_gap_fraction": float(
            metrics.mean_charge_time_gap_fraction.max()
        ),
        "minimum_target_reach_fraction": float(
            metrics.target_reach_fraction.min()
        ),
        "maximum_voltage_v": float(metrics.maximum_voltage_v.max()),
        "maximum_voltage_violation_v": float(
            metrics.maximum_voltage_violation_v.max()
        ),
        "maximum_current_violation_a": float(
            metrics.maximum_current_violation_a.max()
        ),
        "maximum_slew_violation_a": float(
            metrics.maximum_slew_violation_a.max()
        ),
        "maximum_current_step_a": float(metrics.maximum_current_step_a.max()),
        "minimum_speedup": float(metrics.speedup.min()),
        "projection_fraction_min": float(
            metrics.projection_intervention_fraction.min()
        ),
        "projection_fraction_max": float(
            metrics.projection_intervention_fraction.max()
        ),
        "maximum_early_taper_count": int(
            diagnostics.ann_early_taper_count.max()
        ),
        "maximum_early_taper_excess_vs_mpc": int(
            (
                diagnostics.ann_early_taper_count
                - diagnostics.mpc_early_taper_count
            ).max()
        ),
        "mean_early_taper_excess_vs_mpc": float(
            (
                diagnostics.ann_early_taper_count
                - diagnostics.mpc_early_taper_count
            ).mean()
        ),
        "maximum_direction_reversals": int(
            diagnostics.ann_direction_reversals.max()
        ),
        "maximum_absolute_continuous_crossing_time_difference_s": float(
            diagnostics.continuous_crossing_time_difference_s.abs().max()
        ),
    }
    checks = {
        "all_frozen_hashes_match": all(
            item["matched"] for item in hashes.values()
        ),
        "no_new_training_or_teacher_data": True,
        "mpc_target_reach_100_percent": baseline["target_reach_fraction"] >= 1.0,
        "mpc_dfn_voltage_safe": (
            baseline["maximum_voltage_violation_v"]
            <= config.dfn.voltage_tolerance_v
        ),
        "all_seed_current_nrmse_below_1_percent": bool(
            (metrics.mean_current_nrmse < inherited.gates.closed_loop_current_nrmse_max).all()
        ),
        "all_seed_charge_time_gap_below_2_percent": bool(
            (
                metrics.mean_charge_time_gap_fraction
                < inherited.gates.charge_time_gap_fraction_max
            ).all()
        ),
        "all_seed_target_reach_100_percent": bool(
            (
                metrics.target_reach_fraction
                >= inherited.gates.minimum_target_reach_fraction
            ).all()
        ),
        "ann_dfn_voltage_safe": bool(
            (
                metrics.maximum_voltage_violation_v
                <= config.dfn.voltage_tolerance_v
            ).all()
        ),
        "current_bounds_strictly_satisfied": bool(
            (metrics.maximum_current_violation_a <= 1.0e-12).all()
        ),
        "slew_bound_strictly_satisfied": bool(
            (metrics.maximum_slew_violation_a <= 1.0e-12).all()
        ),
        "all_seed_speedup_above_100": bool(
            (metrics.speedup > inherited.gates.minimum_speedup).all()
        ),
    }
    success = bool(all(checks.values()))
    voltage_only_failure = bool(
        not success
        and not checks["ann_dfn_voltage_safe"]
        and all(
            value
            for key, value in checks.items()
            if key not in {"ann_dfn_voltage_safe", "mpc_dfn_voltage_safe"}
        )
    )
    if success:
        conclusion = "Phase 7B-0 严格通过：冻结 ANN＋投影在 25 ℃ DFN 上保持安全与性能"
    elif voltage_only_failure:
        conclusion = "Phase 7B-0 仅电压安全失败：进入电压感知安全层，不重新训练 ANN"
    else:
        conclusion = "Phase 7B-0 未严格通过：按失败指标定位跨模型失配，不进入多温度或扰动阶段"
    payload = {
        "study_name": config.study_name,
        "configuration": asdict(config),
        "frozen_artifact_verification": hashes,
        "frozen_contract": {
            "artifact_count": len(hashes),
            "matched_count": sum(item["matched"] for item in hashes.values()),
            "network_seeds": list(networks),
            "initial_state_count": len(initial),
            "sample_period_s": level3.model.sample_period_s,
        },
        "mpc_baseline": baseline,
        "cross_model": cross,
        "decision": {
            "checks": checks,
            "phase7b0_success": success,
            "voltage_only_failure": voltage_only_failure,
            "proceed_to_multi_temperature": success,
            "conclusion": conclusion,
        },
        "status": "completed",
        "success": success,
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    _write_report(output / "PHASE7B0_中文实验报告.md", payload)
    print(json.dumps(payload["decision"], ensure_ascii=False), flush=True)
    return payload
