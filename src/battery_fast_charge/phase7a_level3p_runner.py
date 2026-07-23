"""执行 Phase 7A Level 3P 的冻结模型最小输出投影验证。"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ann_model import TinyANN
from .phase7a_level1_config import load_phase7a_level1_config
from .phase7a_level1_runner import _regression_metrics
from .phase7a_level1s_runner import continuous_crossing_time_s
from .phase7a_level2_config import load_phase7a_level2_config
from .phase7a_level3_config import load_phase7a_level3_config
from .phase7a_level3_model import Level3Model, Level3State
from .phase7a_level3p_config import Phase7ALevel3PConfig


FEATURES = (
    "state_soc",
    "state_polarization_1_v",
    "state_polarization_2_v",
    "state_previous_current_a",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_artifacts(
    config: Phase7ALevel3PConfig, root: Path
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    failures = []
    for relative, expected in config.frozen_artifacts.items():
        path = root / relative
        actual = _sha256(path) if path.is_file() else ""
        matched = actual == expected
        records[relative] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matched": matched,
        }
        if not matched:
            failures.append(relative)
    if failures:
        raise RuntimeError(f"Level 3 冻结工件哈希不一致：{failures}")
    return records


def project_current(
    raw_current_a: float,
    previous_current_a: float,
    minimum_current_a: float = 0.0,
    maximum_current_a: float = 10.0,
    maximum_current_step_a: float = 2.0,
) -> tuple[float, float, float]:
    lower = max(minimum_current_a, previous_current_a - maximum_current_step_a)
    upper = min(maximum_current_a, previous_current_a + maximum_current_step_a)
    safe = min(upper, max(lower, raw_current_a))
    return float(safe), float(lower), float(upper)


def _load_context(config: Phase7ALevel3PConfig, root: Path):
    level3 = load_phase7a_level3_config(root / config.source_level3_config)
    level2 = load_phase7a_level2_config(root / level3.source_level2_config)
    inherited = load_phase7a_level1_config(root / level2.source_level1_config)
    model = Level3Model(level3, inherited, root)
    networks = {
        seed: TinyANN.load(
            root
            / "outputs"
            / "phase7a_level3_slew"
            / "models"
            / f"level3_deep_lbfgs_seed_{seed}.npz"
        )
        for seed in inherited.network.initialization_seeds
    }
    return level3, inherited, model, networks


def _rollout_projected(
    level3: Any,
    model: Level3Model,
    network: TinyANN,
    initial: pd.Series,
    seed: int,
    tolerance: float,
) -> list[dict[str, Any]]:
    state = Level3State(
        float(initial.initial_soc),
        float(initial.initial_polarization_1_v),
        float(initial.initial_polarization_2_v),
        float(initial.initial_previous_current_a),
    )
    rows = []
    lower_bound, upper_bound = model.inherited.mpc.current_bounds_a
    for step in range(level3.data.maximum_closed_loop_steps):
        started = perf_counter()
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
        safe, lower, upper = project_current(
            raw,
            state.previous_current_a,
            lower_bound,
            upper_bound,
            level3.constraint.maximum_current_step_a,
        )
        elapsed = perf_counter() - started
        delta = safe - raw
        voltage = model.terminal_voltage(state, safe)
        next_state = model.step(state, safe)
        rows.append(
            {
                "controller": "dnn_projection",
                "seed": seed,
                "trajectory_id": initial.trajectory_id,
                "step_index": step,
                "soc": state.soc,
                "polarization_1_v": state.polarization_1_v,
                "polarization_2_v": state.polarization_2_v,
                "previous_current_a": state.previous_current_a,
                "raw_current_a": raw,
                "safe_current_a": safe,
                "feasible_lower_a": lower,
                "feasible_upper_a": upper,
                "projection_delta_a": delta,
                "projection_intervened": abs(delta) > tolerance,
                "current_step_a": abs(safe - state.previous_current_a),
                "terminal_voltage_v": voltage,
                "next_soc": next_state.soc,
                "elapsed_s": elapsed,
            }
        )
        state = next_state
        if state.soc >= model.inherited.mpc.target_soc - 5e-4:
            break
    return rows


def _key(row: Any) -> tuple[int, str, int]:
    return int(float(row.seed)), str(row.trajectory_id), int(row.step_index)


def _intervention_locality(
    projected: pd.DataFrame,
    frozen_raw: pd.DataFrame,
    maximum_step_a: float,
    raw_tolerance_a: float,
    neighborhood_radius: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_violations = frozen_raw[
        frozen_raw.current_step_a > maximum_step_a + raw_tolerance_a
    ]
    interventions = projected[projected.projection_intervened.astype(bool)]
    raw_keys = {_key(row) for row in raw_violations.itertuples()}
    intervention_keys = {_key(row) for row in interventions.itertuples()}

    def near(key: tuple[int, str, int], references: set[tuple[int, str, int]]) -> bool:
        seed, trajectory, step = key
        return any(
            ref_seed == seed
            and ref_trajectory == trajectory
            and abs(ref_step - step) <= neighborhood_radius
            for ref_seed, ref_trajectory, ref_step in references
        )

    exact = intervention_keys & raw_keys
    near_interventions = {key for key in intervention_keys if near(key, raw_keys)}
    covered_raw = {key for key in raw_keys if near(key, intervention_keys)}
    records = []
    for row in interventions.itertuples():
        key = _key(row)
        records.append(
            {
                "seed": key[0],
                "trajectory_id": key[1],
                "step_index": key[2],
                "raw_current_a": row.raw_current_a,
                "safe_current_a": row.safe_current_a,
                "projection_delta_a": row.projection_delta_a,
                "exact_frozen_raw_violation": key in raw_keys,
                "within_one_step_of_frozen_raw_violation": key in near_interventions,
            }
        )
    summary = {
        "frozen_raw_violation_count": len(raw_keys),
        "projection_intervention_count": len(intervention_keys),
        "exact_key_overlap_count": len(exact),
        "interventions_within_one_step_count": len(near_interventions),
        "interventions_outside_one_step_count": len(intervention_keys - near_interventions),
        "frozen_raw_violations_covered_within_one_step_count": len(covered_raw),
        "all_interventions_near_frozen_raw_violations": intervention_keys <= near_interventions,
        "all_frozen_raw_violations_covered_nearby": raw_keys <= covered_raw,
    }
    return pd.DataFrame(records), summary


def _evaluate(
    level3: Any,
    inherited: Any,
    model: Level3Model,
    projected: pd.DataFrame,
    frozen: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    teacher = frozen[frozen.controller == "mpc"]
    metric_rows, diagnostic_rows = [], []
    threshold = inherited.mpc.target_soc - 5e-4
    dt = level3.model.sample_period_s
    for seed in inherited.network.initialization_seeds:
        dnn = projected[projected.seed == seed]
        per = []
        for trajectory_id, mpc_group in teacher.groupby("trajectory_id"):
            dnn_group = dnn[dnn.trajectory_id == trajectory_id]
            paired = mpc_group[["step_index", "current_a"]].merge(
                dnn_group[["step_index", "safe_current_a"]],
                on="step_index",
            )
            nrmse = _regression_metrics(
                paired.current_a.to_numpy(float),
                paired.safe_current_a.to_numpy(float),
            )["nrmse"]
            gap = abs(len(dnn_group) - len(mpc_group)) / len(mpc_group)
            per.append(
                (
                    nrmse,
                    gap,
                    float(dnn_group.next_soc.iloc[-1]) >= threshold,
                )
            )
            diagnostic_rows.append(
                {
                    "seed": seed,
                    "trajectory_id": trajectory_id,
                    "signed_step_difference": len(dnn_group) - len(mpc_group),
                    "continuous_crossing_time_difference_s": (
                        continuous_crossing_time_s(dnn_group, threshold, dt)
                        - continuous_crossing_time_s(mpc_group, threshold, dt)
                    ),
                    "cumulative_charge_error_ah": float(
                        (
                            dnn_group.safe_current_a.sum()
                            - mpc_group.current_a.sum()
                        )
                        * dt
                        / 3600.0
                    ),
                }
            )
        voltage_violation = np.maximum(
            dnn.terminal_voltage_v.to_numpy(float)
            - inherited.mpc.terminal_voltage_max_v,
            0.0,
        )
        current_violation = np.maximum(
            dnn.safe_current_a.to_numpy(float) - inherited.mpc.current_bounds_a[1],
            0.0,
        )
        slew_violation = np.maximum(
            dnn.current_step_a.to_numpy(float)
            - level3.constraint.maximum_current_step_a,
            0.0,
        )
        metric_rows.append(
            {
                "seed": seed,
                "mean_current_nrmse": float(np.mean([v[0] for v in per])),
                "maximum_current_nrmse": float(np.max([v[0] for v in per])),
                "mean_charge_time_gap_fraction": float(np.mean([v[1] for v in per])),
                "target_reach_fraction": float(np.mean([v[2] for v in per])),
                "maximum_voltage_violation_v": float(np.max(voltage_violation)),
                "maximum_current_violation_a": float(np.max(current_violation)),
                "maximum_slew_violation_a": float(np.max(slew_violation)),
                "maximum_current_step_a": float(dnn.current_step_a.max()),
                "projection_intervention_count": int(
                    dnn.projection_intervened.astype(bool).sum()
                ),
                "projection_intervention_fraction": float(
                    dnn.projection_intervened.astype(bool).mean()
                ),
                "maximum_projection_magnitude_a": float(
                    dnn.projection_delta_a.abs().max()
                ),
                "mean_projection_magnitude_when_active_a": float(
                    dnn.loc[
                        dnn.projection_intervened.astype(bool),
                        "projection_delta_a",
                    ].abs().mean()
                ),
                "mpc_time_s": float(teacher.elapsed_s.sum()),
                "projected_dnn_time_s": float(dnn.elapsed_s.sum()),
                "speedup": float(teacher.elapsed_s.sum() / dnn.elapsed_s.sum()),
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(diagnostic_rows)


def _plots(
    output: Path,
    raw_metrics: pd.DataFrame,
    projected_metrics: pd.DataFrame,
    interventions: pd.DataFrame,
) -> None:
    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    merged = raw_metrics[["seed", "maximum_current_step_a"]].merge(
        projected_metrics[["seed", "maximum_current_step_a"]],
        on="seed",
        suffixes=("_raw", "_projected"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(merged))
    axes[0].bar(x - 0.18, merged.maximum_current_step_a_raw, 0.36, label="Raw")
    axes[0].bar(
        x + 0.18, merged.maximum_current_step_a_projected, 0.36, label="Projected"
    )
    axes[0].axhline(2.0, color="red", linestyle="--")
    axes[0].set(
        xticks=x,
        xticklabels=merged.seed.astype(int).astype(str),
        ylabel="Maximum current step [A]",
        title="Hard slew constraint",
    )
    axes[0].legend()
    counts = interventions.groupby("seed").size().reindex(merged.seed, fill_value=0)
    axes[1].bar(merged.seed.astype(int).astype(str), counts)
    axes[1].set(
        xlabel="Seed",
        ylabel="Projected actions",
        title="Projection interventions",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "projection_gate_and_interventions.png", dpi=180)
    plt.close(fig)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    p = payload["projection"]
    c = payload["closed_loop"]
    d = payload["decision"]
    path.write_text(
        f"""# Phase 7A Level 3P：最小输出投影验证报告

## 结论

Level 3P 判定：**{d['conclusion']}**。本实验明确不进入 Level 4。

## 冻结合同

- Level 3 教师数据、双冻结测试、MPC 实现、闭环初态、离线指标和五种子模型共 13 个工件哈希全部匹配。
- 未新增教师数据，未重新训练网络，未改变 2RC 模型、MPC、初态、五个种子或验收门槛。
- 唯一变化为将 DNN 原始输出裁剪到 `[max(0,I_previous-2), min(10,I_previous+2)]`。

## 投影介入

- 冻结 Level 3 原始斜率风险动作：{p['frozen_raw_violation_count']} 步。
- Level 3P 实际介入：{p['projection_intervention_count']} 步，占全部动作 {100*p['projection_intervention_fraction']:.4f}%。
- 精确位置重合：{p['exact_key_overlap_count']} 步；位于原风险动作 ±1 步内：{p['interventions_within_one_step_count']} 步。
- ±1 步邻域外新增介入：{p['interventions_outside_one_step_count']} 步。
- 最大投影修正：{p['maximum_projection_magnitude_a']:.6f} A。

## 五种子闭环

- 电流 NRMSE：{100*c['current_nrmse_min']:.4f}%–{100*c['current_nrmse_max']:.4f}%。
- 最大平均充电时间偏差：{100*c['maximum_charge_time_gap_fraction']:.4f}%。
- 最低目标到达率：{100*c['minimum_target_reach_fraction']:.1f}%。
- 最大电压违约：{c['maximum_voltage_violation_v']:.6e} V。
- 最大斜率违约：{c['maximum_slew_violation_a']:.6e} A；最大单步变化 {c['maximum_current_step_a']:.6f} A。
- 最低在线加速：{c['minimum_speedup']:.1f}×。

## 严格门槛

```json
{json.dumps(d['checks'], ensure_ascii=False, indent=2)}
```

Level 3P 只验证最小安全投影能否修复 Level 3 已确认的硬斜率失效，不包含温度、DFN、扰动或 Level 4 内容。
""",
        encoding="utf-8",
    )


def _json_default(value: object) -> object:
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    raise TypeError(type(value).__name__)


def run_phase7a_level3p(
    config: Phase7ALevel3PConfig, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    data_dir = root / config.output.data_directory
    output = root / config.output.result_directory
    data_dir.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    hashes = verify_frozen_artifacts(config, root)
    level3, inherited, model, networks = _load_context(config, root)
    initial = pd.read_csv(
        root / "data" / "phase7a_level3_slew" / "closed_loop_initial_states.csv"
    )
    frozen = pd.read_csv(
        root / "data" / "phase7a_level3_slew" / "closed_loop_trajectories.csv"
    )
    rows = []
    for seed, network in networks.items():
        for _, state in initial.iterrows():
            rows.extend(
                _rollout_projected(
                    level3,
                    model,
                    network,
                    state,
                    seed,
                    config.projection.intervention_tolerance_a,
                )
            )
    projected = pd.DataFrame(rows)
    projected.to_csv(data_dir / "projected_closed_loop_trajectories.csv", index=False)
    frozen_raw = frozen[frozen.controller == "dnn"].copy()
    interventions, locality = _intervention_locality(
        projected,
        frozen_raw,
        level3.constraint.maximum_current_step_a,
        config.projection.frozen_raw_violation_tolerance_a,
        config.projection.neighborhood_radius_steps,
    )
    interventions.to_csv(data_dir / "projection_interventions.csv", index=False)
    metrics, diagnostics = _evaluate(level3, inherited, model, projected, frozen)
    metrics.to_csv(data_dir / "projected_closed_loop_metrics.csv", index=False)
    diagnostics.to_csv(data_dir / "projected_closed_loop_diagnostics.csv", index=False)
    raw_metrics = pd.read_csv(
        root / "data" / "phase7a_level3_slew" / "closed_loop_metrics.csv"
    )
    _plots(output, raw_metrics, metrics, interventions)

    projection = {
        **locality,
        "projection_intervention_fraction": float(
            projected.projection_intervened.astype(bool).mean()
        ),
        "maximum_projection_magnitude_a": float(
            projected.projection_delta_a.abs().max()
        ),
        "mean_projection_magnitude_when_active_a": float(
            projected.loc[
                projected.projection_intervened.astype(bool),
                "projection_delta_a",
            ].abs().mean()
        ),
    }
    closed = {
        "current_nrmse_min": float(metrics.mean_current_nrmse.min()),
        "current_nrmse_max": float(metrics.mean_current_nrmse.max()),
        "maximum_charge_time_gap_fraction": float(
            metrics.mean_charge_time_gap_fraction.max()
        ),
        "minimum_target_reach_fraction": float(metrics.target_reach_fraction.min()),
        "maximum_voltage_violation_v": float(
            metrics.maximum_voltage_violation_v.max()
        ),
        "maximum_current_violation_a": float(
            metrics.maximum_current_violation_a.max()
        ),
        "maximum_slew_violation_a": float(metrics.maximum_slew_violation_a.max()),
        "maximum_current_step_a": float(metrics.maximum_current_step_a.max()),
        "minimum_speedup": float(metrics.speedup.min()),
    }
    checks = {
        "all_frozen_hashes_match": all(v["matched"] for v in hashes.values()),
        "frozen_raw_violation_count_is_48": locality[
            "frozen_raw_violation_count"
        ]
        == 48,
        "zero_slew_violation": bool(
            (metrics.maximum_slew_violation_a <= 1e-12).all()
        ),
        "all_seed_current_nrmse": bool(
            (metrics.mean_current_nrmse < inherited.gates.closed_loop_current_nrmse_max).all()
        ),
        "all_seed_charge_time_gap": bool(
            (
                metrics.mean_charge_time_gap_fraction
                < inherited.gates.charge_time_gap_fraction_max
            ).all()
        ),
        "all_seed_target_reach": bool(
            (
                metrics.target_reach_fraction
                >= inherited.gates.minimum_target_reach_fraction
            ).all()
        ),
        "voltage_constraint_satisfied": bool(
            (
                metrics.maximum_voltage_violation_v
                <= inherited.gates.maximum_constraint_violation
            ).all()
        ),
        "current_bounds_satisfied": bool(
            (
                metrics.maximum_current_violation_a
                <= inherited.gates.maximum_constraint_violation
            ).all()
        ),
        "all_interventions_near_frozen_raw_violations": locality[
            "all_interventions_near_frozen_raw_violations"
        ],
        "all_seed_speedup": bool(
            (metrics.speedup > inherited.gates.minimum_speedup).all()
        ),
    }
    success = bool(all(checks.values()))
    payload = {
        "study_name": config.study_name,
        "configuration": asdict(config),
        "frozen_artifact_verification": hashes,
        "projection": projection,
        "closed_loop": closed,
        "decision": {
            "checks": checks,
            "level3p_success": success,
            "proceed_to_level4": False,
            "conclusion": (
                "Level 3P 最小投影严格通过，研究停止在 Level 3P"
                if success
                else "Level 3P 未严格通过，研究停止在 Level 3P"
            ),
        },
        "status": "completed",
        "success": success,
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    _write_report(output / "PHASE7A_LEVEL3P_中文实验报告.md", payload)
    print(
        json.dumps(payload["decision"], ensure_ascii=False, default=_json_default),
        flush=True,
    )
    return payload
