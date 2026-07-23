"""Phase 7B-1A：从冻结 7B-0 轨迹审计电压残差与斜率制动可行性。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .phase7a_level1_config import load_phase7a_level1_config
from .phase7a_level1_runner import _van_der_corput
from .phase7a_level2_config import load_phase7a_level2_config
from .phase7a_level3_config import load_phase7a_level3_config
from .phase7a_level3_model import Level3MPC, Level3Model, Level3State
from .phase7b1a_config import Phase7B1AConfig


GROUP_KEYS = ["controller", "seed", "trajectory_id"]


def _sha256(path: Path) -> str:
    payload = path.read_bytes()
    if path.suffix.lower() in {".py", ".yaml", ".yml"}:
        payload = payload.replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _verify_frozen(
    config: Phase7B1AConfig, root: Path
) -> dict[str, dict[str, Any]]:
    verification = {}
    failures = []
    for relative, expected in config.frozen_artifacts.items():
        actual = _sha256(root / relative)
        matched = actual == expected
        verification[relative] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matched": matched,
        }
        if not matched:
            failures.append(relative)
    if failures:
        raise RuntimeError(f"Phase 7B-1A 冻结工件不匹配：{failures}")
    return verification


def _load_model(config: Phase7B1AConfig, root: Path):
    level3 = load_phase7a_level3_config(root / config.source_level3_config)
    level2 = load_phase7a_level2_config(root / level3.source_level2_config)
    inherited = load_phase7a_level1_config(root / level2.source_level1_config)
    return level3, inherited, Level3Model(level3, inherited, root)


def _stage(voltage_v: pd.Series) -> pd.Series:
    return pd.cut(
        voltage_v,
        [-np.inf, 4.15, 4.18, 4.19, 4.20, np.inf],
        labels=["below_4.15", "4.15_4.18", "4.18_4.19", "4.19_4.20", "above_4.20"],
        right=False,
    ).astype(str)


def _augment_residuals(
    config: Phase7B1AConfig,
    level3: Any,
    model: Level3Model,
    trajectories: pd.DataFrame,
    initial: pd.DataFrame,
) -> pd.DataFrame:
    frame = trajectories.copy()
    dt = level3.model.sample_period_s
    capacity = level3.model.nominal_capacity_ah
    frame["predicted_next_soc_2rc"] = (
        frame.soc + frame.current_a * dt / (3600.0 * capacity)
    )
    frame["predicted_next_voltage_2rc_v"] = (
        model.ocv(frame.predicted_next_soc_2rc.to_numpy(float))
        + level3.model.r0_ohm * frame.current_a
        + frame.polarization_1_v
        + frame.polarization_2_v
    )
    frame["voltage_residual_v"] = (
        frame.terminal_voltage_v - frame.predicted_next_voltage_2rc_v
    )
    grouped = frame.groupby(GROUP_KEYS, sort=False)
    frame["residual_growth_v"] = grouped.voltage_residual_v.diff()
    frame["residual_before_step_v"] = grouped.voltage_residual_v.shift(1).fillna(0.0)
    frame["pre_polarization_1_v"] = grouped.polarization_1_v.shift(1)
    frame["pre_polarization_2_v"] = grouped.polarization_2_v.shift(1)
    initial_map = initial.set_index("trajectory_id")
    first = frame.pre_polarization_1_v.isna()
    frame.loc[first, "pre_polarization_1_v"] = frame.loc[
        first, "trajectory_id"
    ].map(initial_map.initial_polarization_1_v)
    frame.loc[first, "pre_polarization_2_v"] = frame.loc[
        first, "trajectory_id"
    ].map(initial_map.initial_polarization_2_v)
    frame["predicted_voltage_margin_v"] = (
        config.audit.voltage_limit_v - frame.predicted_next_voltage_2rc_v
    )
    frame["dfn_voltage_margin_v"] = (
        config.audit.voltage_limit_v - frame.terminal_voltage_v
    )
    frame["charge_stage"] = _stage(frame.terminal_voltage_v)
    return frame


def _predict_corrected_next_voltage(
    row: Any,
    current_a: float,
    level3: Any,
    model: Level3Model,
    guard_v: float,
) -> float:
    dt = level3.model.sample_period_s
    capacity = level3.model.nominal_capacity_ah
    predicted_soc = float(row.soc) + current_a * dt / (3600.0 * capacity)
    v1 = (
        model.decay_1 * float(row.pre_polarization_1_v)
        + level3.model.r1_ohm * (1.0 - model.decay_1) * current_a
    )
    v2 = (
        model.decay_2 * float(row.pre_polarization_2_v)
        + level3.model.r2_ohm * (1.0 - model.decay_2) * current_a
    )
    return float(
        model.ocv(predicted_soc)
        + level3.model.r0_ohm * current_a
        + v1
        + v2
        + float(row.residual_before_step_v)
        + guard_v
    )


def _maximum_voltage_safe_current(
    row: Any,
    config: Phase7B1AConfig,
    level3: Any,
    model: Level3Model,
    guard_v: float,
) -> float:
    voltage_limit = config.audit.voltage_limit_v
    if _predict_corrected_next_voltage(row, 0.0, level3, model, guard_v) > voltage_limit:
        return -1.0
    if _predict_corrected_next_voltage(row, 10.0, level3, model, guard_v) <= voltage_limit:
        return 10.0
    lower, upper = 0.0, 10.0
    while upper - lower > config.audit.current_search_tolerance_a:
        candidate = 0.5 * (lower + upper)
        voltage = _predict_corrected_next_voltage(
            row, candidate, level3, model, guard_v
        )
        if voltage <= voltage_limit:
            lower = candidate
        else:
            upper = candidate
    return lower


def _add_feasibility(
    frame: pd.DataFrame,
    config: Phase7B1AConfig,
    level3: Any,
    model: Level3Model,
    guard_v: float,
) -> pd.DataFrame:
    maximum = [
        _maximum_voltage_safe_current(
            row, config, level3, model, guard_v
        )
        for row in frame.itertuples()
    ]
    output = frame.copy()
    output["voltage_safe_current_max_a"] = maximum
    maximum_step = level3.constraint.maximum_current_step_a
    output["slew_lower_a"] = np.maximum(
        0.0, output.previous_current_a - maximum_step
    )
    output["slew_upper_a"] = np.minimum(
        10.0, output.previous_current_a + maximum_step
    )
    output["voltage_slew_feasibility_margin_a"] = (
        output.voltage_safe_current_max_a - output.slew_lower_a
    )
    output["voltage_slew_conflict"] = (
        output.voltage_slew_feasibility_margin_a
        < -config.audit.feasibility_tolerance_a
    )
    output["one_step_safe_current_a"] = np.minimum(
        output.slew_upper_a,
        np.maximum(
            output.slew_lower_a,
            np.minimum(output.current_a, output.voltage_safe_current_max_a),
        ),
    )
    output["one_step_current_correction_a"] = (
        output.one_step_safe_current_a - output.current_a
    )
    return output


def _threshold_timing(
    frame: pd.DataFrame, config: Phase7B1AConfig, level3: Any
) -> pd.DataFrame:
    records = []
    dt = level3.model.sample_period_s
    for key, group in frame.groupby(GROUP_KEYS, sort=False):
        ordered = group.sort_values("step_index")
        violation = ordered[
            ordered.terminal_voltage_v > config.audit.voltage_limit_v
        ]
        first_violation_step = (
            int(violation.step_index.iloc[0]) if not violation.empty else None
        )
        for threshold in config.audit.thresholds_v:
            entered = ordered[ordered.terminal_voltage_v >= threshold]
            if entered.empty:
                records.append(
                    {
                        "controller": key[0],
                        "seed": key[1],
                        "trajectory_id": key[2],
                        "threshold_v": threshold,
                        "entered": False,
                    }
                )
                continue
            row = entered.iloc[0]
            entry_step = int(row.step_index)
            required = int(
                np.ceil(
                    max(
                        float(row.current_a)
                        - max(float(row.voltage_safe_current_max_a), 0.0),
                        0.0,
                    )
                    / level3.constraint.maximum_current_step_a
                )
            )
            available = (
                first_violation_step - entry_step
                if first_violation_step is not None
                else -1
            )
            in_time = (
                first_violation_step is None
                or required <= max(available, 0)
            )
            records.append(
                {
                    "controller": key[0],
                    "seed": key[1],
                    "trajectory_id": key[2],
                    "threshold_v": threshold,
                    "entered": True,
                    "first_entry_step": entry_step,
                    "first_entry_time_s": (entry_step + 1) * dt,
                    "first_entry_soc": float(row.next_soc),
                    "first_entry_current_a": float(row.current_a),
                    "voltage_safe_current_max_a": float(
                        row.voltage_safe_current_max_a
                    ),
                    "braking_steps_required": required,
                    "steps_until_first_4p20_violation": available,
                    "maximum_slew_braking_in_time": in_time,
                    "voltage_slew_conflict_at_entry": bool(
                        row.voltage_slew_conflict
                    ),
                }
            )
    return pd.DataFrame(records)


def _relationship_table(frame: pd.DataFrame) -> pd.DataFrame:
    specifications = {
        "soc": (frame.soc, [0.40, 0.50, 0.60, 0.70, 0.75, 0.80]),
        "current_a": (frame.current_a, [0.0, 2.0, 5.0, 8.0, 10.000001]),
        "previous_current_a": (
            frame.previous_current_a,
            [0.0, 2.0, 5.0, 8.0, 10.000001],
        ),
        "predicted_voltage_v": (
            frame.predicted_next_voltage_2rc_v,
            [3.7, 4.0, 4.10, 4.15, 4.18, 4.20, 4.25],
        ),
        "predicted_margin_v": (
            frame.predicted_voltage_margin_v,
            [-0.05, 0.0, 0.01, 0.02, 0.05, 0.10, 0.50],
        ),
    }
    records = []
    for variable, (values, bins) in specifications.items():
        labels = pd.cut(values, bins, include_lowest=True, duplicates="drop")
        for label, group in frame.groupby(labels, observed=True):
            records.append(
                {
                    "variable": variable,
                    "bin": str(label),
                    "sample_count": len(group),
                    "mean_voltage_residual_v": float(
                        group.voltage_residual_v.mean()
                    ),
                    "p95_voltage_residual_v": float(
                        group.voltage_residual_v.quantile(0.95)
                    ),
                    "maximum_voltage_residual_v": float(
                        group.voltage_residual_v.max()
                    ),
                    "maximum_positive_growth_v": float(
                        group.residual_growth_v.clip(lower=0.0).max()
                    ),
                }
            )
    for label, group in frame.groupby("charge_stage", observed=True):
        records.append(
            {
                "variable": "charge_stage",
                "bin": str(label),
                "sample_count": len(group),
                "mean_voltage_residual_v": float(
                    group.voltage_residual_v.mean()
                ),
                "p95_voltage_residual_v": float(
                    group.voltage_residual_v.quantile(0.95)
                ),
                "maximum_voltage_residual_v": float(
                    group.voltage_residual_v.max()
                ),
                "maximum_positive_growth_v": float(
                    group.residual_growth_v.clip(lower=0.0).max()
                ),
            }
        )
    return pd.DataFrame(records)


def _design_confirmation_states(
    config: Phase7B1AConfig,
    level3: Any,
    model: Level3Model,
    regression: pd.DataFrame,
) -> pd.DataFrame:
    bounds = config.confirmation
    records = []
    candidate = bounds.design_start_index
    voltage_limit = (
        model.inherited.mpc.terminal_voltage_max_v
        - bounds.initial_voltage_margin_v
    )
    while len(records) < bounds.trajectory_count:
        soc = bounds.soc_bounds[0] + np.ptp(bounds.soc_bounds) * _van_der_corput(
            candidate, 2
        )
        v1 = bounds.v1_bounds_v[0] + np.ptp(bounds.v1_bounds_v) * _van_der_corput(
            candidate, 3
        )
        v2 = bounds.v2_bounds_v[0] + np.ptp(bounds.v2_bounds_v) * _van_der_corput(
            candidate, 5
        )
        previous = bounds.previous_current_bounds_a[0] + np.ptp(
            bounds.previous_current_bounds_a
        ) * _van_der_corput(candidate, 7)
        candidate += 1
        state = Level3State(soc, v1, v2, previous)
        minimum_current = max(
            0.0, previous - level3.constraint.maximum_current_step_a
        )
        if model.terminal_voltage(state, minimum_current) > voltage_limit:
            continue
        feasibility = Level3MPC(model).solve(state)
        if not (
            feasibility.optimizer_success and feasibility.prediction_feasible
        ):
            continue
        records.append(
            {
                "trajectory_id": f"phase7b1_confirm_{len(records):03d}",
                "initial_soc": soc,
                "initial_polarization_1_v": v1,
                "initial_polarization_2_v": v2,
                "initial_previous_current_a": previous,
                "design_candidate_index": candidate - 1,
                "design_seed": bounds.design_seed,
            }
        )
    confirmation = pd.DataFrame(records)
    regression_values = regression[
        [
            "initial_soc",
            "initial_polarization_1_v",
            "initial_polarization_2_v",
            "initial_previous_current_a",
        ]
    ].to_numpy(float)
    confirmation_values = confirmation[
        [
            "initial_soc",
            "initial_polarization_1_v",
            "initial_polarization_2_v",
            "initial_previous_current_a",
        ]
    ].to_numpy(float)
    if any(
        np.any(np.all(np.isclose(row, regression_values, atol=1.0e-14), axis=1))
        for row in confirmation_values
    ):
        raise RuntimeError("独立确认集与 12 条回归初态发生重合。")
    return confirmation


def _plot(output: Path, frame: pd.DataFrame, guard_v: float) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    axes[0].hist(1000.0 * frame.voltage_residual_v, bins=50, color="#2878B5")
    axes[0].axvline(
        1000.0 * frame.voltage_residual_v.quantile(0.99),
        color="orange",
        linestyle="--",
        label="P99",
    )
    axes[0].set(
        xlabel="DFN − 2RC voltage residual [mV]",
        ylabel="Samples",
        title="Frozen 72-trajectory residual",
    )
    axes[0].legend()
    sample = frame.iloc[:: max(len(frame) // 4000, 1)]
    axes[1].scatter(
        sample.predicted_next_voltage_2rc_v,
        1000.0 * sample.voltage_residual_v,
        c=sample.soc,
        s=5,
        alpha=0.35,
        cmap="viridis",
    )
    axes[1].set(
        xlabel="2RC next-step voltage [V]",
        ylabel="Residual [mV]",
        title="Residual structure",
    )
    axes[2].scatter(
        frame.terminal_voltage_v,
        frame.voltage_slew_feasibility_margin_a,
        s=4,
        alpha=0.25,
    )
    axes[2].axhline(0.0, color="red", linestyle="--")
    axes[2].axvline(4.2, color="red", linestyle=":")
    axes[2].set(
        xlabel="Measured DFN voltage [V]",
        ylabel="Voltage cap − slew lower bound [A]",
        title=f"One-step feasibility, guard={1000*guard_v:.2f} mV",
    )
    figure.savefig(output / "phase7b1a_voltage_mismatch_audit.png", dpi=180)
    plt.close(figure)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    residual = payload["voltage_residual"]
    braking = payload["braking_feasibility"]
    decision = payload["decision"]
    path.write_text(
        rf"""# Phase 7B-1A：25 ℃ DFN 电压失配与制动可行性审计

## 问题与符号

充电电流取正，电压单位为 V，电流单位为 A，采样周期为 5 s。冻结轨迹中每个控制步的电压残差定义为：

\[
e_{{V,k+1}}
=
V_{{\mathrm{{DFN}},k+1}}
-
\hat V_{{\mathrm{{2RC}},k+1}}(I_k).
\]

安全层在决策时刻只使用上一时刻已经测得的残差，并加入冻结回归轨迹中的最大正向一步增长：

\[
\Delta V_{{\mathrm{{guard}}}}
=
\max_k\max(e_{{V,k+1}}-e_{{V,k}},0).
\]

## 冻结合同

- Phase 7B-0 工件哈希：{payload['frozen_contract']['matched_count']}/{payload['frozen_contract']['artifact_count']} 匹配。
- 审计样本：{payload['frozen_contract']['sample_count']} 个控制步，72 条闭环。
- 独立确认集：{payload['confirmation_set']['trajectory_count']} 个新初态；它们不是 ANN 教师数据，且与 12 个回归初态无重合。

## 电压残差

- 最大正向残差：{1000*residual['maximum_positive_residual_v']:.3f} mV。
- 正向残差 P95 / P99：{1000*residual['positive_residual_p95_v']:.3f} / {1000*residual['positive_residual_p99_v']:.3f} mV。
- 一步正向增长最大值：{1000*residual['maximum_positive_growth_v']:.3f} mV。
- 一步正向增长 P95 / P99：{1000*residual['positive_growth_p95_v']:.3f} / {1000*residual['positive_growth_p99_v']:.3f} mV。
- 固定裕量诊断基线建议值（P99残差＋最大增长）：{1000*residual['diagnostic_fixed_margin_v']:.3f} mV。

## 电压—斜率可行性

在每一步求满足修正后下一步电压不超过 4.2 V 的最大电流：

\[
I_{{V,k}}^{{\max}}
=
\max\left\{{
I\in[0,10]:
\hat V_{{k+1}}^{{\mathrm{{2RC}}}}(I)
+e_{{V,k}}
+\Delta V_{{\mathrm{{guard}}}}
\leq 4.2
\right\}}.
\]

并与斜率下界

\[
I_k^-=\max(0,I_{{k-1}}-2)
\]

比较。结果：

- 电压—斜率空区间：{braking['voltage_slew_conflict_count']} 次。
- 最小可行性裕量：{braking['minimum_voltage_slew_margin_a']:.6f} A。
- 4.15、4.18、4.19 V 三个提前阈值中“最大斜率制动仍来不及”：{braking['late_braking_proactive_threshold_count']} 次。
- 已经进入 4.20 V 后才检测到的事后时刻：{braking['late_at_voltage_limit_count']} 次；该项只说明 4.20 V 不能作为提前触发阈值，不用于否决一步残差修正。
- 4.15 / 4.18 / 4.19 / 4.20 V 的首次进入时刻已逐轨迹保存在 `threshold_timing.csv`。

## 判定

**{decision['conclusion']}**

```json
{json.dumps(decision['checks'], ensure_ascii=False, indent=2)}
```

本阶段只决定一步残差修正是否具有斜率可行性，不宣称安全层已经通过 DFN 闭环。独立确认集在安全层参数冻结后才可用于最终验收。
""",
        encoding="utf-8",
    )


def _json_default(value: object) -> object:
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    raise TypeError(type(value).__name__)


def run_phase7b1a(
    config: Phase7B1AConfig, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    verification = _verify_frozen(config, root)
    level3, inherited, model = _load_model(config, root)
    trajectories = pd.read_csv(root / config.source_trajectories)
    initial = pd.read_csv(root / config.source_regression_initial_states)
    data_dir = root / config.output.data_directory
    output = root / config.output.result_directory
    data_dir.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    residuals = _augment_residuals(
        config, level3, model, trajectories, initial
    )
    positive = residuals.voltage_residual_v.clip(lower=0.0)
    growth = residuals.residual_growth_v.dropna().clip(lower=0.0)
    guard_v = float(growth.max())
    residuals = _add_feasibility(
        residuals, config, level3, model, guard_v
    )
    timing = _threshold_timing(residuals, config, level3)
    relationships = _relationship_table(residuals)
    confirmation = _design_confirmation_states(
        config, level3, model, initial
    )

    residuals.to_csv(data_dir / "voltage_residual_step_audit.csv", index=False)
    timing.to_csv(data_dir / "threshold_timing.csv", index=False)
    relationships.to_csv(
        data_dir / "voltage_residual_relationships.csv", index=False
    )
    confirmation_path = data_dir / "confirmation_initial_states.csv"
    confirmation.to_csv(confirmation_path, index=False)
    confirmation_hash = _sha256(confirmation_path)
    (data_dir / "confirmation_freeze.json").write_text(
        json.dumps(
            {
                "path": str(
                    confirmation_path.relative_to(root)
                ).replace("\\", "/"),
                "sha256": confirmation_hash,
                "trajectory_count": len(confirmation),
                "design_seed": config.confirmation.design_seed,
                "created_before_phase7b1b": True,
                "not_teacher_data": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _plot(output, residuals, guard_v)

    correlations = {}
    for name in (
        "soc",
        "current_a",
        "previous_current_a",
        "predicted_next_voltage_2rc_v",
        "predicted_voltage_margin_v",
    ):
        coefficient, pvalue = spearmanr(
            residuals[name], residuals.voltage_residual_v
        )
        correlations[name] = {
            "spearman_r": float(coefficient),
            "pvalue": float(pvalue),
        }
    entered = timing[timing.entered.astype(bool)]
    conflicts = residuals[residuals.voltage_slew_conflict.astype(bool)]
    late = entered[~entered.maximum_slew_braking_in_time.astype(bool)]
    proactive = entered[
        entered.threshold_v < config.audit.voltage_limit_v - 1.0e-12
    ]
    proactive_late = proactive[
        ~proactive.maximum_slew_braking_in_time.astype(bool)
    ]
    at_limit_late = late[
        late.threshold_v >= config.audit.voltage_limit_v - 1.0e-12
    ]
    residual_summary = {
        "maximum_positive_residual_v": float(positive.max()),
        "positive_residual_p95_v": float(positive.quantile(0.95)),
        "positive_residual_p99_v": float(positive.quantile(0.99)),
        "maximum_positive_growth_v": guard_v,
        "positive_growth_p95_v": float(growth.quantile(0.95)),
        "positive_growth_p99_v": float(growth.quantile(0.99)),
        "diagnostic_fixed_margin_v": float(
            positive.quantile(0.99) + guard_v
        ),
        "spearman_relationships": correlations,
    }
    braking = {
        "voltage_slew_conflict_count": len(conflicts),
        "voltage_slew_conflict_fraction": float(
            residuals.voltage_slew_conflict.mean()
        ),
        "minimum_voltage_slew_margin_a": float(
            residuals.voltage_slew_feasibility_margin_a.min()
        ),
        "late_braking_threshold_count": len(late),
        "late_braking_proactive_threshold_count": len(proactive_late),
        "late_at_voltage_limit_count": len(at_limit_late),
        "threshold_entry_count": len(entered),
        "maximum_required_braking_steps": int(
            entered.braking_steps_required.max()
        ),
    }
    checks = {
        "all_frozen_hashes_match": all(
            item["matched"] for item in verification.values()
        ),
        "confirmation_set_frozen_before_safety_layer": True,
        "confirmation_set_is_independent": True,
        "zero_voltage_slew_empty_intervals": len(conflicts) == 0,
        "maximum_slew_braking_is_never_late_before_voltage_limit": (
            len(proactive_late) == 0
        ),
    }
    success = bool(all(checks.values()))
    conclusion = (
        "7B-1A 通过：最大残差增长裕量下的一步电压限制始终与 2 A/步斜率约束相容，可进入 7B-1B 一步残差修正"
        if success
        else "7B-1A 判定一步限制来不及：停止一步方案，转入 2–5 步短时域制动修正"
    )
    payload = {
        "study_name": config.study_name,
        "configuration": asdict(config),
        "frozen_artifact_verification": verification,
        "frozen_contract": {
            "artifact_count": len(verification),
            "matched_count": sum(
                item["matched"] for item in verification.values()
            ),
            "sample_count": len(residuals),
            "trajectory_count": int(
                residuals.groupby(GROUP_KEYS).ngroups
            ),
        },
        "confirmation_set": {
            "trajectory_count": len(confirmation),
            "sha256": confirmation_hash,
            "independent_from_regression_initial_states": True,
            "not_teacher_data": True,
        },
        "voltage_residual": residual_summary,
        "braking_feasibility": braking,
        "decision": {
            "checks": checks,
            "phase7b1a_success": success,
            "proceed_to_one_step_phase7b1b": success,
            "proceed_to_short_horizon_instead": not success,
            "conclusion": conclusion,
        },
        "status": "completed",
        "success": success,
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    _write_report(output / "PHASE7B1A_中文实验报告.md", payload)
    print(json.dumps(payload["decision"], ensure_ascii=False), flush=True)
    return payload
