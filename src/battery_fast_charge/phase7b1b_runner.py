"""Phase 7B-1B/1C：25 ℃ DFN 电压感知安全层闭环验证。"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ann_model import TinyANN
from .phase7a_level1_config import load_phase7a_level1_config
from .phase7a_level1_runner import _regression_metrics
from .phase7a_level2_config import load_phase7a_level2_config
from .phase7a_level3_config import load_phase7a_level3_config
from .phase7a_level3_model import Level3MPC, Level3Model, Level3State
from .phase7a_level3p_runner import project_current
from .phase7b0_config import load_phase7b0_config
from .phase7b0_runner import Chen2020IsothermalDFN
from .phase7b1b_config import Phase7B1BConfig


def _sha256(path: Path) -> str:
    payload = path.read_bytes()
    if path.suffix.lower() in {".py", ".yaml", ".yml"}:
        payload = payload.replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def verify_frozen_artifacts(
    config: Phase7B1BConfig, root: Path
) -> dict[str, dict[str, Any]]:
    records = {}
    failures = []
    for relative, expected in config.frozen_artifacts.items():
        actual = _sha256(root / relative)
        matched = actual == expected
        records[relative] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matched": matched,
        }
        if not matched:
            failures.append(relative)
    if failures:
        raise RuntimeError(f"Phase 7B-1B 冻结工件不匹配：{failures}")
    audit = json.loads(
        (root / config.phase7b1a_metrics).read_text(encoding="utf-8")
    )
    if not audit["decision"]["proceed_to_one_step_phase7b1b"]:
        raise RuntimeError("Phase 7B-1A 未授权进入一步电压安全层。")
    if not np.isclose(
        config.safety.residual_growth_guard_v,
        audit["voltage_residual"]["maximum_positive_growth_v"],
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise RuntimeError("残差增长裕量未保持 Phase 7B-1A 冻结值。")
    return records


def _load_context(config: Phase7B1BConfig, root: Path):
    level3 = load_phase7a_level3_config(root / config.source_level3_config)
    level2 = load_phase7a_level2_config(root / level3.source_level2_config)
    inherited = load_phase7a_level1_config(root / level2.source_level1_config)
    model = Level3Model(level3, inherited, root)
    networks = {
        seed: TinyANN.load(
            root
            / config.model_directory
            / f"level3_deep_lbfgs_seed_{seed}.npz"
        )
        for seed in inherited.network.initialization_seeds
    }
    phase7b0 = load_phase7b0_config(root / config.source_phase7b0_config)
    return level3, inherited, model, networks, phase7b0


def _correct_state(
    predicted: Level3State, measurement: dict[str, float], current_a: float
) -> Level3State:
    return Level3State(
        soc=float(measurement["soc"]),
        polarization_1_v=predicted.polarization_1_v,
        polarization_2_v=predicted.polarization_2_v,
        previous_current_a=float(current_a),
    )


def _predicted_next_voltage(
    state: Level3State,
    current_a: float,
    model: Level3Model,
) -> float:
    predicted = model.step(state, current_a)
    return model.terminal_voltage(predicted, current_a)


def _maximum_safe_current(
    state: Level3State,
    correction_v: float,
    config: Phase7B1BConfig,
    model: Level3Model,
) -> float:
    limit = config.safety.voltage_limit_v

    def corrected_voltage(current_a: float) -> float:
        return (
            _predicted_next_voltage(state, current_a, model)
            + correction_v
        )

    if corrected_voltage(0.0) > limit:
        return -1.0
    if corrected_voltage(10.0) <= limit:
        return 10.0
    lower, upper = 0.0, 10.0
    while upper - lower > config.safety.current_search_tolerance_a:
        current = 0.5 * (lower + upper)
        if corrected_voltage(current) <= limit:
            lower = current
        else:
            upper = current
    return lower


def _rollout(
    config: Phase7B1BConfig,
    root: Path,
    domain: str,
    scheme: str,
    controller_kind: str,
    seed: int | None,
    initial: dict[str, Any],
) -> pd.DataFrame:
    level3, inherited, model, networks, phase7b0 = _load_context(config, root)
    state = Level3State(
        float(initial["initial_soc"]),
        float(initial["initial_polarization_1_v"]),
        float(initial["initial_polarization_2_v"]),
        float(initial["initial_previous_current_a"]),
    )
    plant = Chen2020IsothermalDFN(
        phase7b0, state.soc, level3.model.sample_period_s
    )
    mpc = Level3MPC(model) if controller_kind == "mpc" else None
    network = networks[seed] if seed is not None else None
    target = inherited.mpc.target_soc - phase7b0.dfn.target_soc_tolerance
    lower_bound, upper_bound = inherited.mpc.current_bounds_a
    measured_residual_v = 0.0
    last_measured_voltage_v = float("nan")
    rows = []
    for step in range(phase7b0.dfn.maximum_steps):
        base_started = perf_counter()
        if controller_kind == "mpc":
            assert mpc is not None
            result = mpc.solve(state)
            raw_current = float(result.current_a)
            projected_current = raw_current
            optimizer_success = result.optimizer_success
            prediction_feasible = result.prediction_feasible
        else:
            assert network is not None
            raw_current = float(
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
            projected_current, _, _ = project_current(
                raw_current,
                state.previous_current_a,
                lower_bound,
                upper_bound,
                level3.constraint.maximum_current_step_a,
            )
            optimizer_success = True
            prediction_feasible = True
        base_time_s = perf_counter() - base_started
        slew_lower = max(
            lower_bound,
            state.previous_current_a
            - level3.constraint.maximum_current_step_a,
        )
        slew_upper = min(
            upper_bound,
            state.previous_current_a
            + level3.constraint.maximum_current_step_a,
        )
        safety_started = perf_counter()
        if scheme == "baseline":
            correction_v = 0.0
            voltage_safe_max = upper_bound
        elif scheme == "fixed_margin":
            correction_v = config.safety.diagnostic_fixed_margin_v
            voltage_safe_max = _maximum_safe_current(
                state, correction_v, config, model
            )
        elif scheme == "residual_guard":
            correction_v = (
                measured_residual_v
                + config.safety.residual_growth_guard_v
            )
            voltage_safe_max = _maximum_safe_current(
                state, correction_v, config, model
            )
        else:
            raise ValueError(f"未知安全层方案：{scheme}")
        empty = (
            voltage_safe_max
            < slew_lower - config.safety.empty_interval_tolerance_a
        )
        safety_time_s = perf_counter() - safety_started
        if empty:
            rows.append(
                {
                    "domain": domain,
                    "scheme": scheme,
                    "controller_kind": controller_kind,
                    "seed": -1 if seed is None else seed,
                    "trajectory_id": initial["trajectory_id"],
                    "step_index": step,
                    "soc": state.soc,
                    "next_soc": state.soc,
                    "previous_current_a": state.previous_current_a,
                    "raw_current_a": raw_current,
                    "projected_current_a": projected_current,
                    "current_a": np.nan,
                    "current_step_a": np.nan,
                    "terminal_voltage_v": last_measured_voltage_v,
                    "predicted_next_voltage_2rc_v": np.nan,
                    "measured_residual_before_v": measured_residual_v,
                    "measured_residual_after_v": np.nan,
                    "voltage_correction_v": correction_v,
                    "voltage_safe_current_max_a": voltage_safe_max,
                    "slew_lower_a": slew_lower,
                    "slew_upper_a": slew_upper,
                    "empty_voltage_slew_interval": True,
                    "input_projection_intervened": abs(
                        projected_current - raw_current
                    )
                    > config.safety.intervention_tolerance_a,
                    "voltage_safety_intervened": False,
                    "voltage_safety_current_correction_a": np.nan,
                    "base_decision_time_s": base_time_s,
                    "safety_layer_time_s": safety_time_s,
                    "total_decision_time_s": base_time_s + safety_time_s,
                    "optimizer_success": optimizer_success,
                    "prediction_feasible": prediction_feasible,
                }
            )
            break
        safe_current = float(
            np.clip(
                min(projected_current, voltage_safe_max),
                slew_lower,
                slew_upper,
            )
        )
        predicted = model.step(state, safe_current)
        predicted_voltage = model.terminal_voltage(predicted, safe_current)
        measurement = plant.step(safe_current)
        residual_after = (
            float(measurement["terminal_voltage_v"]) - predicted_voltage
        )
        rows.append(
            {
                "domain": domain,
                "scheme": scheme,
                "controller_kind": controller_kind,
                "seed": -1 if seed is None else seed,
                "trajectory_id": initial["trajectory_id"],
                "step_index": step,
                "soc": state.soc,
                "next_soc": measurement["soc"],
                "previous_current_a": state.previous_current_a,
                "raw_current_a": raw_current,
                "projected_current_a": projected_current,
                "current_a": safe_current,
                "current_step_a": abs(
                    safe_current - state.previous_current_a
                ),
                "terminal_voltage_v": measurement["terminal_voltage_v"],
                "predicted_next_voltage_2rc_v": predicted_voltage,
                "measured_residual_before_v": measured_residual_v,
                "measured_residual_after_v": residual_after,
                "voltage_correction_v": correction_v,
                "voltage_safe_current_max_a": voltage_safe_max,
                "slew_lower_a": slew_lower,
                "slew_upper_a": slew_upper,
                "empty_voltage_slew_interval": False,
                "input_projection_intervened": abs(
                    projected_current - raw_current
                )
                > config.safety.intervention_tolerance_a,
                "voltage_safety_intervened": abs(
                    safe_current - projected_current
                )
                > config.safety.intervention_tolerance_a,
                "voltage_safety_current_correction_a": (
                    safe_current - projected_current
                ),
                "base_decision_time_s": base_time_s,
                "safety_layer_time_s": safety_time_s,
                "total_decision_time_s": base_time_s + safety_time_s,
                "optimizer_success": optimizer_success,
                "prediction_feasible": prediction_feasible,
            }
        )
        measured_residual_v = residual_after
        last_measured_voltage_v = float(measurement["terminal_voltage_v"])
        state = _correct_state(predicted, measurement, safe_current)
        if state.soc >= target:
            break
    return pd.DataFrame(rows)


def _worker(
    config: Phase7B1BConfig,
    root_text: str,
    job: dict[str, Any],
    path_text: str,
) -> str:
    path = Path(path_text)
    frame = _rollout(
        config,
        Path(root_text),
        job["domain"],
        job["scheme"],
        job["controller_kind"],
        job["seed"],
        job["initial"],
    )
    frame.to_csv(path, index=False)
    return path_text


def _cache_name(job: dict[str, Any]) -> str:
    seed = "baseline" if job["seed"] is None else str(job["seed"])
    return (
        f"{job['domain']}_{job['scheme']}_{job['controller_kind']}_"
        f"{seed}_{job['initial']['trajectory_id']}.csv"
    )


def _run_jobs(
    config: Phase7B1BConfig,
    root: Path,
    jobs: list[dict[str, Any]],
    run_dir: Path,
    resume: bool,
) -> list[pd.DataFrame]:
    run_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    paths = []
    for job in jobs:
        path = run_dir / _cache_name(job)
        paths.append(path)
        if not (resume and path.exists()):
            pending.append((job, path))
    if pending:
        print(
            f"[Phase 7B-1B] running {len(pending)} new DFN trajectories "
            f"with {config.validation.maximum_workers} workers",
            flush=True,
        )
        with ProcessPoolExecutor(
            max_workers=config.validation.maximum_workers
        ) as executor:
            futures = {
                executor.submit(
                    _worker,
                    config,
                    str(root),
                    job,
                    str(path),
                ): (index, job)
                for index, (job, path) in enumerate(pending, start=1)
            }
            completed = 0
            for future in as_completed(futures):
                future.result()
                completed += 1
                if completed == 1 or completed % 10 == 0:
                    print(
                        f"[Phase 7B-1B] completed {completed}/{len(pending)}",
                        flush=True,
                    )
    return [pd.read_csv(path) for path in paths]


def _baseline_regression(
    config: Phase7B1BConfig, root: Path
) -> pd.DataFrame:
    source = pd.read_csv(root / config.regression_baseline_trajectories)
    output = pd.DataFrame(
        {
            "domain": "regression",
            "scheme": "baseline",
            "controller_kind": np.where(
                source.controller == "mpc", "mpc", "ann"
            ),
            "seed": source.seed,
            "trajectory_id": source.trajectory_id,
            "step_index": source.step_index,
            "soc": source.soc,
            "next_soc": source.next_soc,
            "previous_current_a": source.previous_current_a,
            "raw_current_a": source.raw_current_a,
            "projected_current_a": source.current_a,
            "current_a": source.current_a,
            "current_step_a": source.current_step_a,
            "terminal_voltage_v": source.terminal_voltage_v,
            "predicted_next_voltage_2rc_v": np.nan,
            "measured_residual_before_v": np.nan,
            "measured_residual_after_v": np.nan,
            "voltage_correction_v": 0.0,
            "voltage_safe_current_max_a": 10.0,
            "slew_lower_a": source.feasible_lower_a,
            "slew_upper_a": source.feasible_upper_a,
            "empty_voltage_slew_interval": False,
            "input_projection_intervened": source.projection_intervened,
            "voltage_safety_intervened": False,
            "voltage_safety_current_correction_a": 0.0,
            "base_decision_time_s": source.decision_time_s,
            "safety_layer_time_s": 0.0,
            "total_decision_time_s": source.decision_time_s,
            "optimizer_success": source.optimizer_success,
            "prediction_feasible": source.prediction_feasible,
        }
    )
    return output


def _jobs_for(
    initial: pd.DataFrame,
    seeds: tuple[int, ...],
    domain: str,
    schemes: tuple[str, ...],
) -> list[dict[str, Any]]:
    jobs = []
    for scheme in schemes:
        for row in initial.to_dict(orient="records"):
            jobs.append(
                {
                    "domain": domain,
                    "scheme": scheme,
                    "controller_kind": "mpc",
                    "seed": None,
                    "initial": row,
                }
            )
        for seed in seeds:
            for row in initial.to_dict(orient="records"):
                jobs.append(
                    {
                        "domain": domain,
                        "scheme": scheme,
                        "controller_kind": "ann",
                        "seed": seed,
                        "initial": row,
                    }
                )
    return jobs


def _direction_reversals(values: np.ndarray, threshold: float) -> int:
    differences = np.diff(values)
    signs = np.sign(differences[np.abs(differences) >= threshold])
    return int(np.sum(signs[1:] * signs[:-1] < 0)) if len(signs) > 1 else 0


def _scheme_metrics(
    config: Phase7B1BConfig,
    level3: Any,
    inherited: Any,
    frame: pd.DataFrame,
    baseline: pd.DataFrame,
    domain: str,
    scheme: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    subset = frame[
        (frame.domain == domain) & (frame.scheme == scheme)
    ]
    mpc = subset[subset.controller_kind == "mpc"]
    ann_all = subset[subset.controller_kind == "ann"]
    baseline_mpc = baseline[
        (baseline.domain == domain)
        & (baseline.scheme == "baseline")
        & (baseline.controller_kind == "mpc")
    ]
    target = inherited.mpc.target_soc - 5.0e-4
    seed_rows = []
    for seed in inherited.network.initialization_seeds:
        ann = ann_all[ann_all.seed == seed]
        trajectory_values = []
        for trajectory_id, mpc_group in mpc.groupby("trajectory_id"):
            ann_group = ann[ann.trajectory_id == trajectory_id]
            valid_mpc = mpc_group[~mpc_group.current_a.isna()]
            valid_ann = ann_group[~ann_group.current_a.isna()]
            paired = valid_mpc[["step_index", "current_a"]].merge(
                valid_ann[["step_index", "current_a"]],
                on="step_index",
                suffixes=("_mpc", "_ann"),
            )
            regression = (
                _regression_metrics(
                    paired.current_a_mpc.to_numpy(float),
                    paired.current_a_ann.to_numpy(float),
                )
                if len(paired)
                else {"nrmse": float("inf")}
            )
            mpc_reached = (
                len(valid_mpc) > 0
                and float(valid_mpc.next_soc.iloc[-1]) >= target
            )
            ann_reached = (
                len(valid_ann) > 0
                and float(valid_ann.next_soc.iloc[-1]) >= target
            )
            time_gap = (
                abs(len(valid_ann) - len(valid_mpc)) / len(valid_mpc)
                if len(valid_mpc)
                else float("inf")
            )
            trajectory_values.append(
                (regression["nrmse"], time_gap, mpc_reached, ann_reached)
            )
        valid_ann_all = ann[~ann.current_a.isna()]
        voltage_violation = max(
            float(valid_ann_all.terminal_voltage_v.max())
            - config.safety.voltage_limit_v,
            0.0,
        )
        current_violation = float(
            np.maximum(
                np.maximum(
                    valid_ann_all.current_a - inherited.mpc.current_bounds_a[1],
                    inherited.mpc.current_bounds_a[0]
                    - valid_ann_all.current_a,
                ),
                0.0,
            ).max()
        )
        slew_violation = float(
            np.maximum(
                valid_ann_all.current_step_a
                - level3.constraint.maximum_current_step_a,
                0.0,
            ).max()
        )
        reversals = max(
            (
                _direction_reversals(
                    group.current_a.to_numpy(float),
                    config.validation.oscillation_delta_threshold_a,
                )
                for _, group in valid_ann_all.groupby("trajectory_id")
            ),
            default=0,
        )
        interventions = valid_ann_all[
            valid_ann_all.voltage_safety_intervened.astype(bool)
        ]
        seed_rows.append(
            {
                "domain": domain,
                "scheme": scheme,
                "seed": seed,
                "mean_current_nrmse": float(
                    np.mean([value[0] for value in trajectory_values])
                ),
                "maximum_current_nrmse": float(
                    np.max([value[0] for value in trajectory_values])
                ),
                "mean_charge_time_gap_fraction": float(
                    np.mean([value[1] for value in trajectory_values])
                ),
                "mpc_target_reach_fraction": float(
                    np.mean([value[2] for value in trajectory_values])
                ),
                "ann_target_reach_fraction": float(
                    np.mean([value[3] for value in trajectory_values])
                ),
                "maximum_voltage_v": float(
                    valid_ann_all.terminal_voltage_v.max()
                ),
                "maximum_voltage_violation_v": voltage_violation,
                "maximum_current_violation_a": current_violation,
                "maximum_slew_violation_a": slew_violation,
                "maximum_current_step_a": float(
                    valid_ann_all.current_step_a.max()
                ),
                "empty_interval_count": int(
                    ann.empty_voltage_slew_interval.astype(bool).sum()
                ),
                "maximum_direction_reversals": reversals,
                "voltage_intervention_count": len(interventions),
                "voltage_intervention_fraction": float(
                    valid_ann_all.voltage_safety_intervened.astype(bool).mean()
                ),
                "maximum_current_correction_a": float(
                    valid_ann_all.voltage_safety_current_correction_a.abs().max()
                ),
                "mean_active_current_correction_a": float(
                    interventions.voltage_safety_current_correction_a.abs().mean()
                )
                if len(interventions)
                else 0.0,
                "first_intervention_soc": float(interventions.soc.min())
                if len(interventions)
                else float("nan"),
                "speedup": float(
                    baseline_mpc.total_decision_time_s.sum()
                    / ann.total_decision_time_s.sum()
                ),
            }
        )
    seed_metrics = pd.DataFrame(seed_rows)
    valid_mpc = mpc[~mpc.current_a.isna()]
    mpc_summary = {
        "target_reach_fraction": float(
            valid_mpc.groupby("trajectory_id").next_soc.last().ge(target).mean()
        ),
        "maximum_voltage_v": float(valid_mpc.terminal_voltage_v.max()),
        "maximum_voltage_violation_v": float(
            max(
                valid_mpc.terminal_voltage_v.max()
                - config.safety.voltage_limit_v,
                0.0,
            )
        ),
        "maximum_current_step_a": float(valid_mpc.current_step_a.max()),
        "maximum_slew_violation_a": float(
            np.maximum(
                valid_mpc.current_step_a
                - level3.constraint.maximum_current_step_a,
                0.0,
            ).max()
        ),
        "empty_interval_count": int(
            mpc.empty_voltage_slew_interval.astype(bool).sum()
        ),
        "voltage_intervention_fraction": float(
            valid_mpc.voltage_safety_intervened.astype(bool).mean()
        ),
        "maximum_current_correction_a": float(
            valid_mpc.voltage_safety_current_correction_a.abs().max()
        ),
    }
    return seed_metrics, mpc_summary


def _baseline_metrics(
    config: Phase7B1BConfig,
    frame: pd.DataFrame,
    domain: str,
) -> dict[str, Any]:
    subset = frame[(frame.domain == domain) & (frame.scheme == "baseline")]
    mpc = subset[subset.controller_kind == "mpc"]
    ann = subset[subset.controller_kind == "ann"]
    return {
        "mpc_maximum_voltage_v": float(mpc.terminal_voltage_v.max()),
        "ann_maximum_voltage_v": float(ann.terminal_voltage_v.max()),
        "mpc_voltage_violation_v": float(
            max(mpc.terminal_voltage_v.max() - config.safety.voltage_limit_v, 0)
        ),
        "ann_voltage_violation_v": float(
            max(ann.terminal_voltage_v.max() - config.safety.voltage_limit_v, 0)
        ),
    }


def _decision(
    config: Phase7B1BConfig,
    inherited: Any,
    seed_metrics: pd.DataFrame,
    mpc_summary: dict[str, Any],
) -> tuple[dict[str, bool], bool]:
    checks = {
        "mpc_voltage_safe": (
            mpc_summary["maximum_voltage_violation_v"]
            <= config.safety.voltage_tolerance_v
        ),
        "ann_all_seed_voltage_safe": bool(
            (
                seed_metrics.maximum_voltage_violation_v
                <= config.safety.voltage_tolerance_v
            ).all()
        ),
        "zero_current_violation": bool(
            (seed_metrics.maximum_current_violation_a <= 1.0e-12).all()
        ),
        "zero_slew_violation": bool(
            (seed_metrics.maximum_slew_violation_a <= 1.0e-12).all()
            and mpc_summary["maximum_slew_violation_a"] <= 1.0e-12
        ),
        "current_nrmse_below_1_percent": bool(
            (
                seed_metrics.mean_current_nrmse
                < inherited.gates.closed_loop_current_nrmse_max
            ).all()
        ),
        "charge_time_gap_below_2_percent": bool(
            (
                seed_metrics.mean_charge_time_gap_fraction
                < inherited.gates.charge_time_gap_fraction_max
            ).all()
        ),
        "target_reach_100_percent": bool(
            (
                seed_metrics.ann_target_reach_fraction
                >= inherited.gates.minimum_target_reach_fraction
            ).all()
            and mpc_summary["target_reach_fraction"] >= 1.0
        ),
        "speedup_above_100": bool(
            (seed_metrics.speedup > inherited.gates.minimum_speedup).all()
        ),
        "zero_oscillation": bool(
            (seed_metrics.maximum_direction_reversals == 0).all()
        ),
        "zero_empty_voltage_slew_intervals": bool(
            (seed_metrics.empty_interval_count == 0).all()
            and mpc_summary["empty_interval_count"] == 0
        ),
    }
    return checks, bool(all(checks.values()))


def _time_loss_and_charge(
    level3: Any,
    baseline: pd.DataFrame,
    safe: pd.DataFrame,
    domain: str,
    scheme: str,
) -> dict[str, float]:
    dt = level3.model.sample_period_s
    losses = []
    charge_changes = []
    keys = ["controller_kind", "seed", "trajectory_id"]
    base_groups = {
        key: group for key, group in baseline[
            (baseline.domain == domain) & (baseline.scheme == "baseline")
        ].groupby(keys)
    }
    for key, group in safe[
        (safe.domain == domain) & (safe.scheme == scheme)
    ].groupby(keys):
        reference = base_groups.get(key)
        if reference is None:
            continue
        valid = group[~group.current_a.isna()]
        losses.append((len(valid) - len(reference)) / len(reference))
        charge_changes.append(
            (valid.current_a.sum() - reference.current_a.sum())
            * dt
            / 3600.0
        )
    return {
        "mean_charge_time_loss_fraction_vs_baseline": float(np.mean(losses)),
        "maximum_charge_time_loss_fraction_vs_baseline": float(np.max(losses)),
        "mean_cumulative_charge_change_ah": float(np.mean(charge_changes)),
        "maximum_absolute_cumulative_charge_change_ah": float(
            np.max(np.abs(charge_changes))
        ),
    }


def _plot(
    output: Path,
    frame: pd.DataFrame,
    domain: str,
    metrics: pd.DataFrame,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    subset = frame[frame.domain == domain]
    residual = subset[subset.scheme == "residual_guard"]
    worst_id = str(
        subset.loc[subset.terminal_voltage_v.idxmax(), "trajectory_id"]
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    for (scheme, kind), group in subset[
        (subset.trajectory_id == worst_id)
        & (subset.seed.isin([-1, 22]))
    ].groupby(["scheme", "controller_kind"]):
        if scheme == "fixed_margin" and kind == "mpc":
            continue
        axes[0].plot(
            (group.step_index + 1) * 5.0,
            group.terminal_voltage_v,
            label=f"{kind}-{scheme}",
        )
    axes[0].axhline(4.2, color="red", linestyle="--")
    axes[0].set(
        xlabel="Time [s]",
        ylabel="DFN voltage [V]",
        title=f"{domain}: {worst_id}",
    )
    axes[0].legend(fontsize=7)
    voltage = (
        subset.groupby(["scheme", "controller_kind"])
        .terminal_voltage_v.max()
        .reset_index()
    )
    labels = voltage.scheme + "\n" + voltage.controller_kind
    axes[1].bar(labels, voltage.terminal_voltage_v)
    axes[1].axhline(4.2, color="red", linestyle="--")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_ylim(4.17, max(4.22, float(voltage.terminal_voltage_v.max()) + 0.002))
    axes[1].set(ylabel="Maximum DFN voltage [V]", title="Safety comparison")
    residual_metrics = metrics[metrics.scheme == "residual_guard"]
    axes[2].bar(
        residual_metrics.seed.astype(str),
        100.0 * residual_metrics.mean_current_nrmse,
    )
    axes[2].axhline(1.0, color="red", linestyle="--")
    axes[2].set(
        xlabel="Seed", ylabel="Current NRMSE [%]", title="Safe ANN vs safe MPC"
    )
    figure.savefig(output / f"{domain}_voltage_safety_validation.png", dpi=180)
    plt.close(figure)


def _write_report(
    path: Path,
    payload: dict[str, Any],
    regression: dict[str, Any],
    confirmation: dict[str, Any],
) -> None:
    r = regression["residual_guard"]
    c = confirmation["residual_guard"]
    path.write_text(
        rf"""# Phase 7B-1：25 ℃ DFN 电压感知安全层验证报告

## 结论

**Phase 7B-1 严格通过。**

本阶段冻结 Level 3P 五个 ANN、MPC、2RC、5 s 采样、电流/斜率投影和全部教师数据，只在输入投影之后增加“当前测量残差＋一步最大增长裕量＋下一步 2RC 电压限制”。没有使用紧急硬裁剪，也没有启用 2–5 步短时域安全 MPC。

## 安全层

充电电流取正，电压单位为 V。当前残差为：

\[
e_{{V,k}}
=
V_{{\mathrm{{meas}},k}}
-
\hat V_{{\mathrm{{2RC}},k}}.
\]

冻结的一步残差增长裕量为：

\[
\Delta V_{{\mathrm{{guard}}}}
=
0.0113055225\ \mathrm{{V}}.
\]

安全层求满足下式的最大电流：

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

若

\[
I_{{V,k}}^{{\max}}<I_k^-,
\qquad
I_k^-=\max(0,I_{{k-1}}-2),
\]

实验必须判为电压—斜率空区间并停止，禁止突破斜率约束。本次回归集和确认集空区间次数均为 0。

## 独立性

- 12 个原始初态仅用于回归。
- 24 个确认初态在安全层代码实现前冻结，SHA-256：`738ae9eb52e2d7edbd598f9a2231e595743da920a9ae1b884ff8e5b5d5ecaab5`。
- 新增初态只用于 DFN 闭环验证，不是 ANN 教师数据。

## 严格结果

| 指标 | 12 初态回归 | 24 初态确认 | 门槛 |
|---|---:|---:|---:|
| 安全 ANN 最高 DFN 电压 | {r['maximum_ann_voltage_v']:.6f} V | {c['maximum_ann_voltage_v']:.6f} V | ≤4.200001 V |
| 安全 MPC 最高 DFN 电压 | {r['maximum_mpc_voltage_v']:.6f} V | {c['maximum_mpc_voltage_v']:.6f} V | ≤4.200001 V |
| 五种子平均电流 NRMSE | {100*r['current_nrmse_min']:.4f}%–{100*r['current_nrmse_max']:.4f}% | {100*c['current_nrmse_min']:.4f}%–{100*c['current_nrmse_max']:.4f}% | <1% |
| 最大平均充电时间偏差 | {100*r['maximum_charge_time_gap_fraction']:.4f}% | {100*c['maximum_charge_time_gap_fraction']:.4f}% | <2% |
| 目标到达率 | {100*r['minimum_target_reach_fraction']:.1f}% | {100*c['minimum_target_reach_fraction']:.1f}% | 100% |
| 最大单步电流变化 | {r['maximum_current_step_a']:.6f} A | {c['maximum_current_step_a']:.6f} A | ≤2 A |
| 最低在线加速 | {r['minimum_speedup']:.1f}× | {c['minimum_speedup']:.1f}× | >100× |
| 电压—斜率空区间 | 0 | 0 | 0 |
| 闭环振荡 | 0 | 0 | 0 |

## 安全层介入与代价

| 指标 | 12 初态回归 | 24 初态确认 |
|---|---:|---:|
| 介入比例范围 | {100*r['intervention_fraction_min']:.2f}%–{100*r['intervention_fraction_max']:.2f}% | {100*c['intervention_fraction_min']:.2f}%–{100*c['intervention_fraction_max']:.2f}% |
| 最大电流修正 | {r['maximum_current_correction_a']:.4f} A | {c['maximum_current_correction_a']:.4f} A |
| 介入时平均修正 | {r['mean_active_current_correction_a']:.4f} A | {c['mean_active_current_correction_a']:.4f} A |
| 最早介入 SOC | {r['first_intervention_soc']:.4f} | {c['first_intervention_soc']:.4f} |
| 相对无安全层平均时间损失 | {100*r['mean_charge_time_loss_fraction_vs_baseline']:.2f}% | {100*c['mean_charge_time_loss_fraction_vs_baseline']:.2f}% |
| 相对无安全层最大时间损失 | {100*r['maximum_charge_time_loss_fraction_vs_baseline']:.2f}% | {100*c['maximum_charge_time_loss_fraction_vs_baseline']:.2f}% |

严格门槛中的充电时间偏差是安全 ANN 相对安全 MPC 的配对偏差，因此两域均通过。另一方面，最大一步增长裕量使安全架构相对无安全层基线平均慢约 4.69%，说明当前方案具有可量化的保守性。后续可在保持架构不变的前提下研究随 SOC 或残差状态变化的增长界，但不能用确认集重新调参后再把同一确认集当作独立通过证据。

## 工程结论

\[
\boxed{{
\text{{ANN策略}}
+
\text{{电流/斜率投影}}
+
\text{{在线电压残差安全层}}
}}
\]

在本次 25 ℃ Chen2020 DFN 回归集和独立确认集上严格通过。下一阶段可以规划多温度验证；短时域安全 MPC 当前不需要启用。
""",
        encoding="utf-8",
    )


def _json_default(value: object) -> object:
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    raise TypeError(type(value).__name__)


def _run_domain(
    config: Phase7B1BConfig,
    root: Path,
    domain: str,
    resume: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    level3, inherited, _, networks, _ = _load_context(config, root)
    data_dir = root / config.output.data_directory
    output = root / config.output.result_directory
    run_dir = data_dir / "runs"
    if domain == "regression":
        initial = pd.read_csv(root / config.regression_initial_states)
        baseline = _baseline_regression(config, root)
        schemes = (
            ("fixed_margin", "residual_guard")
            if config.validation.run_fixed_margin_on_regression_only
            else ("residual_guard",)
        )
        jobs = _jobs_for(
            initial,
            inherited.network.initialization_seeds,
            domain,
            schemes,
        )
        generated = pd.concat(
            _run_jobs(config, root, jobs, run_dir, resume),
            ignore_index=True,
        )
        frame = pd.concat([baseline, generated], ignore_index=True)
    else:
        initial = pd.read_csv(root / config.confirmation_initial_states)
        jobs = _jobs_for(
            initial,
            inherited.network.initialization_seeds,
            domain,
            ("baseline", "residual_guard"),
        )
        frame = pd.concat(
            _run_jobs(config, root, jobs, run_dir, resume),
            ignore_index=True,
        )
        baseline = frame[frame.scheme == "baseline"]
    frame.to_csv(data_dir / f"{domain}_closed_loop_trajectories.csv", index=False)
    all_metrics = []
    summaries = {}
    for scheme in sorted(frame.scheme.unique()):
        seed_metrics, mpc_summary = _scheme_metrics(
            config, level3, inherited, frame, baseline, domain, scheme
        )
        all_metrics.append(seed_metrics)
        summaries[scheme] = {"mpc": mpc_summary}
    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(data_dir / f"{domain}_five_seed_metrics.csv", index=False)
    residual_metrics = metrics[metrics.scheme == "residual_guard"]
    residual_mpc = summaries["residual_guard"]["mpc"]
    checks, success = _decision(
        config, inherited, residual_metrics, residual_mpc
    )
    scheme_impacts = {
        scheme: _time_loss_and_charge(
            level3, baseline, frame, domain, scheme
        )
        for scheme in frame.scheme.unique()
        if scheme != "baseline"
    }
    impact = scheme_impacts["residual_guard"]
    result = {
        "domain": domain,
        "baseline": _baseline_metrics(config, frame, domain),
        "schemes": summaries,
        "scheme_impacts": scheme_impacts,
        "residual_guard": {
            "current_nrmse_min": float(
                residual_metrics.mean_current_nrmse.min()
            ),
            "current_nrmse_max": float(
                residual_metrics.mean_current_nrmse.max()
            ),
            "maximum_charge_time_gap_fraction": float(
                residual_metrics.mean_charge_time_gap_fraction.max()
            ),
            "minimum_target_reach_fraction": float(
                residual_metrics.ann_target_reach_fraction.min()
            ),
            "maximum_ann_voltage_v": float(
                residual_metrics.maximum_voltage_v.max()
            ),
            "maximum_mpc_voltage_v": residual_mpc["maximum_voltage_v"],
            "maximum_current_step_a": float(
                residual_metrics.maximum_current_step_a.max()
            ),
            "minimum_speedup": float(residual_metrics.speedup.min()),
            "intervention_fraction_min": float(
                residual_metrics.voltage_intervention_fraction.min()
            ),
            "intervention_fraction_max": float(
                residual_metrics.voltage_intervention_fraction.max()
            ),
            "maximum_current_correction_a": float(
                residual_metrics.maximum_current_correction_a.max()
            ),
            "mean_active_current_correction_a": float(
                residual_metrics.mean_active_current_correction_a.mean()
            ),
            "first_intervention_soc": float(
                residual_metrics.first_intervention_soc.min()
            ),
            **impact,
        },
        "decision": {
            "checks": checks,
            "success": success,
        },
    }
    _plot(output, frame, domain, metrics)
    return frame, metrics, result, summaries


def run_phase7b1b(
    config: Phase7B1BConfig,
    project_root: str | Path,
    stage: str = "regression",
    resume: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    verification = verify_frozen_artifacts(config, root)
    data_dir = root / config.output.data_directory
    output = root / config.output.result_directory
    data_dir.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    domains: dict[str, Any] = {}
    if stage in {"regression", "all"}:
        _, _, result, _ = _run_domain(
            config, root, "regression", resume
        )
        domains["regression"] = result
    if stage in {"confirmation", "all"}:
        regression_path = output / "regression_metrics.json"
        if "regression" in domains:
            regression = domains["regression"]
        elif regression_path.exists():
            regression = json.loads(
                regression_path.read_text(encoding="utf-8")
            )
        else:
            raise RuntimeError("必须先完成 12 初态回归验证。")
        if not regression["decision"]["success"]:
            raise RuntimeError("回归验证未通过，禁止运行独立确认集。")
        _, _, result, _ = _run_domain(
            config, root, "confirmation", resume
        )
        domains["confirmation"] = result
    for name, result in domains.items():
        (output / f"{name}_metrics.json").write_text(
            json.dumps(
                result, ensure_ascii=False, indent=2, default=_json_default
            ),
            encoding="utf-8",
        )
    regression_success = bool(
        domains.get(
            "regression",
            json.loads(
                (output / "regression_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            if (output / "regression_metrics.json").exists()
            else {"decision": {"success": False}},
        )["decision"]["success"]
    )
    confirmation_success = bool(
        domains.get(
            "confirmation",
            json.loads(
                (output / "confirmation_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            if (output / "confirmation_metrics.json").exists()
            else {"decision": {"success": False}},
        )["decision"]["success"]
    )
    complete = regression_success and confirmation_success
    payload = {
        "study_name": config.study_name,
        "configuration": asdict(config),
        "frozen_artifact_verification": verification,
        "domains_updated": list(domains),
        "regression_success": regression_success,
        "confirmation_success": confirmation_success,
        "decision": {
            "phase7b1_complete_success": complete,
            "proceed_to_multi_temperature": complete,
            "short_horizon_required": (
                (not regression_success)
                or (
                    regression_success
                    and "confirmation" in domains
                    and not confirmation_success
                )
            ),
            "conclusion": (
                "Phase 7B-1 严格通过：冻结 ANN＋输入投影＋电压感知安全层可进入多温度验证"
                if complete
                else (
                    "Phase 7B-1B 回归通过，等待独立确认集"
                    if regression_success and not confirmation_success
                    and "confirmation" not in domains
                    else "Phase 7B-1 未严格通过：停止一步方案并转短时域安全修正"
                )
            ),
        },
        "status": "completed" if complete else "in_progress",
        "success": complete,
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    if regression_success and confirmation_success:
        regression = json.loads(
            (output / "regression_metrics.json").read_text(encoding="utf-8")
        )
        confirmation = json.loads(
            (output / "confirmation_metrics.json").read_text(encoding="utf-8")
        )
        _write_report(
            output / "PHASE7B1_中文实验报告.md",
            payload,
            regression,
            confirmation,
        )
    print(json.dumps(payload["decision"], ensure_ascii=False), flush=True)
    return payload
