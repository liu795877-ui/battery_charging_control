"""Phase 2R-A：Chen2020 降阶电池模型充分性审计。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pybamm
from scipy.optimize import least_squares

from .identification import build_ocv_function, fit_electrical_2rc
from .phase2_config import PhaseTwoConfig
from .phase2r_config import PhaseTwoRConfig
from .reduced_model import simulate_electrical_2rc, simulate_two_node_thermal


def _solution_frame(
    solution: pybamm.Solution,
    initial_soc: float,
    nominal_capacity_ah: float,
    profile_name: str,
) -> pd.DataFrame:
    def values(name: str) -> np.ndarray:
        return np.asarray(solution[name].entries, dtype=float).reshape(-1)

    discharge_capacity = values("Discharge capacity [A.h]")
    return pd.DataFrame(
        {
            "time_s": values("Time [s]"),
            "charge_current_a": -values("Current [A]"),
            "terminal_voltage_v": values("Terminal voltage [V]"),
            "battery_ocv_v": values("Battery open-circuit voltage [V]"),
            "soc": initial_soc - discharge_capacity / nominal_capacity_ah,
            "average_temperature_c": values("Volume-averaged cell temperature [C]"),
            "total_heating_w": values("Total heating [W]"),
            "profile_name": profile_name,
            "initial_soc": initial_soc,
        }
    ).drop_duplicates("time_s", keep="last").reset_index(drop=True)


def _run_dfn(
    phase2: PhaseTwoConfig,
    audit: PhaseTwoRConfig,
    temperature_c: float,
    initial_soc: float,
    steps: list[str],
    profile_name: str,
) -> pd.DataFrame:
    model = pybamm.lithium_ion.DFN(options={"thermal": phase2.battery.thermal_model})
    parameters = pybamm.ParameterValues(phase2.battery.parameter_set)
    parameters.update(
        {
            "Ambient temperature [K]": temperature_c + 273.15,
            "Initial temperature [K]": temperature_c + 273.15,
            "Upper voltage cut-off [V]": audit.model_audit.dfn_upper_voltage_cutoff_v,
        }
    )
    experiment = pybamm.Experiment(steps, period=f"{audit.model_audit.sample_period_s:g} seconds")
    solution = pybamm.Simulation(model, parameter_values=parameters, experiment=experiment).solve(initial_soc=initial_soc)
    frame = _solution_frame(solution, initial_soc, phase2.battery.nominal_capacity_ah, profile_name)
    frame["temperature_c"] = temperature_c
    return frame


def generate_phase_two_r_dfn_data(
    phase2: PhaseTwoConfig,
    config: PhaseTwoRConfig,
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成高 SOC/多温度 OCV 和独立 300 s 充电脉冲，支持逐文件恢复。"""
    data_dir.mkdir(parents=True, exist_ok=True)
    ocv_path = data_dir / "dfn_ocv_temperature_grid.csv"
    if ocv_path.exists():
        ocv = pd.read_csv(ocv_path)
    else:
        records: list[dict[str, float]] = []
        for temperature in config.model_audit.temperatures_c:
            for soc in config.model_audit.ocv_soc_points:
                frame = _run_dfn(
                    phase2,
                    config,
                    temperature,
                    soc,
                    ["Rest for 10 seconds"],
                    f"ocv_t{temperature:g}_soc{soc:.2f}",
                )
                records.append(
                    {
                        "temperature_c": temperature,
                        "soc": soc,
                        "ocv_v": float(frame["battery_ocv_v"].iloc[-1]),
                    }
                )
        ocv = pd.DataFrame.from_records(records)
        ocv.to_csv(ocv_path, index=False)

    pulse_dir = data_dir / "pulse_profiles"
    pulse_dir.mkdir(exist_ok=True)
    frames: list[pd.DataFrame] = []
    for temperature in config.model_audit.temperatures_c:
        for soc in config.model_audit.initial_soc_points:
            for c_rate in config.model_audit.pulse_c_rates:
                key = f"t{temperature:g}_soc{soc:.2f}_c{c_rate:g}"
                path = pulse_dir / f"{key}.csv"
                if path.exists():
                    frame = pd.read_csv(path)
                else:
                    steps = [
                        f"Rest for {config.model_audit.rest_before_s:g} seconds",
                        f"Charge at {c_rate:g} C for {config.model_audit.pulse_duration_s:g} seconds",
                        f"Rest for {config.model_audit.rest_after_s:g} seconds",
                    ]
                    frame = _run_dfn(phase2, config, temperature, soc, steps, key)
                    frame["c_rate"] = c_rate
                    frame.to_csv(path, index=False)
                frames.append(frame)
    return ocv, pd.concat(frames, ignore_index=True)


def _temperature_ocv(ocv_grid: pd.DataFrame, temperature_c: float) -> Callable[[np.ndarray], np.ndarray]:
    table = ocv_grid[np.isclose(ocv_grid["temperature_c"], temperature_c)][["soc", "ocv_v"]]
    return build_ocv_function(table)


def fit_related_electrical_parameters(
    pulses: pd.DataFrame,
    ocv_grid: pd.DataFrame,
    phase2: PhaseTwoConfig,
    config: PhaseTwoRConfig,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for temperature in config.model_audit.temperatures_c:
        ocv_table = ocv_grid[np.isclose(ocv_grid["temperature_c"], temperature)][["soc", "ocv_v"]]
        for soc in config.model_audit.parameter_anchor_soc_points:
            subset = pulses[
                np.isclose(pulses["temperature_c"], temperature)
                & np.isclose(pulses["initial_soc"], soc)
            ].copy()
            parameters, diagnostics = fit_electrical_2rc(
                subset,
                ocv_table,
                phase2.battery.nominal_capacity_ah,
                config.model_audit.electrical_fit_maximum_evaluations,
            )
            records.append(
                {
                    "temperature_c": temperature,
                    "soc": soc,
                    **parameters,
                    **{f"fit_{name}": value for name, value in diagnostics.items()},
                }
            )
    return pd.DataFrame.from_records(records)


def _interpolate_related_parameters(grid: pd.DataFrame, temperature_c: float, soc: float) -> dict[str, float]:
    names = ("r0_ohm", "r1_ohm", "tau1_s", "r2_ohm", "tau2_s")
    by_temperature: dict[float, dict[str, float]] = {}
    for temperature, group in grid.groupby("temperature_c"):
        ordered = group.sort_values("soc")
        by_temperature[float(temperature)] = {
            name: float(np.interp(soc, ordered["soc"], ordered[name])) for name in names
        }
    temperatures = np.asarray(sorted(by_temperature), dtype=float)
    result = {
        name: float(np.interp(temperature_c, temperatures, [by_temperature[value][name] for value in temperatures]))
        for name in names
    }
    result["c1_f"] = result["tau1_s"] / result["r1_ohm"]
    result["c2_f"] = result["tau2_s"] / result["r2_ohm"]
    return result


def simulate_single_node_thermal(
    time_s: np.ndarray,
    heat_w: np.ndarray,
    initial_temperature_c: float,
    ambient_temperature_c: float,
    tau_s: float,
    gain_k_per_w: float,
) -> np.ndarray:
    temperature = np.empty(len(time_s), dtype=float)
    temperature[0] = initial_temperature_c
    for index in range(1, len(time_s)):
        dt = float(time_s[index] - time_s[index - 1])
        decay = np.exp(-dt / tau_s)
        equilibrium = ambient_temperature_c + gain_k_per_w * heat_w[index - 1]
        temperature[index] = equilibrium + decay * (temperature[index - 1] - equilibrium)
    return temperature


def fit_temperature_specific_thermal_models(
    pulses: pd.DataFrame,
    fixed_thermal: dict[str, float],
    core_fraction: float,
    config: PhaseTwoRConfig,
) -> pd.DataFrame:
    """在 60/70/80% 锚点上分别拟合单节点结构和双节点热增益。"""
    anchor = pulses[pulses["initial_soc"].isin(config.model_audit.parameter_anchor_soc_points)]
    records: list[dict[str, float]] = []
    for temperature, temperature_frame in anchor.groupby("temperature_c"):
        groups = [group.sort_values("time_s") for _, group in temperature_frame.groupby("profile_name")]

        def single_residual(log_values: np.ndarray) -> np.ndarray:
            tau, gain = np.exp(log_values)
            return np.concatenate(
                [
                    simulate_single_node_thermal(
                        group["time_s"].to_numpy(),
                        group["total_heating_w"].to_numpy(),
                        float(group["average_temperature_c"].iloc[0]),
                        float(temperature),
                        float(tau),
                        float(gain),
                    )
                    - group["average_temperature_c"].to_numpy()
                    for group in groups
                ]
            )

        single = least_squares(
            single_residual,
            np.log([500.0, 5.0]),
            bounds=(np.log([10.0, 0.01]), np.log([10000.0, 100.0])),
        )
        tau, gain = np.exp(single.x)

        def dual_residual(log_gain: np.ndarray) -> np.ndarray:
            parameters = dict(fixed_thermal)
            parameters["heat_gain"] = float(np.exp(log_gain[0]))
            errors = []
            for group in groups:
                prediction = simulate_two_node_thermal(
                    group["time_s"].to_numpy(),
                    group["total_heating_w"].to_numpy(),
                    float(group["average_temperature_c"].iloc[0]),
                    float(temperature),
                    core_fraction,
                    parameters,
                )
                errors.append(prediction["average_temperature_predicted_c"].to_numpy() - group["average_temperature_c"].to_numpy())
            return np.concatenate(errors)

        dual = least_squares(dual_residual, np.log([fixed_thermal["heat_gain"]]), bounds=(np.log([0.01]), np.log([100.0])))
        records.append(
            {
                "temperature_c": float(temperature),
                "single_tau_s": float(tau),
                "single_gain_k_per_w": float(gain),
                "single_training_rmse_c": float(np.sqrt(np.mean(single.fun**2))),
                "dual_heat_gain": float(np.exp(dual.x[0])),
                "dual_training_rmse_c": float(np.sqrt(np.mean(dual.fun**2))),
            }
        )
    return pd.DataFrame.from_records(records)


def _classification(true_feasible: bool, predicted_feasible: bool) -> str:
    if true_feasible and predicted_feasible:
        return "true_feasible"
    if not true_feasible and not predicted_feasible:
        return "true_infeasible"
    if not true_feasible and predicted_feasible:
        return "false_safe"
    return "false_conservative"


def evaluate_model_variants(
    pulses: pd.DataFrame,
    ocv_grid: pd.DataFrame,
    related_grid: pd.DataFrame,
    thermal_grid: pd.DataFrame,
    fixed_ocv_table: pd.DataFrame,
    fixed_parameters: dict[str, Any],
    phase2: PhaseTwoConfig,
    config: PhaseTwoRConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fixed_ocv = build_ocv_function(fixed_ocv_table)
    fixed_electrical = fixed_parameters["electrical_2rc"]
    fixed_thermal = fixed_parameters["thermal_two_node"]
    core_fraction = float(fixed_parameters["core_heat_capacity_fraction"])
    prediction_frames: list[pd.DataFrame] = []
    horizon_records: list[dict[str, Any]] = []
    structure_records: list[dict[str, Any]] = []

    for profile_name, frame in pulses.groupby("profile_name", sort=True):
        frame = frame.sort_values("time_s").reset_index(drop=True)
        temperature = float(frame["temperature_c"].iloc[0])
        initial_soc = float(frame["initial_soc"].iloc[0])
        related_ocv = _temperature_ocv(ocv_grid, temperature)
        related_electrical = _interpolate_related_parameters(related_grid, temperature, initial_soc)
        electrical_predictions = {
            "fixed_2rc_dual": simulate_electrical_2rc(
                frame["time_s"].to_numpy(), frame["charge_current_a"].to_numpy(), initial_soc,
                phase2.battery.nominal_capacity_ah, fixed_ocv, fixed_electrical,
            ),
            "related_2rc_dual": simulate_electrical_2rc(
                frame["time_s"].to_numpy(), frame["charge_current_a"].to_numpy(), initial_soc,
                phase2.battery.nominal_capacity_ah, related_ocv, related_electrical,
            ),
            "related_2rc_single": simulate_electrical_2rc(
                frame["time_s"].to_numpy(), frame["charge_current_a"].to_numpy(), initial_soc,
                phase2.battery.nominal_capacity_ah, related_ocv, related_electrical,
            ),
        }
        thermal_row = thermal_grid.iloc[int(np.argmin(np.abs(thermal_grid["temperature_c"].to_numpy() - temperature)))]
        for variant, electrical in electrical_predictions.items():
            if variant == "fixed_2rc_dual":
                thermal_parameters = fixed_thermal
                thermal = simulate_two_node_thermal(
                    frame["time_s"].to_numpy(), electrical["electrical_loss_predicted_w"].to_numpy(),
                    float(frame["average_temperature_c"].iloc[0]), temperature, core_fraction, thermal_parameters,
                )["average_temperature_predicted_c"].to_numpy()
            elif variant == "related_2rc_dual":
                thermal_parameters = dict(fixed_thermal)
                thermal_parameters["heat_gain"] = float(thermal_row["dual_heat_gain"])
                thermal = simulate_two_node_thermal(
                    frame["time_s"].to_numpy(), electrical["electrical_loss_predicted_w"].to_numpy(),
                    float(frame["average_temperature_c"].iloc[0]), temperature, core_fraction, thermal_parameters,
                )["average_temperature_predicted_c"].to_numpy()
            else:
                thermal = simulate_single_node_thermal(
                    frame["time_s"].to_numpy(), electrical["electrical_loss_predicted_w"].to_numpy(),
                    float(frame["average_temperature_c"].iloc[0]), temperature,
                    float(thermal_row["single_tau_s"]), float(thermal_row["single_gain_k_per_w"]),
                )
            combined = frame.copy()
            combined["variant"] = variant
            combined["terminal_voltage_predicted_v"] = electrical["terminal_voltage_predicted_v"].to_numpy()
            combined["average_temperature_predicted_c"] = thermal
            prediction_frames.append(combined)

        # 使用真实 DFN 热源单独比较热结构，避免电模型误差混入。
        fixed_dual_truth_heat = simulate_two_node_thermal(
            frame["time_s"].to_numpy(), frame["total_heating_w"].to_numpy(),
            float(frame["average_temperature_c"].iloc[0]), temperature, core_fraction, fixed_thermal,
        )["average_temperature_predicted_c"].to_numpy()
        related_dual_parameters = dict(fixed_thermal)
        related_dual_parameters["heat_gain"] = float(thermal_row["dual_heat_gain"])
        related_dual_truth_heat = simulate_two_node_thermal(
            frame["time_s"].to_numpy(), frame["total_heating_w"].to_numpy(),
            float(frame["average_temperature_c"].iloc[0]), temperature, core_fraction, related_dual_parameters,
        )["average_temperature_predicted_c"].to_numpy()
        single_truth_heat = simulate_single_node_thermal(
            frame["time_s"].to_numpy(), frame["total_heating_w"].to_numpy(),
            float(frame["average_temperature_c"].iloc[0]), temperature,
            float(thermal_row["single_tau_s"]), float(thermal_row["single_gain_k_per_w"]),
        )
        actual_temperature = frame["average_temperature_c"].to_numpy()
        for structure, values in (
            ("fixed_dual", fixed_dual_truth_heat),
            ("temperature_related_dual", related_dual_truth_heat),
            ("temperature_related_single", single_truth_heat),
        ):
            structure_records.append(
                {
                    "profile_name": profile_name,
                    "temperature_c": temperature,
                    "initial_soc": initial_soc,
                    "c_rate": float(frame["c_rate"].iloc[0]),
                    "holdout_soc": initial_soc not in config.model_audit.parameter_anchor_soc_points,
                    "thermal_structure": structure,
                    "rmse_c": float(np.sqrt(np.mean((values - actual_temperature) ** 2))),
                    "maximum_absolute_error_c": float(np.max(np.abs(values - actual_temperature))),
                }
            )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    for (profile_name, variant), frame in predictions.groupby(["profile_name", "variant"], sort=True):
        pulse_indices = np.flatnonzero(frame["charge_current_a"].to_numpy() > 1.0e-8)
        if len(pulse_indices) == 0:
            continue
        start_time = float(frame["time_s"].iloc[pulse_indices[0]])
        for horizon in config.model_audit.prediction_horizons_s:
            window = frame[(frame["time_s"] >= start_time) & (frame["time_s"] <= start_time + horizon + 1.0e-9)]
            if window.empty or float(window["time_s"].max()) < start_time + horizon - config.model_audit.sample_period_s:
                continue
            voltage_error = window["terminal_voltage_predicted_v"] - window["terminal_voltage_v"]
            temperature_error = window["average_temperature_predicted_c"] - window["average_temperature_c"]
            true_voltage_feasible = bool(window["terminal_voltage_v"].max() <= config.model_audit.physical_maximum_voltage_v)
            predicted_voltage_feasible = bool(window["terminal_voltage_predicted_v"].max() <= config.model_audit.physical_maximum_voltage_v)
            true_temperature_feasible = bool(window["average_temperature_c"].max() <= config.model_audit.physical_maximum_temperature_c)
            predicted_temperature_feasible = bool(window["average_temperature_predicted_c"].max() <= config.model_audit.physical_maximum_temperature_c)
            horizon_records.append(
                {
                    "profile_name": profile_name,
                    "variant": variant,
                    "temperature_c": float(window["temperature_c"].iloc[0]),
                    "initial_soc": float(window["initial_soc"].iloc[0]),
                    "c_rate": float(window["c_rate"].iloc[0]),
                    "holdout_soc": float(window["initial_soc"].iloc[0]) not in config.model_audit.parameter_anchor_soc_points,
                    "horizon_s": horizon,
                    "voltage_rmse_mv": float(np.sqrt(np.mean(voltage_error**2)) * 1000.0),
                    "voltage_maximum_absolute_error_mv": float(np.max(np.abs(voltage_error)) * 1000.0),
                    "temperature_rmse_c": float(np.sqrt(np.mean(temperature_error**2))),
                    "temperature_maximum_absolute_error_c": float(np.max(np.abs(temperature_error))),
                    "true_voltage_feasible": true_voltage_feasible,
                    "predicted_voltage_feasible": predicted_voltage_feasible,
                    "voltage_classification": _classification(true_voltage_feasible, predicted_voltage_feasible),
                    "true_temperature_feasible": true_temperature_feasible,
                    "predicted_temperature_feasible": predicted_temperature_feasible,
                    "temperature_classification": _classification(true_temperature_feasible, predicted_temperature_feasible),
                }
            )
    return predictions, pd.DataFrame.from_records(horizon_records), pd.DataFrame.from_records(structure_records)


def summarize_model_audit(
    horizon_metrics: pd.DataFrame,
    thermal_structure: pd.DataFrame,
    config: PhaseTwoRConfig,
) -> dict[str, Any]:
    holdout = horizon_metrics[horizon_metrics["holdout_soc"]]
    summary = (
        holdout.groupby(["variant", "horizon_s"])
        .agg(
            voltage_rmse_mv_mean=("voltage_rmse_mv", "mean"),
            voltage_rmse_mv_max=("voltage_rmse_mv", "max"),
            temperature_rmse_c_mean=("temperature_rmse_c", "mean"),
            temperature_rmse_c_max=("temperature_rmse_c", "max"),
            voltage_false_safe_count=("voltage_classification", lambda values: int((values == "false_safe").sum())),
            temperature_false_safe_count=("temperature_classification", lambda values: int((values == "false_safe").sum())),
        )
        .reset_index()
    )
    thermal_holdout = thermal_structure[thermal_structure["holdout_soc"]]
    thermal_summary = (
        thermal_holdout.groupby("thermal_structure")
        .agg(rmse_c_mean=("rmse_c", "mean"), rmse_c_max=("rmse_c", "max"), maximum_absolute_error_c=("maximum_absolute_error_c", "max"))
        .reset_index()
    )
    variant_checks: dict[str, Any] = {}
    for variant, group in summary.groupby("variant"):
        variant_checks[str(variant)] = {
            "all_horizon_mean_voltage_rmse_within_limit": bool((group["voltage_rmse_mv_mean"] <= config.model_audit.voltage_rmse_limit_mv).all()),
            "all_horizon_mean_temperature_rmse_within_limit": bool((group["temperature_rmse_c_mean"] <= config.model_audit.temperature_rmse_limit_c).all()),
            "voltage_false_safe_count": int(group["voltage_false_safe_count"].sum()),
            "temperature_false_safe_count": int(group["temperature_false_safe_count"].sum()),
        }
    return {
        "holdout_summary": summary.to_dict("records"),
        "thermal_structure_summary": thermal_summary.to_dict("records"),
        "variant_checks": variant_checks,
    }
