"""Level 4 entry and frozen multi-temperature performance attribution audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .phase7d_config import Phase7DConfig


CONFIG_RELATIVE = "configs/phase7d_level4_performance.yaml"


def _sha256(path: Path, normalize_text: bool = False) -> str:
    payload = path.read_bytes()
    if normalize_text:
        payload = payload.replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _verify_r3t2_authorization(config: Phase7DConfig, root: Path) -> dict[str, Any]:
    entry = config.section("entry_authorization")
    metrics_path = root / entry["r3t2_metrics"]
    manifest_path = root / entry["r3t2_manifest"]
    if _sha256(metrics_path) != entry["r3t2_metrics_sha256"]:
        raise RuntimeError("R3T2 metrics changed after Level 4 authorization")
    if _sha256(manifest_path) != entry["r3t2_manifest_sha256"]:
        raise RuntimeError("R3T2 freeze manifest changed after Level 4 authorization")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if metrics["status"] != entry["required_status"] or not metrics["success"]:
        raise RuntimeError("R3T2 did not strictly pass")
    if metrics["failed_checks"]:
        raise RuntimeError("R3T2 contains failed checks")
    if not metrics["decision"]["level4_authorized"]:
        raise RuntimeError("R3T2 did not authorize Level 4")
    if not manifest["level4_authorized"]:
        raise RuntimeError("R3T2 manifest did not authorize Level 4")
    return {
        "metrics_sha256": _sha256(metrics_path),
        "manifest_sha256": _sha256(manifest_path),
        "minimum_end_to_end_speedup": metrics["timing_summary"][
            "minimum_end_to_end_speedup"
        ],
        "r3t2_level4_entered": metrics["decision"]["level4_entered"],
    }


def _temperature_summary(frame: pd.DataFrame, voltage_limit: float, temperature_limit: float) -> dict[str, Any]:
    maximum_voltage = float(frame.maximum_voltage_v.max())
    maximum_temperature = float(frame.maximum_temperature_c.max())
    return {
        "trajectory_count": int(len(frame)),
        "mean_charge_time_s": float(frame.charge_time_s.mean()),
        "minimum_charge_time_s": float(frame.charge_time_s.min()),
        "maximum_charge_time_s": float(frame.charge_time_s.max()),
        "maximum_voltage_v": maximum_voltage,
        "voltage_headroom_v": float(voltage_limit - maximum_voltage),
        "maximum_temperature_c": maximum_temperature,
        "temperature_headroom_c": float(temperature_limit - maximum_temperature),
        "mean_voltage_intervention_fraction": float(
            frame.voltage_intervention_fraction.mean()
        ),
        "maximum_voltage_intervention_fraction": float(
            frame.voltage_intervention_fraction.max()
        ),
        "mean_thermal_intervention_fraction": float(
            frame.thermal_intervention_fraction.mean()
        ),
        "maximum_thermal_intervention_fraction": float(
            frame.thermal_intervention_fraction.max()
        ),
        "mean_both_layers_fraction": float(frame.both_layers_fraction.mean()),
        "maximum_voltage_correction_a": float(
            frame.maximum_voltage_correction_a.max()
        ),
        "maximum_thermal_correction_a": float(
            frame.maximum_thermal_correction_a.max()
        ),
    }


def run_phase7d_baseline(config: Phase7DConfig, root: Path) -> dict[str, Any]:
    authorization = _verify_r3t2_authorization(config, root)
    baseline = config.section("baseline")
    contract = config.section("level4_contract")
    source_path = root / baseline["trajectory_metrics"]
    if _sha256(source_path) != baseline["trajectory_metrics_sha256"]:
        raise RuntimeError("R3T2 trajectory metrics changed")

    trajectories = pd.read_csv(source_path)
    canonical_prefix = f"confirmation_{baseline['canonical_repetition']}_"
    trajectories = trajectories[trajectories.role.str.startswith(canonical_prefix)].copy()
    temperatures = tuple(int(value) for value in baseline["temperatures_c"])
    trajectories = trajectories[
        trajectories.ambient_temperature_c.isin(temperatures)
    ].copy()
    if len(trajectories) != int(baseline["expected_trajectory_count"]):
        raise RuntimeError("Unexpected Level 4 baseline trajectory count")

    zero_columns = (
        "guard_exceedance_count",
        "empty_voltage_slew_count",
        "empty_thermal_slew_count",
        "solver_failure_count",
        "prediction_infeasible_count",
        "sustained_oscillation_count",
    )
    checks = {
        "target_reach_100_percent": bool(trajectories.target_reached.all()),
        "all_frozen_safety_counts_zero": all(
            int(trajectories[column].sum()) == 0 for column in zero_columns
        ),
        "voltage_safe": bool(
            trajectories.maximum_voltage_v.max() <= baseline["voltage_limit_v"]
        ),
        "temperature_safe": bool(
            trajectories.maximum_temperature_c.max()
            <= baseline["temperature_limit_c"]
        ),
        "level4_scope_is_audit_only": bool(
            contract["level4_entered"]
            and contract["stage"] == "baseline_attribution_only"
            and not contract["ann_retraining_authorized"]
            and not contract["safety_contract_change_authorized"]
            and not contract["r3_confirmation_tuning_authorized"]
            and not contract["optimization_parameters_changed"]
        ),
    }
    by_temperature: dict[str, Any] = {}
    for temperature in temperatures:
        subset = trajectories[trajectories.ambient_temperature_c == temperature]
        if len(subset) != int(baseline["expected_trajectories_per_temperature"]):
            raise RuntimeError(f"Unexpected {temperature} C trajectory count")
        by_temperature[str(temperature)] = _temperature_summary(
            subset,
            float(baseline["voltage_limit_v"]),
            float(baseline["temperature_limit_c"]),
        )

    success = all(checks.values())
    payload = {
        "phase": config.payload["phase"],
        "status": "baseline_frozen" if success else "strict_stop_failed",
        "authorization": authorization,
        "canonical_source": {
            "repetition": baseline["canonical_repetition"],
            "trajectory_metrics_sha256": _sha256(source_path),
            "confirmation_used_for_tuning": False,
        },
        "baseline_by_temperature": by_temperature,
        "optimization_priorities": {
            "15c": "voltage-layer conservatism and charge-time performance",
            "30c": "thermal-layer charge-time performance under the binding 35 C limit",
            "primary_metric": contract["primary_objective"],
            "intervention_rate_is_secondary_not_a_safety_replacement": True,
        },
        "checks": checks,
        "success": success,
        "decision": {
            "level4_authorized": True,
            "level4_entered": success,
            "level4_1_authorized": success,
            "ann_retrained": False,
            "safety_contract_changed": False,
            "optimization_parameters_changed": False,
            "r3_confirmation_used_for_tuning": False,
        },
    }

    output_dir = root / config.section("output")["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "PHASE7D_LEVEL4_0_中文实验报告.md"
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path.write_text(
        "# Phase 7D / Level 4-0 性能基线归因审计\n\n"
        f"- 状态：{payload['status']}；\n"
        "- R3T2严格通过并授权Level 4，本报告正式记录Level 4进入；\n"
        "- R3确认数据仅用于冻结基线归因，不用于调参；\n"
        f"- 15 ℃平均电压层介入率：{by_temperature['15']['mean_voltage_intervention_fraction']:.4%}；\n"
        f"- 30 ℃平均热层介入率：{by_temperature['30']['mean_thermal_intervention_fraction']:.4%}；\n"
        "- 下一步：使用全新开发集和内部验证集执行Level 4-1性能优化。\n",
        encoding="utf-8",
    )
    artifacts = (
        CONFIG_RELATIVE,
        "pyproject.toml",
        "src/battery_fast_charge/phase7d_config.py",
        "src/battery_fast_charge/phase7d_runner.py",
        "src/battery_fast_charge/phase7d_cli.py",
        str(source_path.relative_to(root)),
        str(metrics_path.relative_to(root)),
        str(report_path.relative_to(root)),
    )
    manifest = {
        "phase": config.payload["phase"],
        "status": payload["status"],
        "level4_entered": success,
        "level4_1_authorized": success,
        "r3_confirmation_used_for_tuning": False,
        "artifacts": {
            relative: _sha256(root / relative, Path(relative).suffix in {".py", ".yaml", ".md"})
            for relative in artifacts
        },
    }
    (output_dir / "freeze_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload
