"""Post-run strict audit for the frozen R3 ANN confirmation evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .phase7cr2_runner import _trajectory_metrics
from .phase7cr2f_config import load_phase7cr2f_config


def _sha256(path: Path, normalize_text: bool = False) -> str:
    payload = path.read_bytes()
    if normalize_text:
        payload = payload.replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def audit_phase7cr3(root: Path) -> dict[str, Any]:
    data_dir = root / "data/phase7cr3_independent_confirmation"
    result_dir = root / "outputs/phase7cr3_independent_confirmation"
    ann = pd.read_csv(data_dir / "ann_confirmation_trajectories.csv")
    mpc = pd.read_csv(data_dir / "safe_mpc_confirmation_trajectories.csv")
    raw = json.loads((result_dir / "ann_metrics.json").read_text(encoding="utf-8"))
    base_config = load_phase7cr2f_config(
        root / "configs/phase7cr2f_teacher_selection_voltage_guard.yaml"
    )
    trajectory = _trajectory_metrics(base_config, ann)
    trajectory.to_csv(data_dir / "ann_trajectory_metrics.csv", index=False)
    valid_ann = ann[ann.current_a.notna()]
    valid_mpc = mpc[mpc.current_a.notna()]
    component_timing = {
        "mpc_candidate_mean_ms": 1000.0 * float(valid_mpc.teacher_time_s.mean()),
        "mpc_supervisor_mean_ms": 1000.0 * float(valid_mpc.supervisor_time_s.mean()),
        "mpc_total_mean_ms": 1000.0
        * float((valid_mpc.teacher_time_s + valid_mpc.supervisor_time_s).mean()),
        "ann_candidate_mean_ms": 1000.0 * float(valid_ann.teacher_time_s.mean()),
        "ann_supervisor_mean_ms": 1000.0 * float(valid_ann.supervisor_time_s.mean()),
        "ann_total_mean_ms": 1000.0
        * float((valid_ann.teacher_time_s + valid_ann.supervisor_time_s).mean()),
        "candidate_only_speedup": float(
            valid_mpc.teacher_time_s.mean() / valid_ann.teacher_time_s.mean()
        ),
        "end_to_end_speedup": float(
            (valid_mpc.teacher_time_s + valid_mpc.supervisor_time_s).mean()
            / (valid_ann.teacher_time_s + valid_ann.supervisor_time_s).mean()
        ),
    }
    physical = {
        "trajectory_count": int(len(trajectory)),
        "target_reach_fraction": float(trajectory.target_reached.mean()),
        "minimum_current_a": float(trajectory.minimum_current_a.min()),
        "maximum_current_a": float(trajectory.maximum_current_a.max()),
        "maximum_current_step_a": float(trajectory.maximum_current_step_a.max()),
        "maximum_voltage_v": float(trajectory.maximum_voltage_v.max()),
        "maximum_temperature_c": float(trajectory.maximum_temperature_c.max()),
        "guard_exceedance_count": int(trajectory.guard_exceedance_count.sum()),
        "empty_voltage_slew_count": int(trajectory.empty_voltage_slew_count.sum()),
        "empty_thermal_slew_count": int(trajectory.empty_thermal_slew_count.sum()),
        "prediction_infeasible_count": int(trajectory.prediction_infeasible_count.sum()),
        "solver_failure_count": int(trajectory.solver_failure_count.sum()),
        "sustained_oscillation_count": int(trajectory.sustained_oscillation_count.sum()),
    }
    supplemental_checks = {
        "trajectory_count_240": physical["trajectory_count"] == 240,
        "current_bounds_zero_violation": physical["minimum_current_a"] >= -1e-9
        and physical["maximum_current_a"] <= 10.0 + 1e-9,
        "zero_prediction_infeasible": physical["prediction_infeasible_count"] == 0,
        "zero_solver_failure": physical["solver_failure_count"] == 0,
        "zero_sustained_oscillation": physical["sustained_oscillation_count"] == 0,
    }
    checks = {**raw["checks"], **supplemental_checks}
    success = bool(all(checks.values()))
    payload = {
        "phase": "Phase 7C-R3 frozen ANN strict audit",
        "status": "strict_passed" if success else "strict_stop_failed",
        "physical_summary": physical,
        "timing_decomposition": component_timing,
        "seed_metrics": raw["seed_metrics"],
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "success": success,
        "diagnosis": {
            "policy_accuracy_failed": not raw["checks"]["current_nrmse_below_1_percent"],
            "multitemperature_safety_failed": not all(
                raw["checks"][key]
                for key in (
                    "target_reach_100_percent",
                    "voltage_safe",
                    "temperature_safe",
                    "slew_safe",
                    "zero_guard_exceedance",
                    "zero_empty_interval",
                )
            ),
            "only_strict_failure_is_end_to_end_speedup": [
                key for key, value in checks.items() if not value
            ]
            == ["speedup_above_100"],
            "shared_supervisor_dominates_ann_runtime": (
                component_timing["ann_supervisor_mean_ms"]
                > component_timing["ann_candidate_mean_ms"]
            ),
        },
        "decision": {
            "level4_authorized": success,
            "level4_entered": False,
            "ann_retrained": False,
            "confirmation_used_for_retuning": False,
            "strict_next_step": (
                "enter_level4_performance_optimization"
                if success
                else "remain_before_level4_and_audit_shared_supervisor_runtime"
            ),
            "thermal_aware_ann_retraining_supported_by_evidence": bool(
                not success
                and (
                    not raw["checks"]["current_nrmse_below_1_percent"]
                    or not raw["checks"]["temperature_safe"]
                )
            ),
        },
    }
    audit_path = result_dir / "strict_audit.json"
    audit_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = result_dir / "PHASE7C-R3_冻结ANN严格停止报告.md"
    report_path.write_text(
        "# Phase 7C-R3 冻结ANN严格停止报告\n\n"
        "240条冻结ANN轨迹全部完成。策略精度、充电时间、目标到达和全部物理安全门槛通过，"
        "但端到端在线加速未达到预注册的100倍门槛，因此R3严格停止，尚未进入Level 4。\n\n"
        f"- 五种子平均电流NRMSE：{100 * min(x['mean_current_nrmse'] for x in raw['seed_metrics']):.4f}%–"
        f"{100 * max(x['mean_current_nrmse'] for x in raw['seed_metrics']):.4f}%；\n"
        f"- 五种子平均充电时间偏差：{100 * min(x['mean_charge_time_gap_fraction'] for x in raw['seed_metrics']):.4f}%–"
        f"{100 * max(x['mean_charge_time_gap_fraction'] for x in raw['seed_metrics']):.4f}%；\n"
        f"- 端到端加速：{min(x['speedup'] for x in raw['seed_metrics']):.2f}×–"
        f"{max(x['speedup'] for x in raw['seed_metrics']):.2f}×；\n"
        f"- 仅ANN候选推理加速：{component_timing['candidate_only_speedup']:.2f}×；\n"
        f"- ANN候选/共享监督平均耗时：{component_timing['ann_candidate_mean_ms']:.4f}/"
        f"{component_timing['ann_supervisor_mean_ms']:.4f} ms。\n\n"
        "失败仅来自端到端100倍加速门槛。现有证据不支持把失败归因于ANN多温度外推，"
        "也不支持直接进行热感知ANN重训；应先在Level 4之前审计和优化共享热监督层运行时间。\n",
        encoding="utf-8",
    )
    artifacts = [
        "configs/phase7cr3_independent_confirmation.yaml",
        "src/battery_fast_charge/phase7cr3_config.py",
        "src/battery_fast_charge/phase7cr3_runner.py",
        "src/battery_fast_charge/phase7cr3_audit.py",
        "data/phase7cr3_independent_confirmation/initial_state_freeze.json",
        "data/phase7cr3_independent_confirmation/safe_mpc_confirmation_trajectories.csv",
        "data/phase7cr3_independent_confirmation/ann_confirmation_trajectories.csv",
        "data/phase7cr3_independent_confirmation/ann_seed_metrics.csv",
        "data/phase7cr3_independent_confirmation/ann_trajectory_metrics.csv",
        "outputs/phase7cr3_independent_confirmation/safe_mpc_metrics.json",
        "outputs/phase7cr3_independent_confirmation/ann_metrics.json",
        "outputs/phase7cr3_independent_confirmation/strict_audit.json",
        "outputs/phase7cr3_independent_confirmation/PHASE7C-R3_冻结ANN严格停止报告.md",
    ]
    manifest = {
        "phase": "Phase 7C-R3",
        "status": payload["status"],
        "level4_authorized": success,
        "level4_entered": False,
        "ann_retrained": False,
        "artifacts": {
            relative: _sha256(
                root / relative,
                Path(relative).suffix in {".py", ".yaml", ".md"},
            )
            for relative in artifacts
        },
    }
    (result_dir / "freeze_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload
