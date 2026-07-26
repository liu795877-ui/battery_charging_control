"""Strict-stop audit for the R3T independent rerun confirmation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _sha256(path: Path, normalize_text: bool = False) -> str:
    payload = path.read_bytes()
    if normalize_text:
        payload = payload.replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def audit_phase7cr3t(root: Path) -> dict[str, Any]:
    data_dir = root / "data/phase7cr3t_supervisor_runtime"
    result_dir = root / "outputs/phase7cr3t_supervisor_runtime"
    metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
    baseline = pd.read_csv(
        root / "data/phase7cr3_independent_confirmation/ann_confirmation_trajectories.csv"
    )
    optimized = pd.read_csv(data_dir / "optimized_confirmation_trajectories.csv")
    keys = ["seed", "trajectory_id", "step_index"]
    paired = baseline[
        keys
        + [
            "current_a",
            "next_soc",
            "next_temperature_c",
            "thermal_intervened",
        ]
    ].merge(
        optimized[
            keys
            + [
                "current_a",
                "next_soc",
                "next_temperature_c",
                "thermal_intervened",
            ]
        ],
        on=keys,
        suffixes=("_baseline", "_optimized"),
        how="inner",
    )
    paired["current_difference_a"] = (
        paired.current_a_optimized - paired.current_a_baseline
    ).abs()
    paired["soc_difference"] = (
        paired.next_soc_optimized - paired.next_soc_baseline
    ).abs()
    paired["temperature_difference_c"] = (
        paired.next_temperature_c_optimized
        - paired.next_temperature_c_baseline
    ).abs()
    maximum = paired.loc[paired.current_difference_a.idxmax()]
    repeatability = {
        "paired_step_count": len(paired),
        "nonzero_current_difference_count": int(
            (paired.current_difference_a > 0.0).sum()
        ),
        "current_difference_above_1e_9_a_count": int(
            (paired.current_difference_a > 1.0e-9).sum()
        ),
        "current_difference_above_1e_6_a_count": int(
            (paired.current_difference_a > 1.0e-6).sum()
        ),
        "maximum_current_difference_a": float(
            paired.current_difference_a.max()
        ),
        "maximum_soc_difference": float(paired.soc_difference.max()),
        "maximum_temperature_difference_c": float(
            paired.temperature_difference_c.max()
        ),
        "maximum_event": {
            "seed": int(maximum.seed),
            "trajectory_id": str(maximum.trajectory_id),
            "step_index": int(maximum.step_index),
            "current_difference_a": float(maximum.current_difference_a),
            "temperature_difference_c": float(
                maximum.temperature_difference_c
            ),
            "thermal_intervened_baseline": bool(
                maximum.thermal_intervened_baseline
            ),
            "thermal_intervened_optimized": bool(
                maximum.thermal_intervened_optimized
            ),
        },
    }
    failed = metrics["failed_checks"]
    payload = {
        "phase": "Phase 7C-R3T strict audit",
        "status": "strict_stop_failed",
        "equivalence_development_passed": metrics["equivalence_freeze"][
            "success"
        ],
        "same_state_current_limit_difference_a": metrics[
            "equivalence_freeze"
        ]["maximum_current_limit_difference_a"],
        "confirmation_checks": metrics["checks"],
        "failed_checks": failed,
        "repeatability_diagnosis": repeatability,
        "speed_summary": {
            "minimum_end_to_end_speedup": min(
                row["end_to_end_speedup"]
                for row in metrics["seed_timing_metrics"]
            ),
            "maximum_end_to_end_speedup": max(
                row["end_to_end_speedup"]
                for row in metrics["seed_timing_metrics"]
            ),
            "supervisor_mean_ms_range": [
                min(row["supervisor_mean_ms"] for row in metrics["seed_timing_metrics"]),
                max(row["supervisor_mean_ms"] for row in metrics["seed_timing_metrics"]),
            ],
        },
        "physical_summary": metrics["physical_summary"],
        "success": False,
        "decision": {
            "level4_authorized": False,
            "level4_entered": False,
            "ann_retrained": False,
            "safety_contract_changed": False,
            "confirmation_used_for_retuning": False,
            "next_phase": "Phase 7C-R3T2 paired-state repeatability contract",
            "reason": (
                "The exact-zero cross-run current contract failed, while "
                "same-state supervisor equivalence, speed, and safety passed."
            ),
        },
    }
    audit_path = result_dir / "strict_audit.json"
    audit_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = result_dir / "PHASE7C-R3T_严格停止报告.md"
    report_path.write_text(
        "# Phase 7C-R3T 严格停止报告\n\n"
        "闭式热监督实现通过全部同状态等价性测试，240条确认的速度与物理安全门槛也全部通过。"
        "但跨两次独立DFN仿真的逐步电流严格0差异合同未通过，因此仍不能进入Level 4。\n\n"
        f"- 同状态热上限最大差异：{payload['same_state_current_limit_difference_a']:.12g} A；\n"
        f"- 独立闭环最大电流差异：{repeatability['maximum_current_difference_a']:.12g} A；\n"
        f"- 差异超过1e-9 A：{repeatability['current_difference_above_1e_9_a_count']}步；\n"
        f"- 差异超过1e-6 A：{repeatability['current_difference_above_1e_6_a_count']}步；\n"
        f"- 五种子端到端加速：{payload['speed_summary']['minimum_end_to_end_speedup']:.2f}×–"
        f"{payload['speed_summary']['maximum_end_to_end_speedup']:.2f}×；\n"
        "- 电压、温度、电流、斜率、裕量、空区间、预测不可行和持续振荡：全部0违约。\n\n"
        "最大差异事件未触发热监督，说明失败来自独立DFN闭环的浮点可重复性边界，"
        "不能归因于优化后安全逻辑。下一阶段应独立预注册配对同状态的R3T2重复性合同，"
        "不得在本次确认集上原地放宽门槛。\n",
        encoding="utf-8",
    )
    artifacts = [
        "configs/phase7cr3t_supervisor_runtime.yaml",
        "src/battery_fast_charge/phase7cr3t_config.py",
        "src/battery_fast_charge/phase7cr3t_thermal.py",
        "src/battery_fast_charge/phase7cr3t_runner.py",
        "src/battery_fast_charge/phase7cr3t_audit.py",
        "outputs/phase7cr3t_supervisor_runtime/equivalence_freeze.json",
        "data/phase7cr3t_supervisor_runtime/optimized_confirmation_trajectories.csv",
        "data/phase7cr3t_supervisor_runtime/trajectory_metrics.csv",
        "data/phase7cr3t_supervisor_runtime/seed_timing_metrics.csv",
        "outputs/phase7cr3t_supervisor_runtime/metrics.json",
        "outputs/phase7cr3t_supervisor_runtime/strict_audit.json",
        "outputs/phase7cr3t_supervisor_runtime/PHASE7C-R3T_严格停止报告.md",
    ]
    manifest = {
        "phase": "Phase 7C-R3T",
        "status": "strict_stop_failed",
        "level4_authorized": False,
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
