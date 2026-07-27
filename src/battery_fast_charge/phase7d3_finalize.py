"""JSON-safe postprocessing for completed Level 4-3 cached trajectories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .phase7cr3_runner import _model_hashes
from .phase7d2_runner import _summary_row
from .phase7d3_config import Phase7D3Config


CONTROL_RUNNER_RELATIVE = "src/battery_fast_charge/phase7d3_runner.py"
FINALIZER_RELATIVE = "src/battery_fast_charge/phase7d3_finalize.py"


def _sha256(path: Path, normalize_text: bool = False) -> str:
    payload = path.read_bytes()
    if normalize_text:
        payload = payload.replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def finalize_completed_confirmation(
    config: Phase7D3Config, root: Path
) -> dict[str, Any]:
    output = config.section("output")
    data_dir = root / output["data_directory"]
    result_dir = root / output["result_directory"]
    result_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = data_dir / "state_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze["config_sha256"] != _sha256(
        root / "configs/phase7d3_final_confirmation.yaml", True
    ):
        raise RuntimeError("Frozen Level 4-3 config changed")
    if freeze["runner_sha256"] != _sha256(root / CONTROL_RUNNER_RELATIVE, True):
        raise RuntimeError("Frozen Level 4-3 control runner changed")
    sources = config.section("sources")
    if freeze["selection_sha256"] != _sha256(root / sources["phase7d2_selection"]):
        raise RuntimeError("Frozen Level 4-2 selection changed")
    if freeze["validation_sha256"] != _sha256(
        root / sources["phase7d2_internal_validation"]
    ):
        raise RuntimeError("Frozen Level 4-2 validation changed")
    for name, record in freeze["files"].items():
        if _sha256(data_dir / name) != record["sha256"]:
            raise RuntimeError(f"Frozen Level 4-3 state changed: {name}")

    baseline_path = data_dir / "baseline_trajectories.csv"
    selected_path = data_dir / "selected_trajectories.csv"
    baseline_metrics_path = data_dir / "baseline_trajectory_metrics.csv"
    selected_metrics_path = data_dir / "selected_trajectory_metrics.csv"
    timing_path = data_dir / "selected_seed_timing.csv"
    required = (
        baseline_path,
        selected_path,
        baseline_metrics_path,
        selected_metrics_path,
        timing_path,
    )
    if not all(path.exists() for path in required):
        raise RuntimeError("Level 4-3 cached confirmation is incomplete")

    baseline = pd.read_csv(baseline_path)
    selected = pd.read_csv(selected_path)
    baseline_metrics = pd.read_csv(baseline_metrics_path)
    selected_metrics = pd.read_csv(selected_metrics_path)
    timing = pd.read_csv(timing_path)
    thermal = config.section("thermal_candidates")
    baseline_summary = _summary_row(
        config,
        baseline,
        baseline_metrics,
        "baseline",
        float(thermal["baseline_temperature_guard_c"]),
    )
    selected_summary = _summary_row(
        config,
        selected,
        selected_metrics,
        "thermal_guard_025mc",
        float(thermal["selected_temperature_guard_c"]),
    )
    baseline_30 = baseline_metrics[baseline_metrics.ambient_temperature_c == 30]
    selected_30 = selected_metrics[selected_metrics.ambient_temperature_c == 30]
    improvement_30 = float(
        (baseline_30.charge_time_s.mean() - selected_30.charge_time_s.mean())
        / baseline_30.charge_time_s.mean()
    )
    minimum_speedup = float(timing.end_to_end_speedup.min())
    seeds = tuple(int(seed) for seed in config.section("datasets")["confirmation_seeds"])
    contract = config.section("confirmation_contract")
    checks = {
        "baseline_strictly_safe": bool(baseline_summary["strictly_safe"]),
        "selected_strictly_safe": bool(selected_summary["strictly_safe"]),
        "expected_baseline_trajectory_count": bool(
            len(baseline_metrics)
            == int(contract["expected_trajectory_count_per_variant"])
        ),
        "expected_selected_trajectory_count": bool(
            len(selected_metrics)
            == int(contract["expected_trajectory_count_per_variant"])
        ),
        "all_five_seeds_present": bool(set(selected_metrics.seed) == set(seeds)),
        "overall_charge_time_noninferior": bool(
            selected_metrics.charge_time_s.mean()
            <= baseline_metrics.charge_time_s.mean()
        ),
        "30c_improvement_meets_contract": bool(
            improvement_30
            >= float(contract["minimum_30c_mean_charge_time_improvement_fraction"])
        ),
        "all_seed_speedups_above_100": bool(
            minimum_speedup
            > float(config.section("gates")["minimum_end_to_end_speedup"])
        ),
        "confirmation_not_used_for_retuning": True,
    }
    success = bool(all(checks.values()))
    seed_timing = [
        {
            "seed": int(row.seed),
            "ann_total_mean_ms": float(row.ann_total_mean_ms),
            "end_to_end_speedup": float(row.end_to_end_speedup),
        }
        for row in timing.itertuples(index=False)
    ]
    payload = {
        "phase": config.payload["phase"],
        "status": "strict_passed" if success else "strict_stop_failed",
        "selected_temperature_guard_c": float(
            thermal["selected_temperature_guard_c"]
        ),
        "prediction_horizon_s": float(thermal["prediction_horizon_s"]),
        "baseline_summary": baseline_summary,
        "selected_summary": selected_summary,
        "baseline_30c_mean_charge_time_s": float(
            baseline_30.charge_time_s.mean()
        ),
        "selected_30c_mean_charge_time_s": float(
            selected_30.charge_time_s.mean()
        ),
        "confirmation_30c_improvement_fraction": improvement_30,
        "minimum_seed_end_to_end_speedup": minimum_speedup,
        "seed_timing": seed_timing,
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "success": success,
        "frozen_model_hashes": _model_hashes(root),
        "confirmation_used_for_retuning": False,
        "ann_retrained": False,
        "postprocessing": {
            "reason": "numpy_boolean_json_serialization_only",
            "control_runner_unchanged": True,
            "cached_trajectory_count": int(len(baseline_metrics) + len(selected_metrics)),
            "finalizer_sha256": _sha256(root / FINALIZER_RELATIVE, True),
        },
        "decision": {"level4_completed": success, "next_phase_authorized": success},
    }
    metrics_path = result_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = result_dir / "PHASE7D_LEVEL4_中文实验报告.md"
    report_path.write_text(
        "# Phase 7D / Level 4 性能优化最终报告\n\n"
        f"- 状态：{payload['status']}；\n"
        f"- 30 ℃平均充电时间改善：{100 * improvement_30:.4f}%；\n"
        f"- 最高温度：{selected_summary['maximum_temperature_c']:.6f} ℃；\n"
        f"- 最高电压：{selected_summary['maximum_voltage_v']:.6f} V；\n"
        f"- 最低五种子端到端加速：{minimum_speedup:.2f}×；\n"
        "- ANN、电压裕量和300 s预测窗口保持冻结；确认集未用于调参。\n"
        "- 闭环runner未修改；最终JSON由独立后处理器从已完成缓存生成。\n",
        encoding="utf-8",
    )
    artifacts = (
        "configs/phase7d3_final_confirmation.yaml",
        CONTROL_RUNNER_RELATIVE,
        FINALIZER_RELATIVE,
        str(freeze_path.relative_to(root)),
        *(str(path.relative_to(root)) for path in required),
        str(metrics_path.relative_to(root)),
        str(report_path.relative_to(root)),
    )
    manifest = {
        "phase": config.payload["phase"],
        "status": payload["status"],
        "level4_completed": success,
        "confirmation_used_for_retuning": False,
        "control_runner_unchanged": True,
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
