"""Seed R2F5 validation caches for unchanged 15 C and 30 C contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from battery_fast_charge.phase7cr2f5_config import load_phase7cr2f5_config
from battery_fast_charge.phase7cr2f5_runner import _historical_roles


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path.cwd()
    config = load_phase7cr2f5_config(
        root / "configs/phase7cr2f5_25c_two_stage_guards.yaml"
    )
    data_dir = root / config.section("output")["data_directory"]
    frozen = json.loads(
        (data_dir / "frozen_25c_two_stage_guards.json").read_text(
            encoding="utf-8"
        )
    )["guards"]
    source_path = (
        root
        / "data/phase7cr2f3_temperature_two_stage_guards/combined_validation_trajectories.csv"
    )
    source = pd.read_csv(source_path)
    source_guards = {
        str(temperature): {
            stage: float(
                source[
                    np.isclose(source.ambient_temperature_c, temperature)
                    & (source.guard_stage == stage_name)
                ].guard_v.iloc[0]
            )
            for stage, stage_name in (("boot_v", "boot"), ("running_v", "running"))
        }
        for temperature in (15, 30)
    }
    for temperature in (15, 30):
        expected = frozen[str(temperature)]
        actual = source_guards[str(temperature)]
        if not all(
            np.isclose(actual[key], expected[key], atol=1.0e-15, rtol=0.0)
            for key in ("boot_v", "running_v")
        ):
            raise RuntimeError(f"{temperature} C guard contract changed")
    run_root = data_dir / "runs" / "validation"
    counts = {"15": 0, "30": 0}
    for role, rows in _historical_roles(config, root):
        role_dir = run_root / role
        role_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            temperature = int(row["ambient_temperature_c"])
            if temperature == 25:
                continue
            target_id = row["trajectory_id"]
            source_id = row.get("source_trajectory_id", target_id)
            match = source[source.trajectory_id == source_id].copy()
            if match.empty:
                match = source[source.trajectory_id == target_id].copy()
            if match.trajectory_id.nunique() != 1:
                raise RuntimeError(f"Source trajectory not unique: {source_id}")
            if not np.isclose(match.ambient_temperature_c, temperature).all():
                raise RuntimeError(f"Temperature mismatch: {source_id}")
            match["trajectory_id"] = target_id
            match["role"] = role
            match.to_csv(role_dir / f"{target_id}.csv", index=False)
            counts[str(temperature)] += 1
    if counts != {"15": 72, "30": 176}:
        raise RuntimeError(f"Unexpected cache counts: {counts}")
    manifest = {
        "phase": "Phase 7C-R2F5",
        "purpose": "reuse_exact_unchanged_15c_and_30c_closed_loop_results",
        "source": str(source_path.relative_to(root)).replace("\\", "/"),
        "source_sha256": _sha256(source_path),
        "temperature_guard_contracts": source_guards,
        "reused_trajectory_counts": counts,
        "reused_total": sum(counts.values()),
        "25c_reused": False,
        "new_internal_validation_reused": False,
    }
    (data_dir / "unchanged_temperature_cache_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
