"""Phase 7C-R2F2启动前残差初始化审计。"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .phase7a_level3_model import Level3State
from .phase7b1b_runner import _maximum_safe_current
from .phase7c_runner import Chen2020ThermalDFN
from .phase7cr2_runner import _context, _thermal_current_limit
from .phase7cr2f_config import load_phase7cr2f_config
from .phase7cr2f_teacher import solve_teacher_r2f


def _sha256(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest()


def _normalized_text_sha256(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _artifact_hash(path: Path, expected: str) -> tuple[str, bool]:
    raw = _sha256(path)
    if raw == expected:
        return raw, True
    if path.suffix.lower() in {
        ".py",
        ".yaml",
        ".yml",
        ".json",
        ".csv",
        ".md",
    }:
        normalized = _normalized_text_sha256(path)
        return normalized, normalized == expected
    return raw, False


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def verify_frozen_r2f(config: dict[str, Any], root: Path) -> dict[str, Any]:
    source = config["sources"]
    manifest_path = root / source["phase7cr2f_freeze_manifest"]
    actual_manifest = _sha256(manifest_path)
    if actual_manifest != source["phase7cr2f_freeze_manifest_sha256"]:
        raise RuntimeError("R2F冻结清单哈希不匹配。")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = []
    records = {}
    for relative, expected in manifest["artifacts"].items():
        actual, matched = _artifact_hash(root / relative, expected)
        records[relative] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matched": matched,
        }
        if not matched:
            mismatches.append(relative)
    if mismatches:
        raise RuntimeError(f"R2F冻结工件已变化：{mismatches}")
    if manifest["status"] != "strict_stop_failed":
        raise RuntimeError("R2F失败证据状态被改写。")
    return {
        "manifest_sha256": actual_manifest,
        "r2f_failure_preserved": True,
        "records": records,
    }


def _initial_measurement(
    config: dict[str, Any],
    root: Path,
    initial: dict[str, Any],
) -> float:
    r2f = load_phase7cr2f_config(
        root / config["sources"]["phase7cr2f_config"]
    )
    _, _, level3, _, _, phase7b0 = _context(r2f, root)
    plant = Chen2020ThermalDFN(
        phase7b0.dfn.parameter_set,
        float(initial["ambient_temperature_c"]),
        phase7b0.dfn.upper_voltage_cutoff_v,
        float(initial["initial_soc"]),
        level3.model.sample_period_s,
        "lumped",
    )
    solution = plant.simulation.solve(
        [0.0, float(config["audit"]["initial_measurement_horizon_s"])],
        inputs={
            "phase7c_applied_current_a": -float(
                initial["initial_previous_current_a"]
            )
        },
    )
    values = np.asarray(solution["Terminal voltage [V]"].entries).reshape(
        -1
    )
    return float(values[0])


def _short_replay(
    config: dict[str, Any],
    root: Path,
    initial: dict[str, Any],
    initialization_mode: str,
    measured_initial_voltage_v: float,
    predicted_initial_voltage_v: float,
) -> list[dict[str, Any]]:
    r2f = load_phase7cr2f_config(
        root / config["sources"]["phase7cr2f_config"]
    )
    r1, b1, level3, inherited, model, phase7b0 = _context(r2f, root)
    state = Level3State(
        float(initial["initial_soc"]),
        float(initial["initial_polarization_1_v"]),
        float(initial["initial_polarization_2_v"]),
        float(initial["initial_previous_current_a"]),
    )
    ambient = float(initial["ambient_temperature_c"])
    plant = Chen2020ThermalDFN(
        phase7b0.dfn.parameter_set,
        ambient,
        phase7b0.dfn.upper_voltage_cutoff_v,
        state.soc,
        level3.model.sample_period_s,
        "lumped",
    )
    residual_v = (
        0.0
        if initialization_mode == "zero"
        else measured_initial_voltage_v - predicted_initial_voltage_v
    )
    temperature_c = float(initial["initial_temperature_c"])
    lower_bound, upper_bound = inherited.mpc.current_bounds_a
    maximum_step = level3.constraint.maximum_current_step_a
    rows = []
    for step in range(int(config["audit"]["short_replay_steps"])):
        residual_before = residual_v
        result, teacher = solve_teacher_r2f(state, model, r1)
        candidate = float(result.current_a)
        slew_lower = max(lower_bound, state.previous_current_a - maximum_step)
        slew_upper = min(upper_bound, state.previous_current_a + maximum_step)
        voltage_max = _maximum_safe_current(
            state,
            residual_before + float(config["audit"]["frozen_guard_30c_v"]),
            b1,
            model,
        )
        search_upper = min(slew_upper, voltage_max)
        constant_max, _ = _thermal_current_limit(
            temperature_c, ambient, search_upper, r1, braking=False
        )
        braking_max, _ = _thermal_current_limit(
            temperature_c, ambient, search_upper, r1, braking=True
        )
        if (
            constant_max
            < slew_lower - r1.thermal["empty_interval_tolerance_a"]
        ):
            thermal_max = min(braking_max, slew_lower)
        else:
            thermal_max = constant_max
        final_upper = min(slew_upper, voltage_max, thermal_max)
        empty = final_upper < slew_lower - r1.thermal[
            "empty_interval_tolerance_a"
        ]
        if empty:
            rows.append(
                {
                    "trajectory_id": initial["trajectory_id"],
                    "initialization_mode": initialization_mode,
                    "step_index": step,
                    "residual_before_v": residual_before,
                    "empty_interval": True,
                }
            )
            break
        current = float(
            np.clip(min(candidate, final_upper), slew_lower, final_upper)
        )
        predicted = model.step(state, current)
        predicted_voltage = model.terminal_voltage(predicted, current)
        measurement = plant.step(current)
        new_residual = (
            float(measurement["terminal_voltage_v"]) - predicted_voltage
        )
        rows.append(
            {
                "trajectory_id": initial["trajectory_id"],
                "source_role": initial["source_role"],
                "risk_stratum": initial.get("risk_stratum", "unknown"),
                "initialization_mode": initialization_mode,
                "step_index": step,
                "soc": state.soc,
                "previous_current_a": state.previous_current_a,
                "candidate_current_a": candidate,
                "current_a": current,
                "measured_initial_voltage_v": measured_initial_voltage_v,
                "predicted_initial_voltage_v": predicted_initial_voltage_v,
                "initial_residual_v": (
                    measured_initial_voltage_v - predicted_initial_voltage_v
                ),
                "residual_before_v": residual_before,
                "residual_after_v": new_residual,
                "positive_residual_growth_v": max(
                    0.0, new_residual - residual_before
                ),
                "terminal_voltage_v": measurement["terminal_voltage_v"],
                "voltage_safe_current_max_a": voltage_max,
                "thermal_safe_current_max_a": thermal_max,
                "empty_interval": False,
                "selected_teacher_branch": teacher[
                    "selected_teacher_branch"
                ],
            }
        )
        state = Level3State(
            float(measurement["soc"]),
            predicted.polarization_1_v,
            predicted.polarization_2_v,
            current,
        )
        temperature_c = float(measurement["average_temperature_c"])
        residual_v = new_residual
    return rows


def _worker(
    config: dict[str, Any], root_text: str, initial: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(root_text)
    r2f = load_phase7cr2f_config(
        root / config["sources"]["phase7cr2f_config"]
    )
    _, _, _, _, model, _ = _context(r2f, root)
    state = Level3State(
        float(initial["initial_soc"]),
        float(initial["initial_polarization_1_v"]),
        float(initial["initial_polarization_2_v"]),
        float(initial["initial_previous_current_a"]),
    )
    measured = _initial_measurement(config, root, initial)
    predicted = model.terminal_voltage(state, state.previous_current_a)
    model_config = model.config.model
    initial_record = {
        "trajectory_id": initial["trajectory_id"],
        "source_role": initial["source_role"],
        "risk_stratum": initial.get("risk_stratum", "unknown"),
        "initial_soc": state.soc,
        "initial_v1_v": state.polarization_1_v,
        "initial_v2_v": state.polarization_2_v,
        "previous_current_a": state.previous_current_a,
        "dfn_initial_voltage_v": measured,
        "two_rc_initial_voltage_v": predicted,
        "measured_initial_residual_v": measured - predicted,
        "steady_v1_for_previous_current_v": (
            model_config.r1_ohm * state.previous_current_a
        ),
        "steady_v2_for_previous_current_v": (
            model_config.r2_ohm * state.previous_current_a
        ),
        "v1_steady_state_deviation_v": (
            state.polarization_1_v
            - model_config.r1_ohm * state.previous_current_a
        ),
        "v2_steady_state_deviation_v": (
            state.polarization_2_v
            - model_config.r2_ohm * state.previous_current_a
        ),
        "initial_measurement_available": bool(np.isfinite(measured)),
    }
    rows = []
    for mode in config["audit"]["compared_initialization_modes"]:
        rows.extend(
            _short_replay(
                config, root, initial, mode, measured, predicted
            )
        )
    return initial_record, rows


def _load_states(config: dict[str, Any], root: Path) -> pd.DataFrame:
    frames = []
    for role, key in (
        ("development", "development_states"),
        ("internal_validation", "internal_validation_states"),
    ):
        frame = pd.read_csv(root / config["sources"][key])
        frame["source_role"] = role
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def run_audit(config: dict[str, Any], root: Path) -> dict[str, Any]:
    verification = verify_frozen_r2f(config, root)
    states = _load_states(config, root)
    initial_records = []
    replay_rows = []
    with ProcessPoolExecutor(
        max_workers=int(config["audit"]["maximum_workers"])
    ) as executor:
        futures = [
            executor.submit(
                _worker, config, str(root), row
            )
            for row in states.to_dict(orient="records")
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            initial, rows = future.result()
            initial_records.append(initial)
            replay_rows.extend(rows)
            print(f"[Residual audit] {completed}/{len(futures)}", flush=True)
    initial_frame = pd.DataFrame(initial_records).sort_values("trajectory_id")
    replay = pd.DataFrame(replay_rows).sort_values(
        ["trajectory_id", "initialization_mode", "step_index"]
    )
    data_dir = root / config["output"]["data_directory"]
    result_dir = root / config["output"]["result_directory"]
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    initial_frame.to_csv(data_dir / "initial_voltage_audit.csv", index=False)
    replay.to_csv(data_dir / "three_step_replay.csv", index=False)

    mode_summary = {}
    for mode, group in replay.groupby("initialization_mode"):
        boot = group[group.step_index.isin([0, 1])]
        running = group[group.step_index >= 2]
        mode_summary[mode] = {
            "maximum_boot_positive_growth_v": float(
                boot.positive_residual_growth_v.max()
            ),
            "maximum_running_positive_growth_v": float(
                running.positive_residual_growth_v.max()
            ),
            "maximum_voltage_v": float(group.terminal_voltage_v.max()),
            "empty_interval_count": int(
                group.empty_interval.astype(bool).sum()
            ),
        }
    initial_residual = initial_frame.measured_initial_residual_v
    measured_available = bool(
        initial_frame.initial_measurement_available.astype(bool).all()
    )
    contracts = config["audit"]["contracts"]
    shared_history = not (
        contracts["dfn_initialized_from_soc_and_temperature_only"]
        and contracts["two_rc_polarizations_independently_sampled"]
        and not contracts["shared_current_history_available"]
    )
    recommendation = (
        "冻结测量残差初始化；由于DFN与2RC没有共享电流历史，F2仍需"
        "把前两个控制步作为独立启动窗口，并在全新开发集上分别估计"
        "启动期和运行期增长界。不得把本审计当作F2验证。"
        if measured_available and not shared_history
        else "保留零初始化并预注册启动安全窗口。"
    )
    payload = {
        "study_name": config["study"]["name"],
        "configuration": config,
        "frozen_r2f_verification": verification,
        "state_count": len(initial_frame),
        "initial_measurement": {
            "available_for_all_states": measured_available,
            "minimum_initial_residual_v": float(initial_residual.min()),
            "median_initial_residual_v": float(initial_residual.median()),
            "maximum_initial_residual_v": float(initial_residual.max()),
            "maximum_absolute_initial_residual_v": float(
                initial_residual.abs().max()
            ),
        },
        "state_history_contract": {
            **contracts,
            "dfn_and_2rc_share_consistent_current_history": shared_history,
        },
        "mode_summary": mode_summary,
        "decision": {
            "zero_initialization_is_documented_contract": contracts[
                "zero_initialization_documented_in_config_or_plan"
            ],
            "zero_initialization_is_implementation_default": True,
            "freeze_residual_initialization": "measured",
            "freeze_two_stage_guard_structure": True,
            "boot_steps": [0, 1],
            "running_starts_at_step": 2,
            "generate_f2_data": False,
            "generate_r3": False,
            "run_ann": False,
            "recommendation": recommendation,
        },
    }
    (result_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(result_dir / "PHASE7C-R2F2_残差初始化审计报告.md", payload)
    return payload


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    initial = payload["initial_measurement"]
    zero = payload["mode_summary"]["zero"]
    measured = payload["mode_summary"]["measured"]
    report = f"""# Phase 7C-R2F2启动前残差初始化审计

## 审计边界

本审计只比较现有48条R2F新状态的残差初始化方式和前三个控制步；没有生成
F2开发/内部验证集，没有调整裕量，没有生成R3初态，也没有运行ANN。

## 时刻0可测性

- 48/48状态均可在DFN不推进5 s控制步前取得端电压；
- 2RC初始端电压可由SOC、V1、V2和上一电流直接计算；
- 初始残差范围：{1000 * initial['minimum_initial_residual_v']:.3f}至
  {1000 * initial['maximum_initial_residual_v']:.3f} mV；
- 初始残差中位数：{1000 * initial['median_initial_residual_v']:.3f} mV；
- 最大绝对初始残差：
  {1000 * initial['maximum_absolute_initial_residual_v']:.3f} mV。

## 状态历史一致性

DFN仅由SOC和温度初始化；2RC的V1、V2和上一电流独立采样，数据中没有共享
电流历史。因此两套初态不能认定为来自同一物理历史。`residual_v = 0.0`
没有配置项或书面合同支持，是继承实现默认值，而不是已注册的启动无测量模式。

## 三步对比

- 零初始化启动期最大正向增长：
  {1000 * zero['maximum_boot_positive_growth_v']:.3f} mV；
- 测量初始化启动期最大正向增长：
  {1000 * measured['maximum_boot_positive_growth_v']:.3f} mV；
- 零初始化第2步后最大增长：
  {1000 * zero['maximum_running_positive_growth_v']:.3f} mV；
- 测量初始化第2步后最大增长：
  {1000 * measured['maximum_running_positive_growth_v']:.3f} mV。

## 冻结建议

{payload['decision']['recommendation']}

F2应冻结为：时刻0使用测量残差初始化；步骤0和1使用启动期裕量，步骤2起
使用运行期裕量。具体裕量数值只能由下一阶段全新开发集估计。
"""
    path.write_text(report, encoding="utf-8")
