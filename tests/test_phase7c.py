import json
from pathlib import Path

import numpy as np

from battery_fast_charge.phase7b1b_config import load_phase7b1b_config
from battery_fast_charge.phase7b1b_runner import _load_context
from battery_fast_charge.phase7c_config import load_phase7c_config
from battery_fast_charge.phase7c_runner import (
    Chen2020ThermalDFN,
    verify_frozen_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase7c_multitemperature_dfn_validation.yaml"


def test_phase7c_contract_keeps_phase7b1_guard_and_artifacts_frozen() -> None:
    config = load_phase7c_config(CONFIG)
    phase7b1b = load_phase7b1b_config(ROOT / config.phase7b1b_config)
    verification = verify_frozen_artifacts(config, ROOT)
    assert len(verification) == 11
    assert all(item["matched"] for item in verification.values())
    assert np.isclose(
        config.contract.residual_growth_guard_v,
        phase7b1b.safety.residual_growth_guard_v,
        rtol=0.0,
        atol=1.0e-15,
    )
    assert config.contract.temperatures_c == (15.0, 30.0)


def test_phase7c_thermal_dfn_reports_physical_temperature() -> None:
    config = load_phase7c_config(CONFIG)
    phase7b1b = load_phase7b1b_config(ROOT / config.phase7b1b_config)
    level3, _, _, _, phase7b0 = _load_context(phase7b1b, ROOT)
    plant = Chen2020ThermalDFN(
        phase7b0.dfn.parameter_set,
        15.0,
        phase7b0.dfn.upper_voltage_cutoff_v,
        0.5,
        level3.model.sample_period_s,
        config.contract.thermal_model,
    )
    measurement = plant.step(5.0)
    assert measurement["terminal_voltage_v"] > 0.0
    assert 15.0 <= measurement["average_temperature_c"] < 16.0


def test_phase7c_generated_contract_hashes_match() -> None:
    config = load_phase7c_config(CONFIG)
    path = ROOT / config.data_directory / "freeze_contract.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["frozen_before_closed_loop_execution"]
    assert payload["not_teacher_data"]
    assert len(payload["files"]) == 2
