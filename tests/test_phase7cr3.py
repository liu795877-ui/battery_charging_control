from pathlib import Path

from battery_fast_charge.phase7cr3_config import load_phase7cr3_config
from battery_fast_charge.phase7cr3_runner import verify_r2f5


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_phase7cr3_config(ROOT / "configs/phase7cr3_independent_confirmation.yaml")


def test_r3_preregistration_requires_safe_mpc_before_ann() -> None:
    contract = CONFIG.section("control_contract")
    assert contract["safe_mpc_must_pass_before_ann"] is True
    assert contract["ann_retraining_authorized"] is False
    assert contract["level4_entry_authorized"] is False
    assert CONFIG.section("datasets")["confirmation_count_per_temperature"] == 24


def test_r2f5_strict_pass_is_frozen() -> None:
    result = verify_r2f5(CONFIG, ROOT)
    assert result["status"] == "strict_passed"
