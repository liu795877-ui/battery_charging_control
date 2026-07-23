from pathlib import Path
import numpy as np

from battery_fast_charge.phase7a_level1_config import load_phase7a_level1_config
from battery_fast_charge.phase7a_level1_model import Level1MPC, Level1Model, Level1State
from battery_fast_charge.phase7a_level1_runner import design_initial_states


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "phase7a_level1_1rc.yaml"


def test_level1_contract_and_trajectory_split_are_frozen():
    config = load_phase7a_level1_config(CONFIG)
    design = design_initial_states(config)
    assert len(design) == 240
    assert design.split.value_counts().to_dict() == {"train": 168, "validation": 36, "test": 36}
    assert design.groupby("trajectory_id").split.nunique().max() == 1
    assert len(config.network.initialization_seeds) == 5


def test_1rc_equations_and_voltage_sign():
    config = load_phase7a_level1_config(CONFIG)
    model = Level1Model(config, ROOT)
    state = Level1State(0.5, 0.02)
    next_state = model.step(state, 5.0)
    expected_soc = 0.5 + 5.0 * 5.0 / (3600.0 * 5.0)
    expected_vp = np.exp(-5.0 / config.model.tau1_s) * 0.02 + config.model.r1_ohm * (1 - np.exp(-5.0 / config.model.tau1_s)) * 5.0
    assert np.isclose(next_state.soc, expected_soc)
    assert np.isclose(next_state.polarization_v, expected_vp)
    assert model.terminal_voltage(state, 5.0) > model.terminal_voltage(state, 0.0)


def test_mpc_enforces_only_current_and_terminal_voltage():
    config = load_phase7a_level1_config(CONFIG)
    model = Level1Model(config, ROOT)
    result = Level1MPC(model).solve(Level1State(0.70, 0.04))
    assert result.optimizer_success
    assert result.prediction_feasible
    assert not result.used_fallback
    assert np.all(result.plan_a >= config.mpc.current_bounds_a[0])
    assert np.all(result.plan_a <= config.mpc.current_bounds_a[1])
    assert result.maximum_voltage_v <= config.mpc.terminal_voltage_max_v + config.mpc.constraint_tolerance


def test_level1_config_references_project_identification():
    config = load_phase7a_level1_config(CONFIG)
    import json
    identified = json.loads((ROOT / "outputs/metrics/phase2_identified_parameters.json").read_text(encoding="utf-8"))["electrical_2rc"]
    assert config.model.r0_ohm == identified["r0_ohm"]
    assert config.model.r1_ohm == identified["r1_ohm"]
    assert config.model.tau1_s == identified["tau1_s"]
