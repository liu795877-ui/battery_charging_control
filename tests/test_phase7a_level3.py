from pathlib import Path

import numpy as np

from battery_fast_charge.phase7a_level1_config import load_phase7a_level1_config
from battery_fast_charge.phase7a_level2_config import load_phase7a_level2_config
from battery_fast_charge.phase7a_level3_config import load_phase7a_level3_config
from battery_fast_charge.phase7a_level3_model import Level3MPC, Level3Model, Level3State
from battery_fast_charge.phase7a_level3_runner import _level3_warm_starts, design_initial_states


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "phase7a_level3_slew.yaml"


def _objects():
    config = load_phase7a_level3_config(CONFIG)
    level2 = load_phase7a_level2_config(ROOT / config.source_level2_config)
    inherited = load_phase7a_level1_config(ROOT / level2.source_level1_config)
    return config, level2, inherited, Level3Model(config, inherited, ROOT)


def test_level3_contract_adds_only_previous_current_and_hard_slew():
    config, level2, inherited, _ = _objects()
    assert config.model == type(config.model)(**level2.model.__dict__)
    assert config.constraint.maximum_current_step_a == 2.0
    assert config.constraint.previous_current_bounds_a == (0.0, 10.0)
    assert inherited.network.hidden_layer_sizes == (32, 32, 16)
    assert inherited.network.initialization_seeds == (22, 42, 73, 101, 137)
    raw = CONFIG.read_text(encoding="utf-8").lower()
    assert "temperature" not in raw
    assert "dfn" not in raw
    assert "disturbance" not in raw


def test_level3_state_update_carries_applied_current():
    config, _, _, model = _objects()
    state = Level3State(0.5, 0.02, 0.03, 1.5)
    next_state = model.step(state, 3.5)
    assert next_state.previous_current_a == 3.5
    assert np.isclose(next_state.soc, 0.5 + 3.5 * config.model.sample_period_s / (3600 * 5))


def test_level3_mpc_enforces_initial_and_horizon_slew():
    config, _, inherited, model = _objects()
    state = Level3State(0.65, 0.02, 0.03, 1.0)
    result = Level3MPC(model).solve(state)
    assert result.optimizer_success
    assert result.prediction_feasible
    assert not result.used_fallback
    assert abs(result.current_a - state.previous_current_a) <= 2.0 + inherited.mpc.constraint_tolerance
    assert result.maximum_current_step_a <= 2.0 + inherited.mpc.constraint_tolerance
    assert result.maximum_voltage_v <= 4.2 + inherited.mpc.constraint_tolerance


def test_level3_multistart_contract_uses_fifteen_feasible_starts():
    config, _, _, model = _objects()
    state = Level3State(0.55, 0.01, 0.02, 7.9)
    controller = Level3MPC(model)
    starts = _level3_warm_starts(
        controller, state, 24, config.data.warm_starts_per_state, config.data.random_seed
    )
    assert len(starts) == 15
    for _, warm in starts:
        if warm is not None:
            assert np.min(model.slew_margins(state, warm)) >= -1e-12
        controller.set_warm_start(warm)
        result = controller.solve(state)
        assert result.optimizer_success
        assert result.prediction_feasible


def test_level3_design_freezes_two_test_domains_and_starts_feasible():
    config, _, inherited, model = _objects()
    global_design = design_initial_states(
        config, model, config.data.global_domain, "global", "test", 0
    )
    terminal_design = design_initial_states(
        config, model, config.data.terminal_domain, "terminal", "terminal_test", 1000
    )
    assert global_design.split.value_counts().to_dict() == {
        "train": 168,
        "validation": 36,
        "test": 36,
    }
    assert terminal_design.split.value_counts().to_dict() == {
        "train": 120,
        "validation": 20,
        "terminal_test": 20,
    }
    for frame in (global_design, terminal_design):
        for row in frame.itertuples():
            state = Level3State(
                row.initial_soc,
                row.initial_polarization_1_v,
                row.initial_polarization_2_v,
                row.initial_previous_current_a,
            )
            minimum_current = max(
                inherited.mpc.current_bounds_a[0],
                state.previous_current_a - config.constraint.maximum_current_step_a,
            )
            assert model.terminal_voltage(state, minimum_current) <= 4.195 + 1e-12
