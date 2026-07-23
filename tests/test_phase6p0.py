from __future__ import annotations

from pathlib import Path

import numpy as np

from battery_fast_charge.phase6p0_config import load_phase_six_p_zero_config
from battery_fast_charge.phase6p0_dnn import (
    _forward_and_jacobian,
    _initialize,
)
from battery_fast_charge.phase6p0_ndc import (
    NDCMPC,
    NDCModel,
    generate_frozen_test_initial_states,
    generate_training_initial_states,
)


CONFIG = Path("configs/phase6p0_ndc_paper.yaml")


def test_paper_contract_is_frozen() -> None:
    config = load_phase_six_p_zero_config(CONFIG)
    assert config.mpc.sample_period_s == 60.0
    assert (config.mpc.prediction_horizon, config.mpc.control_horizon, config.mpc.constraint_horizon) == (10, 2, 1)
    assert config.data.hammersley_count + config.data.factorial_count == 400
    assert config.data.training_trajectory_steps == 5
    assert config.data.independent_test_trajectories == 30
    assert config.dnn.hidden_layer_sizes == (7, 5, 3)


def test_ndc_exact_discretization_conserves_charge_without_current() -> None:
    model = NDCModel(load_phase_six_p_zero_config(CONFIG))
    state = np.asarray([0.35, 0.72])
    next_state = model.step(state, 0.0)
    assert np.isclose(model.soc(next_state), model.soc(state), atol=1.0e-12)
    assert abs(next_state[1] - next_state[0]) < abs(state[1] - state[0])


def test_hybrid_design_and_frozen_test_are_feasible() -> None:
    model = NDCModel(load_phase_six_p_zero_config(CONFIG))
    design = generate_training_initial_states(model)
    test = generate_frozen_test_initial_states(model)
    assert len(design) == 400
    assert design["sampling_method"].value_counts().to_dict() == {"hammersley": 324, "boundary_factorial": 76}
    assert (design["initial_health_margin_v"] >= -1.0e-12).all()
    assert len(test) == 30
    assert (test["initial_health_margin_v"] >= -1.0e-12).all()


def test_mpc_has_no_hard_slew_constraint_and_returns_feasible_action() -> None:
    config = load_phase_six_p_zero_config(CONFIG)
    model = NDCModel(config)
    controller = NDCMPC(model)
    result = controller.solve(np.asarray([0.2, 0.2]), previous_current_a=1.0)
    assert result.feasible
    assert config.mpc.current_bounds_a[0] <= result.current_a <= config.mpc.current_bounds_a[1]
    assert len(result.planned_currents_a) == config.mpc.prediction_horizon
    assert result.current_a - 1.0 > 1.0  # 论文允许首步从 1 A 直接跃迁到接近 3 A。


def test_analytic_network_jacobian_matches_finite_difference() -> None:
    layer_sizes = (2, 7, 5, 3, 1)
    parameters = _initialize(layer_sizes, seed=7)
    x = np.asarray([[-0.8, 0.3], [0.1, -0.2], [0.7, 0.9]])
    prediction, jacobian = _forward_and_jacobian(x, parameters, layer_sizes, need_jacobian=True)
    assert jacobian is not None
    numerical = np.empty_like(jacobian)
    epsilon = 1.0e-6
    for index in range(len(parameters)):
        shifted = parameters.copy()
        shifted[index] += epsilon
        plus, _ = _forward_and_jacobian(x, shifted, layer_sizes, need_jacobian=False)
        shifted[index] -= 2.0 * epsilon
        minus, _ = _forward_and_jacobian(x, shifted, layer_sizes, need_jacobian=False)
        numerical[:, index] = (plus - minus) / (2.0 * epsilon)
    assert np.allclose(prediction.shape, (3,))
    assert np.allclose(jacobian, numerical, atol=2.0e-6, rtol=2.0e-5)
