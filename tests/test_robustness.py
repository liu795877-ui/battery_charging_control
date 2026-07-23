import json
from pathlib import Path

import numpy as np

from battery_fast_charge.phase5a_config import load_phase_five_a_config
from battery_fast_charge.robustness import (
    SCENARIO_COLUMNS,
    generate_reduced_stress_scenarios,
    perturb_identified_parameters,
)


def test_stress_scenarios_are_reproducible_and_inside_declared_ranges() -> None:
    """名义、定向角点和 LHS 样本数目固定，随机样本不越出声明范围。"""
    config = load_phase_five_a_config("configs/phase5a.yaml")
    first = generate_reduced_stress_scenarios(config)
    second = generate_reduced_stress_scenarios(config)

    assert len(first) == 1 + 4 + config.reduced_stress_test.random_scenario_count
    assert first.equals(second)
    assert first["scenario_id"].is_unique
    nominal = first.iloc[0]
    assert nominal["scenario_id"] == "nominal"
    assert nominal["noise_scale"] == 0.0

    random_rows = first[first["scenario_kind"] == "latin_hypercube"]
    stress = config.reduced_stress_test
    ranges = {
        "initial_soc": stress.initial_soc_range,
        "ambient_temperature_c": stress.ambient_temperature_c_range,
        "capacity_multiplier": stress.capacity_multiplier_range,
        "resistance_multiplier": stress.resistance_multiplier_range,
        "time_constant_multiplier": stress.time_constant_multiplier_range,
        "heat_capacity_multiplier": stress.heat_capacity_multiplier_range,
        "thermal_resistance_multiplier": stress.thermal_resistance_multiplier_range,
        "heat_gain_multiplier": stress.heat_gain_multiplier_range,
        "soc_bias": stress.soc_bias_range,
        "temperature_bias_c": stress.temperature_bias_c_range,
        "polarization_fast_bias_v": stress.polarization_bias_v_range,
        "polarization_slow_bias_v": stress.polarization_bias_v_range,
    }
    assert set(ranges) == set(SCENARIO_COLUMNS)
    for column, (low, high) in ranges.items():
        assert random_rows[column].between(low, high, inclusive="both").all()


def test_parameter_perturbation_preserves_rc_time_constants() -> None:
    """电阻与时间常数扰动后，电容仍满足 C=tau/R。"""
    nominal = json.loads(
        Path("outputs/metrics/phase2_identified_parameters.json").read_text(
            encoding="utf-8"
        )
    )
    config = load_phase_five_a_config("configs/phase5a.yaml")
    scenario = generate_reduced_stress_scenarios(config).iloc[1]
    perturbed = perturb_identified_parameters(nominal, scenario)
    electrical = perturbed["electrical_2rc"]

    assert np.isclose(
        electrical["c1_f"], electrical["tau1_s"] / electrical["r1_ohm"]
    )
    assert np.isclose(
        electrical["c2_f"], electrical["tau2_s"] / electrical["r2_ohm"]
    )
