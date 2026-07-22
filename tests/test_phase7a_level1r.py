from pathlib import Path
import hashlib
import pandas as pd

from battery_fast_charge.phase7a_level1_config import load_phase7a_level1_config
from battery_fast_charge.phase7a_level1r_config import load_phase7a_level1r_config
from battery_fast_charge.phase7a_level1r_runner import design_tail_training_states, design_terminal_states

ROOT = Path(__file__).parents[1]
R_CONFIG = ROOT / "configs/phase7a_level1r_terminal_coverage.yaml"


def test_level1r_changes_only_coverage_contract():
    config = load_phase7a_level1r_config(R_CONFIG)
    base = load_phase7a_level1_config(ROOT / config.source_level1_config)
    assert base.model.nominal_capacity_ah == 5.0
    assert base.mpc.current_bounds_a == (0.0, 10.0)
    assert base.network.hidden_layer_sizes == (32, 32, 16)
    assert base.network.initialization_seeds == (22, 42, 73, 101, 137)
    assert base.gates.offline_nrmse_max == 0.01


def test_terminal_design_has_independent_frozen_split_and_bounds():
    config = load_phase7a_level1r_config(R_CONFIG)
    design = design_terminal_states(config)
    assert len(design) == 140
    assert design.split.value_counts().to_dict() == {"train": 100, "validation": 20, "terminal_test": 20}
    assert design.initial_soc.min() >= 0.74 and design.initial_soc.max() <= 0.799
    assert design.initial_polarization_v.min() >= 0.0 and design.initial_polarization_v.max() <= 0.10
    assert design.groupby("trajectory_id").split.nunique().max() == 1
    tail = design_tail_training_states(config)
    assert len(tail) == 20 and (tail.split == "train").all()
    assert tail.initial_soc.min() >= 0.795 and tail.initial_soc.max() <= 0.799


def test_original_frozen_test_contract_is_untouched():
    config = load_phase7a_level1r_config(R_CONFIG)
    path = ROOT / config.source_level1_dataset
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    original = pd.read_csv(path)
    assert original.query("split == 'test'").trajectory_id.nunique() == 36
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
