import pandas as pd
import pytest

from battery_fast_charge.phase6c1_config import load_phase_six_c1_config
from battery_fast_charge.phase6c1_runner import _interpret, audit_frozen_dataset


def test_phase6c1_declares_complete_five_seed_matrix() -> None:
    config = load_phase_six_c1_config(
        "configs/phase6c1_optimizer_generalization_ablation.yaml"
    )

    assert len(config.network.hidden_layer_sizes) == 3
    assert len(config.network.initialization_seeds) >= 5
    assert set(config.optimizers.methods) == {"lbfgs", "adam", "adam_lbfgs"}
    assert config.baseline.expected_sample_count == 7024


def test_phase6c1_frozen_dataset_audit_rejects_split_mutation(tmp_path) -> None:
    config = load_phase_six_c1_config(
        "configs/phase6c1_optimizer_generalization_ablation.yaml"
    )
    source = pd.read_csv(config.baseline.dataset)
    source.loc[0, "split"] = "test"
    mutated = tmp_path / "mutated.csv"
    source.to_csv(mutated, index=False)

    with pytest.raises(RuntimeError, match="Frozen Phase 6B dataset audit failed"):
        audit_frozen_dataset(source, mutated, config)


def test_phase6c1_interpretation_uses_all_groups_for_generalization() -> None:
    config = load_phase_six_c1_config(
        "configs/phase6c1_optimizer_generalization_ablation.yaml"
    )
    summary = pd.DataFrame(
        {
            "architecture": ["16-16", "64-64-32"],
            "optimizer": ["adam", "lbfgs"],
            "best_validation_nrmse": [0.04, 0.08],
            "train_nrmse_mean": [0.051, 0.02],
            "validation_nrmse_mean": [0.04, 0.08],
            "test_nrmse_mean": [0.055, 0.14],
            "test_nrmse_std": [0.001, 0.02],
            "test_nrmse_range": [0.002, 0.04],
        }
    )

    result = _interpret(summary, config)

    assert result["primary_diagnosis"] == "generalization_or_coverage_limited"
    assert result["generalization_limited_group_count"] == 1
