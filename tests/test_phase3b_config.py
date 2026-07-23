from pathlib import Path

from battery_fast_charge.phase3b_config import load_phase_three_b_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_phase_three_b_dataset_protocol() -> None:
    """锁定12条轨迹、8/2/2划分和10% SOC分箱协议。"""
    config = load_phase_three_b_config(PROJECT_ROOT / "configs" / "phase3b.yaml")

    assert len(config.dataset.rollouts) == 12
    assert config.dataset.train_trajectory_count == 8
    assert config.dataset.validation_trajectory_count == 2
    assert config.dataset.test_trajectory_count == 2
    assert config.dataset.soc_bin_edges[0] == 0.10
    assert config.dataset.soc_bin_edges[-1] == 0.80
    assert len({rollout.name for rollout in config.dataset.rollouts}) == 12
