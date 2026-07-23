from pathlib import Path

from battery_fast_charge.phase4b2_config import load_phase_four_b2_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_phase4b2_config_keeps_twelve_isolated_rollouts() -> None:
    """主动轨迹数必须与8/2/2整轨迹划分一致。"""
    config = load_phase_four_b2_config(PROJECT_ROOT / "configs" / "phase4b2.yaml")

    assert len(config.active_data.rollouts) == 12
    assert (
        config.active_data.train_trajectory_count
        + config.active_data.validation_trajectory_count
        + config.active_data.test_trajectory_count
        == 12
    )
    assert config.active_data.samples_per_edge_soc_bin_per_trajectory > (
        config.active_data.samples_per_middle_soc_bin_per_trajectory
    )
