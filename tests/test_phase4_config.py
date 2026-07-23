from pathlib import Path

from battery_fast_charge.phase4_config import load_phase_four_a_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_phase_four_a_uses_small_network_and_grouped_dataset() -> None:
    """锁定五输入、小型隐藏层和阶段3B整轨迹数据。"""
    config = load_phase_four_a_config(PROJECT_ROOT / "configs" / "phase4a.yaml")

    assert config.network.hidden_layer_sizes == (8, 8)
    assert len(config.features) == 5
    assert config.target == "teacher_current_a"
    assert config.teacher_dataset.endswith("teacher_dataset.csv")
    assert config.closed_loop.use_safety_filter is True
