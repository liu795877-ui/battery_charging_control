import json
from pathlib import Path

import pandas as pd

from battery_fast_charge.active_learning import (
    generate_ann_centered_rollout,
    sample_active_states,
)
from battery_fast_charge.ann_model import TinyANN
from battery_fast_charge.identification import build_ocv_function
from battery_fast_charge.mpc import ReducedBatteryModel
from battery_fast_charge.phase3_config import load_phase_three_config
from battery_fast_charge.phase4b2_config import load_phase_four_b2_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _setup():
    phase3 = load_phase_three_config(PROJECT_ROOT / "configs" / "phase3.yaml")
    config = load_phase_four_b2_config(PROJECT_ROOT / "configs" / "phase4b2.yaml")
    parameters = json.loads(
        (PROJECT_ROOT / phase3.artifacts.identified_parameters).read_text(
            encoding="utf-8"
        )
    )
    ocv = pd.read_csv(PROJECT_ROOT / phase3.artifacts.ocv_curve)
    model = ReducedBatteryModel(phase3, build_ocv_function(ocv), parameters)
    ann = TinyANN.load(PROJECT_ROOT / config.seed_ann_model)
    return model, ann, phase3, config


def test_ann_centered_rollout_is_reachable_and_respects_current_change() -> None:
    """主动探索只能通过逐步推进产生，且仍服从2 A/步变化约束。"""
    model, ann, phase3, config = _setup()
    frame = generate_ann_centered_rollout(
        ann, model, phase3, config.active_data.rollouts[0]
    )

    current_change = frame["exploration_applied_current_a"].diff().abs().fillna(
        frame["exploration_applied_current_a"].iloc[0]
    )
    assert frame["reachable_source"].all()
    assert frame["state_soc"].is_monotonic_increasing
    assert current_change.max() <= (
        phase3.constraints.maximum_current_change_a_per_step + 1.0e-8
    )


def test_edge_soc_bins_receive_more_active_samples() -> None:
    """主动采样明确加强10%–20%和70%–80%两个问题区间。"""
    model, ann, phase3, config = _setup()
    frames = []
    for rollout in config.active_data.rollouts:
        frames.append(generate_ann_centered_rollout(ann, model, phase3, rollout))
    sampled = sample_active_states(pd.concat(frames, ignore_index=True), config)
    counts = sampled.groupby("soc_bin_index").size()

    assert counts.loc[0] > counts.loc[1]
    assert counts.loc[6] > counts.loc[5]
    assert sampled.groupby("trajectory_id")["split"].nunique().max() == 1
