"""Generate the corrected rolling first-action teacher dataset for Phase 6R."""

from __future__ import annotations

import argparse
from pathlib import Path

from .phase6r_config import load_phase_six_r_config
from .phase6r_teacher import run_phase_six_r_teacher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase6r_corrected_policy_distillation.yaml")
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run_phase_six_r_teacher(load_phase_six_r_config(args.config), args.project_root)
    print(f"Phase 6R teacher status: {result['status']}")
    print(f"Consistency passed: {result['teacher_consistency']['success']}")
    print(f"Accepted trajectories: {result['teacher_dataset']['accepted_trajectory_count']}")


if __name__ == "__main__":
    main()
