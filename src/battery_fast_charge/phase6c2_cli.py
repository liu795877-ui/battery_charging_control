"""Command-line entry point for Phase 6C-2 targeted teacher data."""

from __future__ import annotations

import argparse
from pathlib import Path

from .phase6c2_config import load_phase_six_c2_config
from .phase6c2_runner import run_phase_six_c2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase6c2_targeted_teacher_data.yaml")
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run_phase_six_c2(load_phase_six_c2_config(args.config), args.project_root)
    teacher = result["teacher_data"]
    print(f"Phase 6C-2 status: {result['status']}")
    print(f"Accepted trajectories: {teacher['accepted_trajectory_count']}")
    print(f"Unfolded samples: {teacher['unfolded_sample_count']}")


if __name__ == "__main__":
    main()
