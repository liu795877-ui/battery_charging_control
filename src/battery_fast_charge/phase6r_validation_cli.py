"""Run Phase 6R nominal 25 C reduced-model and DFN validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .phase6r_config import load_phase_six_r_config
from .phase6r_validation import run_phase_six_r_validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase6r_corrected_policy_distillation.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run_phase_six_r_validation(load_phase_six_r_config(args.config), args.project_root)
    print(f"Phase 6R nominal status: {result['status']}")
    print(f"Controllers passing both plants: {result['controllers_passing_both_plants_by_majority']}")


if __name__ == "__main__":
    main()
