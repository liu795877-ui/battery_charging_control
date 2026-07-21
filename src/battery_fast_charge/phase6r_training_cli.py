"""Train the Phase 6R corrected-policy offline controller comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from .phase6r_config import load_phase_six_r_config
from .phase6r_training import run_phase_six_r_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase6r_corrected_policy_distillation.yaml")
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run_phase_six_r_training(load_phase_six_r_config(args.config), args.project_root)
    print(f"Phase 6R offline status: {result['status']}")
    print(
        "Passing-majority controllers: "
        f"{result['offline_gate']['controllers_with_passing_majority']}"
    )


if __name__ == "__main__":
    main()
