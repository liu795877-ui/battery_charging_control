"""Run Phase 5B-0 nominal/oracle MPC feasibility-envelope experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .phase5b0_config import load_phase_five_b_zero_config
from .phase5b0_runner import run_phase_five_b_zero


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase5b0_mpc_feasibility_envelope.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run_phase_five_b_zero(load_phase_five_b_zero_config(args.config), args.project_root, args.config)
    print(f"Phase 5B-0 status: {result['status']}")
    print(f"Teacher-feasible mask count: {result['teacher_feasible_mask_count']}")


if __name__ == "__main__":
    main()
