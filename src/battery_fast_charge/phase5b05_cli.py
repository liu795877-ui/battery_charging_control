"""Run Phase 5B-0.5 representative MPC recovery recheck."""

from __future__ import annotations

import argparse
from pathlib import Path

from .phase5b05_config import load_phase_five_b_zero_five_config
from .phase5b05_runner import run_phase_five_b_zero_five


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase5b05_mpc_recovery.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run_phase_five_b_zero_five(
        load_phase_five_b_zero_five_config(args.config), args.project_root
    )
    print(f"Phase 5B-0.5 status: {result['status']}")
    print(f"Representative gate passed: {result['representative_gate_passed']}")
    print("Phase 5B-1 remains disabled until the full 69-scenario recheck passes.")


if __name__ == "__main__":
    main()
