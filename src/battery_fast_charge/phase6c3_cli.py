"""Command-line entry point for Phase 6C-3 controller comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from .phase6c3_config import load_phase_six_c3_config
from .phase6c3_runner import run_phase_six_c3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase6c3_structured_dnn_comparison.yaml")
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run_phase_six_c3(load_phase_six_c3_config(args.config), args.project_root)
    gate = result["phase6c_acceptance"]
    print(f"Phase 6C-3 status: {result['status']}")
    print(f"Pure paper method passed: {gate['pure_paper_method_passed']}")
    print(f"Proceed to Phase 6D: {gate['proceed_to_phase6d']}")


if __name__ == "__main__":
    main()
