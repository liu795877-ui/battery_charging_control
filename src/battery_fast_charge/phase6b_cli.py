"""Command line entry point for Phase 6B DNN failure diagnosis."""

from __future__ import annotations

import argparse
from pathlib import Path

from .phase6b_config import load_phase_six_b_config
from .phase6b_runner import run_phase_six_b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase6b_dnn_failure_diagnosis.yaml"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run_phase_six_b(load_phase_six_b_config(args.config), args.project_root)
    print(f"Phase 6B status: {result['status']}")
    print(f"Dataset gate: {result['paper_dataset']['success']}")
    if result["status"] == "completed":
        pure = result["nominal_25c"]["pure_dnn"]["comparison"]
        projected = result["nominal_25c"]["projected_dnn"]["comparison"]
        print(f"Pure DNN current NRMSE: {100 * pure['current_nrmse']:.3f}%")
        print(f"Projected DNN current NRMSE: {100 * projected['current_nrmse']:.3f}%")


if __name__ == "__main__":
    main()
