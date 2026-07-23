"""Command-line entry point for Phase 6C-1."""

from __future__ import annotations

import argparse
from pathlib import Path

from .phase6c1_config import load_phase_six_c1_config
from .phase6c1_runner import run_phase_six_c1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase6c1_optimizer_generalization_ablation.yaml"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run_phase_six_c1(load_phase_six_c1_config(args.config), args.project_root)
    diagnosis = result["interpretation"]
    print(f"Phase 6C-1 status: {result['status']}")
    print(f"Primary diagnosis: {diagnosis['primary_diagnosis']}")
    print(f"Stop pure network scaling: {diagnosis['stop_pure_network_scaling']}")


if __name__ == "__main__":
    main()
