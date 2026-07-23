from __future__ import annotations

import argparse

from .phase5b06_runner import run_phase_five_b_zero_six


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5B-0.6 paired MPC contract audit")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    result = run_phase_five_b_zero_six(args.project_root)
    print(f"Phase 5B-0.6 status: {result['status']}")
    print(f"Paired runs: {result['paired_controller_runs']}")


if __name__ == "__main__":
    main()
