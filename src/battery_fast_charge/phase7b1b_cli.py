"""运行 Phase 7B-1B/1C 电压感知安全层闭环验证。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .phase7b1b_config import load_phase7b1b_config
from .phase7b1b_runner import run_phase7b1b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase7b1b_voltage_safety_layer.yaml"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--stage",
        choices=("regression", "confirmation", "all"),
        default="regression",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_phase7b1b(
        load_phase7b1b_config(args.config),
        args.project_root,
        stage=args.stage,
        resume=args.resume,
    )
    print(f"Phase 7B-1B success: {result['success']}")


if __name__ == "__main__":
    main()
