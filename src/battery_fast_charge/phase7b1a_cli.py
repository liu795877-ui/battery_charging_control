"""运行 Phase 7B-1A 电压残差与制动可行性审计。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .phase7b1a_config import load_phase7b1a_config
from .phase7b1a_runner import run_phase7b1a


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase7b1a_voltage_mismatch_audit.yaml"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run_phase7b1a(
        load_phase7b1a_config(args.config), args.project_root
    )
    print(f"Phase 7B-1A success: {result['success']}")


if __name__ == "__main__":
    main()
