"""运行 Phase 7A Level 3 硬斜率约束验证。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .phase7a_level3_config import load_phase7a_level3_config
from .phase7a_level3_runner import run_phase7a_level3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase7a_level3_slew.yaml"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_phase7a_level3(
        load_phase7a_level3_config(args.config),
        args.project_root,
        args.resume,
    )
    print(f"Phase 7A Level 3 success: {result['success']}")


if __name__ == "__main__":
    main()
