"""Phase 7A Level 1 项目参数 1RC pure DNN 消融入口。"""

from __future__ import annotations
import argparse
from pathlib import Path
from .phase7a_level1_config import load_phase7a_level1_config
from .phase7a_level1_runner import run_phase7a_level1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase7a_level1_1rc.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_phase7a_level1(load_phase7a_level1_config(args.config), args.project_root, args.resume)
    print(f"Phase 7A Level 1 success: {result['success']}")


if __name__ == "__main__":
    main()
