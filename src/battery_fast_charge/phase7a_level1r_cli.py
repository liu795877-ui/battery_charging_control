"""运行 Phase 7A Level 1R 末端覆盖修复。"""
from __future__ import annotations
import argparse
from pathlib import Path
from .phase7a_level1r_config import load_phase7a_level1r_config
from .phase7a_level1r_runner import run_phase7a_level1r


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase7a_level1r_terminal_coverage.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_phase7a_level1r(load_phase7a_level1r_config(args.config), args.project_root, args.resume)
    print(f"Phase 7A Level 1R success: {result['success']}")


if __name__ == "__main__":
    main()
