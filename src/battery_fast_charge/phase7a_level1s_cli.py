"""运行 Phase 7A Level 1S 受限训练稳定性消融。"""
from __future__ import annotations
import argparse
from pathlib import Path
from .phase7a_level1s_config import load_phase7a_level1s_config
from .phase7a_level1s_runner import run_phase7a_level1s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase7a_level1s_training_stability.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_phase7a_level1s(load_phase7a_level1s_config(args.config), args.project_root, args.resume)
    print(f"Phase 7A Level 1S success: {result['success']}")


if __name__ == "__main__":
    main()
