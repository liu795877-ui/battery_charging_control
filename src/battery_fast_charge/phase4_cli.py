"""阶段4A命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .phase4_config import load_phase_four_a_config
from .phase4_runner import run_phase_four_a


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase4a.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    config = load_phase_four_a_config(args.config)
    result = run_phase_four_a(config, args.project_root)
    print(result["comparison"].to_string(index=False))
    print(f"Phase 4A success: {result['metrics']['success']}")


if __name__ == "__main__":
    main()
