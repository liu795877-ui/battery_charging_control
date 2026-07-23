"""阶段5A有界鲁棒性验证命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .phase5a_config import load_phase_five_a_config
from .phase5a_runner import run_phase_five_a


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase5a.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    config = load_phase_five_a_config(args.config)
    result = run_phase_five_a(config, args.project_root)
    print(result["dfn_summary"].to_string(index=False))
    print(f"Phase 5A success: {result['metrics']['success']}")
    print(
        "Ready for observer validation: "
        f"{result['metrics']['ready_for_observer_validation']}"
    )


if __name__ == "__main__":
    main()
