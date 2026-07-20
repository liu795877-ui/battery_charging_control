"""阶段4B-1命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .phase4b_config import load_phase_four_b_config
from .phase4b_runner import run_phase_four_b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase4b.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    config = load_phase_four_b_config(args.config)
    result = run_phase_four_b(config, args.project_root)
    print(result["comparison"].to_string(index=False))
    print(f"Phase 4B-1 success: {result['metrics']['success']}")
    print(
        "Ready for active data aggregation: "
        f"{result['metrics']['ready_for_active_data_aggregation']}"
    )


if __name__ == "__main__":
    main()
