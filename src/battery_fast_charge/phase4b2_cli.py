"""阶段4B-2主动数据聚合与ANN v2命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .phase4b2_config import load_phase_four_b2_config
from .phase4b2_runner import run_phase_four_b2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase4b2.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    config = load_phase_four_b2_config(args.config)
    result = run_phase_four_b2(config, args.project_root)
    print(result["comparison"].to_string(index=False))
    print(f"Phase 4B-2 success: {result['metrics']['success']}")
    print(
        "Ready for robustness validation: "
        f"{result['metrics']['ready_for_robustness_validation']}"
    )


if __name__ == "__main__":
    main()
