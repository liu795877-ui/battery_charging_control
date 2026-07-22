"""Phase 2R 模型与控制状态充分性审计命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .phase2r_config import load_phase_two_r_config
from .phase2r_runner import run_phase_two_r


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase2r_model_and_state_sufficiency.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run_phase_two_r(load_phase_two_r_config(args.config), args.project_root)
    print(f"Fixed model sufficient: {result['decision']['fixed_model_sufficient']}")
    print(f"Related model sufficient: {result['decision']['related_model_sufficient']}")
    print(f"Current DNN state sufficient: {result['decision']['current_dnn_state_sufficient']}")
    print(f"Next action: {result['decision']['next_action']}")


if __name__ == "__main__":
    main()
