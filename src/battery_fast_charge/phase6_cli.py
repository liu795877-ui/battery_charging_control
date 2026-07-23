"""Phase 6 论文式 DNN 显式 MPC 迁移验证命令入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .phase6_config import load_phase_six_config
from .phase6_runner import run_phase_six


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase6_paper_method_validation.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run_phase_six(load_phase_six_config(args.config), args.project_root)
    print(f"Paper dataset gate: {result['paper_dataset']['success']}")
    print(f"Nominal 25 C gate: {result['nominal_25c'].get('success', False)}")
    print(f"Phase 6 success: {result['success']}")


if __name__ == "__main__":
    main()
