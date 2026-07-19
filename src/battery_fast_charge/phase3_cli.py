"""第三阶段命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .phase3_config import load_phase_three_config
from .phase3_runner import run_phase_three


def main() -> None:
    """读取配置并运行完整第三阶段 A。"""
    parser = argparse.ArgumentParser(description="Run phase 3 constrained MPC validation")
    parser.add_argument("--config", default="configs/phase3.yaml")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    config = load_phase_three_config(project_root / args.config)
    result = run_phase_three(config, project_root)
    print(result["comparison"].to_string(index=False))
    print(f"Phase 3 success: {result['metrics']['success']}")


if __name__ == "__main__":
    main()
