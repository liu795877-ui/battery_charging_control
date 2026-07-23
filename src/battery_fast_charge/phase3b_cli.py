"""第三阶段 B 命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .phase3b_config import load_phase_three_b_config
from .phase3b_runner import run_phase_three_b


def main() -> None:
    """读取配置并运行公平基线与MPC教师数据生成。"""
    parser = argparse.ArgumentParser(description="Run phase 3B teacher data generation")
    parser.add_argument("--config", default="configs/phase3b.yaml")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    config = load_phase_three_b_config(project_root / args.config)
    result = run_phase_three_b(config, project_root)
    print(result["comparison"].to_string(index=False))
    print(f"Phase 3B success: {result['metrics']['success']}")
    print(f"Ready for DNN training: {result['metrics']['ready_for_dnn_training']}")


if __name__ == "__main__":
    main()
