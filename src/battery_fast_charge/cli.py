"""命令行入口：读取配置并启动第一阶段基线扫描。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import load_config
from .runner import run_baseline_scan


def parse_args() -> argparse.Namespace:
    """定义用户可从命令行修改的路径参数。"""
    parser = argparse.ArgumentParser(
        description="Run the phase-one Chen2020 baseline scan."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase1.yaml"),
        help="Path to the phase-one YAML configuration.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Root directory for data and outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """执行完整流程，并在终端打印最终汇总表。"""
    args = parse_args()
    # 关闭匿名遥测，避免仿真依赖外部网络，也让实验更容易离线复现。
    os.environ.setdefault("PYBAMM_DISABLE_TELEMETRY", "true")
    config = load_config(args.config)
    summary = run_baseline_scan(config, args.project_root)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
