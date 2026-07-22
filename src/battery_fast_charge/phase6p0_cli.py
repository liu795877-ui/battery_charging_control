"""Phase 6P-0 论文 NDC 原位复现命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .phase6p0_config import load_phase_six_p_zero_config
from .phase6p0_runner import run_phase_six_p_zero


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase6p0_ndc_paper.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--resume", action="store_true", help="复用已保存的教师数据、模型和闭环轨迹，仅重建汇总产物。")
    args = parser.parse_args()
    metrics = run_phase_six_p_zero(load_phase_six_p_zero_config(args.config), args.project_root, resume=args.resume)
    print(f"Phase 6P-0 success: {metrics['success']}")
    print(f"Offline NRMSE: {metrics['offline']['nrmse_percent']:.4f}%")
    print(f"Closed-loop NRMSE: {metrics['closed_loop']['mean_trajectory_current_nrmse_percent']:.4f}%")


if __name__ == "__main__":
    main()
