"""第二阶段命令行入口。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .phase2_config import load_phase_two_config
from .phase2_runner import run_phase_two


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成 DFN 虚拟试验并辨识2RC＋双节点热模型。"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/phase2.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("PYBAMM_DISABLE_TELEMETRY", "true")
    args = parse_args()
    result = run_phase_two(load_phase_two_config(args.config), args.project_root)
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
