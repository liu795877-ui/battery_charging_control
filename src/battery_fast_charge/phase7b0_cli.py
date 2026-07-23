"""运行 Phase 7B-0：冻结 ANN＋投影的 25 ℃ Chen2020 DFN 跨模型审计。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .phase7b0_config import load_phase7b0_config
from .phase7b0_runner import run_phase7b0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase7b0_dfn_cross_model.yaml"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-trajectories", type=int)
    args = parser.parse_args()
    result = run_phase7b0(
        load_phase7b0_config(args.config),
        args.project_root,
        resume=args.resume,
        limit_trajectories=args.limit_trajectories,
    )
    print(f"Phase 7B-0 success: {result['success']}")


if __name__ == "__main__":
    main()
