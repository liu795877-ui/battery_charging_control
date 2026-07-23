"""Phase 7C 命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .phase7c_config import load_phase7c_config
from .phase7c_runner import run_phase7c


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/phase7c_multitemperature_dfn_validation.yaml",
    )
    parser.add_argument(
        "--stage",
        choices=("freeze", "mpc", "ann", "analyze", "all"),
        default="all",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    config = load_phase7c_config(root / args.config)
    payload = run_phase7c(config, root, args.stage, args.resume)
    print(json.dumps(payload.get("decision", payload), ensure_ascii=False))


if __name__ == "__main__":
    main()
