from __future__ import annotations

import argparse
import json
from pathlib import Path

from .phase7cr3t_config import load_phase7cr3t_config
from .phase7cr3t_runner import run_phase7cr3t


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("equivalence", "confirm", "all"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    config = load_phase7cr3t_config(root / "configs/phase7cr3t_supervisor_runtime.yaml")
    print(json.dumps(run_phase7cr3t(config, root, args.stage, args.resume), ensure_ascii=False))


if __name__ == "__main__":
    main()
