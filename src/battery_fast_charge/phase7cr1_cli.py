"""Phase 7C-R1 命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .phase7cr1_config import load_phase7cr1_config
from .phase7cr1_runner import run_phase7cr1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    config = load_phase7cr1_config(
        root / "configs/phase7cr1_thermal_supervisor.yaml"
    )
    payload = run_phase7cr1(config, root, args.resume)
    print(json.dumps(payload["decision"], ensure_ascii=False))


if __name__ == "__main__":
    main()
