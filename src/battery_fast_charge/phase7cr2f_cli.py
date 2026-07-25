"""Phase 7C-R2F命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .phase7cr2f_config import load_phase7cr2f_config
from .phase7cr2f_runner import run_phase7cr2f


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    config = load_phase7cr2f_config(
        root / "configs/phase7cr2f_teacher_selection_voltage_guard.yaml"
    )
    payload = run_phase7cr2f(config, root, args.resume)
    print(json.dumps(payload["decision"], ensure_ascii=False))


if __name__ == "__main__":
    main()
