"""Command-line entry point for Phase 7C-R2F3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .phase7cr2f3_config import load_phase7cr2f3_config
from .phase7cr2f3_runner import run_phase7cr2f3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("prepare", "develop", "validate", "all")
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    config = load_phase7cr2f3_config(
        root / "configs/phase7cr2f3_temperature_two_stage_guards.yaml"
    )
    payload = run_phase7cr2f3(config, root, args.stage, args.resume)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
