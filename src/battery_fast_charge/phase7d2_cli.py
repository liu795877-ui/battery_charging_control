from __future__ import annotations

import argparse
import json
from pathlib import Path

from .phase7d2_config import load_phase7d2_config
from .phase7d2_runner import run_development, run_internal_validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("develop", "validate"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    config = load_phase7d2_config(
        root / "configs/phase7d2_thermal_performance_development.yaml"
    )
    if args.stage == "develop":
        result = run_development(config, root, args.resume)
    else:
        result = run_internal_validation(config, root, args.resume)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

