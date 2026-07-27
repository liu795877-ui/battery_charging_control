from __future__ import annotations

import argparse
import json
from pathlib import Path

from .phase7d3_config import load_phase7d3_config
from .phase7d3_runner import prepare_states, run_final_confirmation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "confirm"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    config = load_phase7d3_config(root / "configs/phase7d3_final_confirmation.yaml")
    result = prepare_states(config, root) if args.stage == "prepare" else run_final_confirmation(config, root, args.resume)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

