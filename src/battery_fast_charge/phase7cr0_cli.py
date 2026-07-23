"""Phase 7C-R0 命令行入口。"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

from .phase7cr0_config import load_phase7cr0_config
from .phase7cr0_runner import refine_low3_candidates, run_phase7cr0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refine-low3", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    config = load_phase7cr0_config(
        root / "configs/phase7cr0_diagnostics.yaml"
    )
    if args.refine_low3:
        payload = refine_low3_candidates(config, root)
        print(json.dumps(payload, ensure_ascii=False))
    else:
        payload = run_phase7cr0(config, root)
        print(json.dumps(payload["decision"], ensure_ascii=False))


if __name__ == "__main__":
    main()
