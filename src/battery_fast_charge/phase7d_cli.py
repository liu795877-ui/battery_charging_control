from __future__ import annotations

import json
from pathlib import Path

from .phase7d_config import load_phase7d_config
from .phase7d_runner import run_phase7d_baseline


def main() -> None:
    root = Path.cwd()
    config = load_phase7d_config(root / "configs/phase7d_level4_performance.yaml")
    print(json.dumps(run_phase7d_baseline(config, root), ensure_ascii=False))


if __name__ == "__main__":
    main()

