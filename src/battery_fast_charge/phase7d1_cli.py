from __future__ import annotations

import json
from pathlib import Path

from .phase7d1_config import load_phase7d1_config
from .phase7d1_runner import prepare_phase7d1_states


def main() -> None:
    root = Path.cwd()
    config = load_phase7d1_config(
        root / "configs/phase7d1_performance_optimization.yaml"
    )
    print(json.dumps(prepare_phase7d1_states(config, root), ensure_ascii=False))


if __name__ == "__main__":
    main()

