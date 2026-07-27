from __future__ import annotations

import json
from pathlib import Path

from .phase7d3_config import load_phase7d3_config
from .phase7d3_finalize import finalize_completed_confirmation


def main() -> None:
    root = Path.cwd()
    config = load_phase7d3_config(root / "configs/phase7d3_final_confirmation.yaml")
    print(json.dumps(finalize_completed_confirmation(config, root), ensure_ascii=False))


if __name__ == "__main__":
    main()

