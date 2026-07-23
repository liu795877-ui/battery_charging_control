"""运行 Phase 7A Level 3P 最小输出投影验证。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .phase7a_level3p_config import load_phase7a_level3p_config
from .phase7a_level3p_runner import run_phase7a_level3p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase7a_level3p_projection.yaml"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run_phase7a_level3p(
        load_phase7a_level3p_config(args.config),
        args.project_root,
    )
    print(f"Phase 7A Level 3P success: {result['success']}")


if __name__ == "__main__":
    main()
