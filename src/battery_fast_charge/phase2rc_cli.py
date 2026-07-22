"""运行 Phase 2R-C 前瞻式原生控制记忆审计。"""

from argparse import ArgumentParser
from pathlib import Path

from .phase2rc_runner import load_phase_two_rc_config, run_phase_two_rc


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase2rc_prospective_control_memory.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run_phase_two_rc(load_phase_two_rc_config(args.config), args.project_root)
    if result["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

