"""Phase 7C-R2F2残差初始化审计入口。"""

from pathlib import Path
import json

from .phase7cr2f2_residual_audit import load_config, run_audit


def main() -> None:
    root = Path.cwd()
    config = load_config(
        root / "configs/phase7cr2f2_residual_initialization_audit.yaml"
    )
    payload = run_audit(config, root)
    print(json.dumps(payload["decision"], ensure_ascii=False))


if __name__ == "__main__":
    main()
