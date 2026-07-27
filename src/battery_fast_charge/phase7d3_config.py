from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Phase7D3Config:
    payload: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        return self.payload[name]


def load_phase7d3_config(path: str | Path) -> Phase7D3Config:
    return Phase7D3Config(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

