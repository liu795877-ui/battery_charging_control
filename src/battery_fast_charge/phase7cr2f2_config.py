"""Configuration loader for the Phase 7C-R2F2 two-stage voltage guard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Phase7CR2F2Config:
    payload: dict[str, Any]

    @property
    def study_name(self) -> str:
        return str(self.payload["study"]["name"])

    def section(self, name: str) -> dict[str, Any]:
        return self.payload[name]


def load_phase7cr2f2_config(path: str | Path) -> Phase7CR2F2Config:
    return Phase7CR2F2Config(
        yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    )
