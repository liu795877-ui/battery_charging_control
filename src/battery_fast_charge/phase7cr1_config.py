"""Phase 7C-R1 热安全监督层配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Phase7CR1Config:
    payload: dict[str, Any]

    @property
    def study_name(self) -> str:
        return str(self.payload["study"]["name"])

    @property
    def sources(self) -> dict[str, Any]:
        return self.payload["sources"]

    @property
    def thermal(self) -> dict[str, Any]:
        return self.payload["thermal_supervisor"]

    @property
    def teacher(self) -> dict[str, Any]:
        return self.payload["teacher_repair"]

    @property
    def development(self) -> dict[str, Any]:
        return self.payload["development"]

    @property
    def gates(self) -> dict[str, Any]:
        return self.payload["gates"]

    @property
    def output(self) -> dict[str, Any]:
        return self.payload["output"]


def load_phase7cr1_config(path: str | Path) -> Phase7CR1Config:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Phase7CR1Config(payload=payload)
