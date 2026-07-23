"""Phase 7C-R2 分温度电压裕量配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Phase7CR2Config:
    payload: dict[str, Any]

    @property
    def study_name(self) -> str:
        return str(self.payload["study"]["name"])

    @property
    def sources(self) -> dict[str, Any]:
        return self.payload["sources"]

    @property
    def voltage(self) -> dict[str, Any]:
        return self.payload["voltage_guard"]

    @property
    def datasets(self) -> dict[str, Any]:
        return self.payload["datasets"]

    @property
    def gates(self) -> dict[str, Any]:
        return self.payload["gates"]

    @property
    def output(self) -> dict[str, Any]:
        return self.payload["output"]


def load_phase7cr2_config(path: str | Path) -> Phase7CR2Config:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Phase7CR2Config(payload)
