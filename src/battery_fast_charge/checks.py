"""对导出的充电轨迹做轻量级完整性和物理方向检查。"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "time_s",
    "charge_current_a",
    "terminal_voltage_v",
    "soc",
    "cell_temperature_c",
    "ambient_temperature_c",
}


def check_trajectory(frame: pd.DataFrame) -> dict[str, bool]:
    """返回各项检查结果；约束是否超限不属于本函数的检查范围。"""
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing trajectory columns: {sorted(missing)}")
    if frame.empty:
        return {
            "nonempty": False,
            "finite": False,
            "time_monotonic": False,
            "soc_nondecreasing": False,
            "charge_current_nonnegative": False,
        }

    values = frame[list(REQUIRED_COLUMNS)].to_numpy(dtype=float)
    return {
        "nonempty": True,
        "finite": bool(np.isfinite(values).all()),
        "time_monotonic": bool(np.all(np.diff(frame["time_s"]) > 0.0)),
        # 数值求解会产生极小舍入误差，因此允许 1e-7 量级的 SOC 反向波动。
        "soc_nondecreasing": bool(np.all(np.diff(frame["soc"]) >= -1.0e-7)),
        # 同理，-1e-8 A 以内视为数值零，而不是真正的放电电流。
        "charge_current_nonnegative": bool(
            np.all(frame["charge_current_a"].to_numpy() >= -1.0e-8)
        ),
    }
