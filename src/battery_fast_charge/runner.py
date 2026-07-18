"""组织整组基线实验，并把轨迹、指标和图形写入固定目录。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PhaseOneConfig
from .high_fidelity import simulate_cccv
from .plotting import plot_baselines


def run_baseline_scan(config: PhaseOneConfig, project_root: str | Path) -> pd.DataFrame:
    """逐一运行配置中的 C-rate，并返回便于比较的汇总表。"""
    project_root = Path(project_root)
    data_dir = project_root / "data" / "processed"
    metrics_dir = project_root / "outputs" / "metrics"
    figures_dir = project_root / "outputs" / "figures"
    # 三类输出分开存放：完整时间序列、单工况指标、对比图。
    for directory in (data_dir, metrics_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    trajectories: dict[float, pd.DataFrame] = {}
    records: list[dict[str, Any]] = []
    for c_rate in config.baseline.c_rates:
        frame, metrics = simulate_cccv(c_rate, config)
        trajectories[c_rate] = frame
        frame.to_csv(data_dir / f"chen2020_cccv_{c_rate:g}C.csv", index=False)
        with (metrics_dir / f"chen2020_cccv_{c_rate:g}C.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(metrics, stream, ensure_ascii=False, indent=2)

        # JSON 保留嵌套结构，适合追溯单个工况；CSV 汇总表则需要扁平列，
        # 所以配置详情不重复展开，只把轨迹检查结果加上 check_ 前缀。
        record = {
            key: value
            for key, value in metrics.items()
            if key not in {"trajectory_checks", "configuration"}
        }
        record.update(
            {
                f"check_{key}": value
                for key, value in metrics["trajectory_checks"].items()
            }
        )
        records.append(record)

    # 按倍率排序，使表格和图例的阅读顺序稳定、便于复现实验结果。
    summary = pd.DataFrame.from_records(records).sort_values("c_rate")
    summary.to_csv(metrics_dir / "baseline_summary.csv", index=False)
    plot_baselines(
        trajectories,
        config,
        figures_dir / "chen2020_cccv_baseline_scan.png",
    )
    return summary
