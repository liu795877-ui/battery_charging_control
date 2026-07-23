"""生成 Phase 7A Level 2–3–3P 核心结论图及源数据。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd


ROOT = Path(__file__).parents[1]
OUTPUT = ROOT / "outputs" / "phase7a_core_conclusion"
OUTPUT.mkdir(parents=True, exist_ok=True)

level2 = json.loads(
    (ROOT / "outputs" / "phase7a_level2_2rc" / "metrics.json").read_text(
        encoding="utf-8"
    )
)
level3 = json.loads(
    (ROOT / "outputs" / "phase7a_level3_slew" / "metrics.json").read_text(
        encoding="utf-8"
    )
)
level3p = json.loads(
    (ROOT / "outputs" / "phase7a_level3p_projection" / "metrics.json").read_text(
        encoding="utf-8"
    )
)

source = pd.DataFrame(
    [
        {
            "stage": "Level 2",
            "added_complexity": "Second polarization state",
            "closed_loop_nrmse_min_percent": 100
            * level2["closed_loop"]["current_nrmse_min"],
            "closed_loop_nrmse_max_percent": 100
            * level2["closed_loop"]["current_nrmse_max"],
            "maximum_current_step_a": float("nan"),
            "hard_slew_status": "Not active",
            "conclusion": "pure DNN passes",
        },
        {
            "stage": "Level 3",
            "added_complexity": "Previous current + hard slew",
            "closed_loop_nrmse_min_percent": 100
            * level3["closed_loop"]["current_nrmse_min"],
            "closed_loop_nrmse_max_percent": 100
            * level3["closed_loop"]["current_nrmse_max"],
            "maximum_current_step_a": level3["closed_loop"][
                "maximum_current_step_a"
            ],
            "hard_slew_status": "Fail",
            "conclusion": "No hard guarantee",
        },
        {
            "stage": "Level 3P",
            "added_complexity": "Minimal output projection",
            "closed_loop_nrmse_min_percent": 100
            * level3p["closed_loop"]["current_nrmse_min"],
            "closed_loop_nrmse_max_percent": 100
            * level3p["closed_loop"]["current_nrmse_max"],
            "maximum_current_step_a": level3p["closed_loop"][
                "maximum_current_step_a"
            ],
            "hard_slew_status": "Pass",
            "conclusion": "ANN + projection passes",
        },
    ]
)
source["projection_intervention_count"] = [
    0,
    0,
    level3p["projection"]["projection_intervention_count"],
]
source["projection_intervention_fraction_percent"] = [
    0.0,
    0.0,
    100 * level3p["projection"]["projection_intervention_fraction"],
]
source.to_csv(OUTPUT / "source_data.csv", index=False)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Microsoft YaHei",
            "Arial",
            "Helvetica",
            "DejaVu Sans",
            "sans-serif",
        ],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)

colors = {
    "navy": "#24435C",
    "blue": "#78A6C8",
    "orange": "#E69F5B",
    "green": "#66A182",
    "light": "#F4F6F7",
    "line": "#D7DDE1",
    "text": "#1D2A32",
    "muted": "#66727A",
}

fig = plt.figure(figsize=(7.2, 4.35), facecolor="white")
grid = fig.add_gridspec(
    2,
    3,
    height_ratios=[1.45, 1.0],
    width_ratios=[1.45, 0.78, 0.77],
    hspace=0.42,
    wspace=0.46,
)

ax_table = fig.add_subplot(grid[0, :])
ax_table.set_axis_off()
ax_table.text(
    0.0,
    1.10,
    "a",
    transform=ax_table.transAxes,
    fontsize=10,
    fontweight="bold",
    va="top",
)
ax_table.text(
    0.035,
    1.10,
    "The first failure is a hard-constraint guarantee—not policy fitting",
    transform=ax_table.transAxes,
    fontsize=10,
    fontweight="bold",
    color=colors["text"],
    va="top",
)

columns = [
    ("Stage", 0.14),
    ("Added complexity", 0.28),
    ("Closed-loop NRMSE", 0.19),
    ("Hard slew", 0.15),
    ("Conclusion", 0.24),
]
x = [0.0]
for _, width in columns:
    x.append(x[-1] + width)
header_y, row_height = 0.78, 0.245
ax_table.add_patch(
    Rectangle(
        (0, header_y),
        1,
        0.16,
        transform=ax_table.transAxes,
        facecolor=colors["navy"],
        edgecolor="none",
    )
)
for index, (label, _) in enumerate(columns):
    ax_table.text(
        (x[index] + x[index + 1]) / 2,
        header_y + 0.08,
        label,
        transform=ax_table.transAxes,
        ha="center",
        va="center",
        color="white",
        fontsize=7.5,
        fontweight="bold",
    )

row_colors = [colors["blue"], colors["orange"], colors["green"]]
rows = [
    [
        "Level 2",
        "Second polarization state",
        f"{source.loc[0, 'closed_loop_nrmse_min_percent']:.3f}–"
        f"{source.loc[0, 'closed_loop_nrmse_max_percent']:.3f}%",
        "Not active",
        "pure DNN passes",
    ],
    [
        "Level 3",
        "Previous current +\nhard slew",
        f"{source.loc[1, 'closed_loop_nrmse_min_percent']:.3f}–"
        f"{source.loc[1, 'closed_loop_nrmse_max_percent']:.3f}%",
        "FAIL",
        "No hard guarantee",
    ],
    [
        "Level 3P",
        "Minimal output projection",
        f"{source.loc[2, 'closed_loop_nrmse_min_percent']:.3f}–"
        f"{source.loc[2, 'closed_loop_nrmse_max_percent']:.3f}%",
        "PASS",
        "ANN + projection passes",
    ],
]
for row_index, values in enumerate(rows):
    y0 = header_y - (row_index + 1) * row_height
    ax_table.add_patch(
        Rectangle(
            (0, y0),
            1,
            row_height,
            transform=ax_table.transAxes,
            facecolor="white" if row_index % 2 == 0 else colors["light"],
            edgecolor=colors["line"],
            linewidth=0.6,
        )
    )
    ax_table.add_patch(
        Rectangle(
            (0, y0),
            0.012,
            row_height,
            transform=ax_table.transAxes,
            facecolor=row_colors[row_index],
            edgecolor="none",
        )
    )
    for column_index, value in enumerate(values):
        color = colors["text"]
        weight = "bold" if column_index in (0, 3) else "normal"
        if value == "FAIL":
            color = "#B45A2A"
        elif value == "PASS":
            color = "#347858"
        ax_table.text(
            (x[column_index] + x[column_index + 1]) / 2,
            y0 + row_height / 2,
            value,
            transform=ax_table.transAxes,
            ha="center",
            va="center",
            fontsize=7.3,
            color=color,
            fontweight=weight,
        )

ax_step = fig.add_subplot(grid[1, 0])
ax_step.text(
    -0.08,
    1.12,
    "b",
    transform=ax_step.transAxes,
    fontsize=10,
    fontweight="bold",
    va="top",
)
values = [
    level3["closed_loop"]["maximum_current_step_a"],
    level3p["closed_loop"]["maximum_current_step_a"],
]
bars = ax_step.bar(
    ["Level 3", "Level 3P"],
    values,
    color=[colors["orange"], colors["green"]],
    width=0.56,
)
ax_step.axhline(2.0, color="#B23A48", linestyle="--", linewidth=1.2)
ax_step.set_ylim(1.92, 2.16)
ax_step.set_ylabel("Maximum current step (A)")
ax_step.set_title("Projection restores the 2 A hard limit", loc="left", fontsize=8.5)
ax_step.grid(axis="y", color=colors["line"], linewidth=0.6)
for bar, value in zip(bars, values):
    ax_step.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.008,
        f"{value:.4f}",
        ha="center",
        va="bottom",
        fontsize=7.2,
        fontweight="bold",
    )

ax_rate = fig.add_subplot(grid[1, 1:])
ax_rate.text(
    -0.08,
    1.12,
    "c",
    transform=ax_rate.transAxes,
    fontsize=10,
    fontweight="bold",
    va="top",
)
rate = 100 * level3p["projection"]["projection_intervention_fraction"]
ax_rate.barh(
    [0],
    [100 - rate],
    color=colors["light"],
    edgecolor=colors["line"],
    height=0.42,
)
ax_rate.barh([0], [rate], color=colors["green"], height=0.42)
ax_rate.set_xlim(0, 100)
ax_rate.set_yticks([])
ax_rate.set_xlabel("Share of closed-loop actions (%)")
ax_rate.set_title(
    "Only the pre-existing risk actions are modified", loc="left", fontsize=8.5
)
ax_rate.text(
    rate + 1.2,
    0,
    f"48 / 13,349 actions\n{rate:.4f}% intervention",
    ha="left",
    va="center",
    fontsize=8,
    color=colors["text"],
    fontweight="bold",
)
ax_rate.text(
    99.2,
    0.34,
    "Exact overlap with Level 3 risk set: 48/48\nNew interventions outside ±1 step: 0",
    ha="right",
    va="center",
    fontsize=7,
    color=colors["muted"],
)
ax_rate.spines["left"].set_visible(False)
ax_rate.spines["bottom"].set_color(colors["line"])

fig.text(
    0.5,
    0.012,
    "Five fixed seeds; same 2RC model, MPC, initial states and networks. "
    "NRMSE is normalized by the 10 A current range.",
    ha="center",
    va="bottom",
    fontsize=6.4,
    color=colors["muted"],
)
fig.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.14)

stem = OUTPUT / "phase7a_level2_level3_level3p_core_conclusion"
fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
plt.close(fig)
