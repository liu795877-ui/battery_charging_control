"""重建第二阶段降阶模型辨识实验 Notebook。"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "02_reduced_model_identification.ipynb"


def main() -> None:
    """按实验问题、协议、结果、验收和限制组织可重复运行的单元格。"""
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    notebook.cells = [
        new_markdown_cell(
            r"""# 第二阶段：2RC＋双节点热模型辨识

## 研究问题

能否用计算量较低的二阶 RC 电模型和双节点热模型，逼近 Chen2020 DFN 虚拟电芯在 25 ℃、10%–80% SOC 附近的电压、SOC 和平均温度动态？

**成功标准：**独立验证集电压 RMSE 不超过 50 mV，平均温度 RMSE 不超过 1.5 ℃。验证轨迹不参加参数拟合。"""
        ),
        new_markdown_cell(
            r"""## 模型和符号约定

项目规定充电电流 (I>0)。二阶 RC 模型为

\[
\frac{dz}{dt}=\frac{I}{3600Q_n},\qquad
\frac{dv_j}{dt}=-\frac{v_j}{R_jC_j}+\frac{I}{C_j},\quad j\in\{1,2\},
\]

\[
V=U_{oc}(z)+R_0I+v_1+v_2.
\]

双节点热模型为

\[
C_c\dot T_c=q-\frac{T_c-T_s}{R_{cs}},\qquad
C_s\dot T_s=\frac{T_c-T_s}{R_{cs}}-\frac{T_s-T_a}{R_{sa}}.
\]

Chen2020 集总热模型只提供平均温度，因此仅验证 (T_{avg}=0.8T_c+0.2T_s)。核心和表面温度是潜在状态，不作为已独立验证的真实温度。"""
        ),
        new_markdown_cell("""## 实验计划

1. 在 10%–80% SOC 读取平衡 OCV；
2. 在多个 SOC 施加 0.5C、1C、2C 短充电脉冲并静置，辨识 2RC；
3. 使用多档长脉冲和冷却轨迹辨识热模型；
4. 在未参与拟合的动态电流轨迹上验证；
5. 导出参数、误差、图形和适用边界。"""),
        new_code_cell("""from pathlib import Path
import json
import os
import pandas as pd
from IPython.display import Image, display

os.environ.setdefault("PYBAMM_DISABLE_TELEMETRY", "true")

# 无论从项目根目录还是 notebooks 目录启动，都自动找到配置和输出。
PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "configs" / "phase2.yaml").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

from battery_fast_charge.phase2_config import load_phase_two_config
from battery_fast_charge.phase2_runner import run_phase_two

config = load_phase_two_config(PROJECT_ROOT / "configs" / "phase2.yaml")
print(f"研究: {config.study_name}")
print(f"采样周期: {config.experiment.sample_period_s:g} s")"""),
        new_markdown_cell(
            """## 运行或复用正式结果

交付版本包含完整结果。将 `RUN_PHASE2` 改为 `True` 会重新运行多组 DFN 虚拟试验和参数优化，耗时明显长于读取缓存。"""
        ),
        new_code_cell("""RUN_PHASE2 = False

metrics_path = PROJECT_ROOT / "outputs" / "metrics" / "phase2_validation_metrics.json"
parameters_path = PROJECT_ROOT / "outputs" / "metrics" / "phase2_identified_parameters.json"
if RUN_PHASE2 or not (metrics_path.exists() and parameters_path.exists()):
    run_phase_two(config, PROJECT_ROOT)

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
ocv = pd.read_csv(PROJECT_ROOT / "data" / "phase2" / "ocv_curve.csv")
display(ocv.round(5))"""),
        new_markdown_cell("## OCV与脉冲表征"),
        new_code_cell("""display(Image(filename=str(
    PROJECT_ROOT / "outputs" / "figures" / "phase2_characterization.png"
)))"""),
        new_markdown_cell("## 已辨识参数"),
        new_code_cell(
            """electrical = pd.Series(parameters["electrical_2rc"], name="value")
thermal = pd.Series(parameters["thermal_two_node"], name="value")
display(electrical.to_frame())
display(thermal.to_frame())"""
        ),
        new_markdown_cell("## 独立动态轨迹验证"),
        new_code_cell("""display(Image(filename=str(
    PROJECT_ROOT / "outputs" / "figures" / "phase2_reduced_model_validation.png"
)))

summary = pd.DataFrame({
    "metric": ["voltage_rmse_mv", "temperature_rmse_c", "overall_success"],
    "value": [
        metrics["validation"]["voltage"]["rmse_mv"],
        metrics["validation"]["average_temperature"]["rmse_c"],
        metrics["success"],
    ],
})
display(summary)"""),
        new_markdown_cell("""## 解释与下一步

- 通过验收只表示降阶模型在当前 25 ℃和已验证激励范围内足够准确，不表示它在所有电芯、温度、老化状态下都有效；
- 核心/表面温度没有独立观测，而且核心—表面热阻命中优化下限，两个节点退化为近似同温；因此未来 MPC 第一版应控制平均温度并保留安全裕量，不能宣称核心温度已经验证；
- 若独立验证通过，下一阶段可在此模型上构建带电压、电流、温度和电流变化率约束的 MPC；
- MPC 最终仍需回到 DFN 虚拟电芯上做闭环验证，不能只在自己的简化模型上证明有效。"""),
    ]
    # 为每个单元格指定稳定 ID，避免未来 nbformat 把缺失 ID 视为格式错误。
    for index, cell in enumerate(notebook.cells):
        cell["id"] = f"phase2-{index:02d}"
    notebook.metadata.setdefault("kernelspec", {})
    notebook.metadata["kernelspec"].update(
        {
            "display_name": "Python 3 (battery-fast-charge)",
            "language": "python",
            "name": "python3",
        }
    )
    notebook.metadata.setdefault("language_info", {})["name"] = "python"
    nbformat.write(notebook, NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
