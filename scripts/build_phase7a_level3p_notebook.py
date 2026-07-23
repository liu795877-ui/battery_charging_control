"""生成并执行 Phase 7A Level 3P 结果 Notebook。"""

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).parents[1]
PATH = ROOT / "notebooks" / "phase7a_level3p_projection_results.ipynb"
notebook = nbf.v4.new_notebook()
notebook.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
notebook.cells = [
    nbf.v4.new_markdown_cell(
        r"""# Phase 7A Level 3P：最小输出投影

在不增加教师数据、不重新训练网络、不改变 MPC、模型、初态和五个随机种子的
条件下，将 DNN 原始输出投影到

\[
\left[\max(0,I_{k-1}-2),\ \min(10,I_{k-1}+2)\right].
\]

本实验只判断最小投影能否修复 Level 3 的硬斜率失效，不进入 Level 4。"""
    ),
    nbf.v4.new_markdown_cell("## 1. 读取冻结验证和投影结果"),
    nbf.v4.new_code_cell(
        """from pathlib import Path
import json, pandas as pd
from IPython.display import display, Image
ROOT=Path.cwd(); OUT=ROOT/'outputs'/'phase7a_level3p_projection'; DATA=ROOT/'data'/'phase7a_level3p_projection'
metrics=json.loads((OUT/'metrics.json').read_text(encoding='utf-8'))
closed=pd.read_csv(DATA/'projected_closed_loop_metrics.csv')
interventions=pd.read_csv(DATA/'projection_interventions.csv')
print(metrics['study_name'], metrics['status'])"""
    ),
    nbf.v4.new_markdown_cell("## 2. Level 3 冻结合同"),
    nbf.v4.new_code_cell(
        """frozen=pd.DataFrame(metrics['frozen_artifact_verification']).T
display(frozen)
assert len(frozen)==13
assert frozen.matched.all()"""
    ),
    nbf.v4.new_markdown_cell("## 3. 投影介入范围"),
    nbf.v4.new_code_cell(
        """display(pd.DataFrame([metrics['projection']]))
display(interventions.groupby('seed').size().rename('intervention_count'))
assert metrics['projection']['frozen_raw_violation_count']==48
assert metrics['projection']['projection_intervention_count']==48
assert metrics['projection']['exact_key_overlap_count']==48
assert metrics['projection']['interventions_outside_one_step_count']==0"""
    ),
    nbf.v4.new_markdown_cell("## 4. 五种子闭环严格门槛"),
    nbf.v4.new_code_cell(
        """view=closed[['seed','mean_current_nrmse','mean_charge_time_gap_fraction','target_reach_fraction',
             'maximum_voltage_violation_v','maximum_slew_violation_a','maximum_current_step_a',
             'projection_intervention_count','speedup']].copy()
view['current_nrmse_percent']=100*view.pop('mean_current_nrmse')
view['charge_time_gap_percent']=100*view.pop('mean_charge_time_gap_fraction')
display(view)
assert (view.current_nrmse_percent<1).all()
assert (view.charge_time_gap_percent<2).all()
assert (view.target_reach_fraction==1).all()
assert (view.maximum_voltage_violation_v==0).all()
assert (view.maximum_slew_violation_a<=1e-12).all()
assert (view.speedup>100).all()
display(Image(filename=str(OUT/'figures'/'projection_gate_and_interventions.png')))"""
    ),
    nbf.v4.new_markdown_cell("## 5. 最终判定"),
    nbf.v4.new_code_cell(
        """display(pd.DataFrame([metrics['decision']['checks']]))
print(metrics['decision']['conclusion'])
assert metrics['decision']['level3p_success']
assert metrics['decision']['proceed_to_level4'] is False"""
    ),
    nbf.v4.new_markdown_cell(
        """## 结论

最小输出投影只介入冻结 Level 3 中原有的 48 个斜率风险动作，没有新增邻域外
干预。投影后斜率违约为零，五种子闭环 NRMSE、到达时间、电压安全、目标到达
和在线加速全部通过。

这说明 Level 3 的失效不是策略拟合不足，而是 unconstrained pure DNN 缺乏硬约束
保证。Level 3P 用最小安全层修复该缺口；本研究停止在 Level 3P，不进入温度
Level 4。"""
    ),
]
for index, cell in enumerate(notebook.cells):
    cell["id"] = f"level3p-{index:02d}"
nbf.write(notebook, PATH)
executed = nbf.read(PATH, as_version=4)
NotebookClient(
    executed,
    timeout=180,
    kernel_name="python3",
    resources={"metadata": {"path": str(ROOT)}},
).execute()
nbf.write(executed, PATH)
print(PATH)
