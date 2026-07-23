"""填充并执行 Phase 7A Level 3 结果 Notebook。"""

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).parents[1]
PATH = ROOT / "notebooks" / "phase7a_level3_slew_results.ipynb"
notebook = nbf.v4.new_notebook()
notebook.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
notebook.cells = [
    nbf.v4.new_markdown_cell(
        r"""# Phase 7A Level 3：硬斜率约束 pure DNN 验证

**问题**：在 Level 2 三状态模型已通过的基础上，只增加上一时刻电流和
\(|I_k-I_{k-1}|\le2\,\mathrm{A}\) 硬约束，pure DNN 是否仍能严格通过？

本实验不增加温度、DFN、扰动、压力场、教师数据规模或网络复杂度。"""
    ),
    nbf.v4.new_markdown_cell(
        r"""## 1. 状态与唯一新增约束

\[
x_k=\begin{bmatrix}SOC_k&V_{1,k}&V_{2,k}&I_{k-1}\end{bmatrix}^{\mathsf T},
\qquad |I_k-I_{k-1}|\le2\ \mathrm{A}.
\]

其余 2RC 动力学、电流边界 \(0\le I_k\le10\,\mathrm{A}\) 和端电压上限
\(V_{\mathrm{tr},k}\le4.20\,\mathrm{V}\) 与 Level 2 保持一致。"""
    ),
    nbf.v4.new_markdown_cell("## 2. 读取冻结结果"),
    nbf.v4.new_code_cell(
        """from pathlib import Path
import json, pandas as pd
from IPython.display import display, Image
ROOT=Path.cwd(); OUT=ROOT/'outputs'/'phase7a_level3_slew'; DATA=ROOT/'data'/'phase7a_level3_slew'
metrics=json.loads((OUT/'metrics.json').read_text(encoding='utf-8'))
offline=pd.read_csv(OUT/'dnn_offline_metrics.csv')
closed=pd.read_csv(DATA/'closed_loop_metrics.csv')
audit=pd.read_csv(DATA/'multistart_state_summary.csv')
print(metrics['study_name'], metrics['status'])"""
    ),
    nbf.v4.new_markdown_cell("## 3. 教师覆盖与双冻结测试哈希"),
    nbf.v4.new_code_cell(
        """display(pd.DataFrame([metrics['teacher']]))
display(pd.DataFrame(metrics['frozen_test_hashes']).T)
assert metrics['teacher']['success']
assert metrics['teacher']['attempted_trajectories']==400
assert metrics['teacher']['low_current_label_count']>=100
assert metrics['teacher']['checks']['hard_slew_constraint']"""
    ),
    nbf.v4.new_markdown_cell("## 4. 100×15 多起点教师审计"),
    nbf.v4.new_code_cell(
        """display(pd.DataFrame([metrics['teacher_audit']]))
assert len(audit)==100
assert metrics['teacher_audit']['success']
assert metrics['teacher_audit']['near_optimal_multivalued_fraction']==0
display(Image(filename=str(OUT/'figures'/'teacher_and_audit.png')))"""
    ),
    nbf.v4.new_markdown_cell("## 5. 深层 LBFGS 五种子离线结果"),
    nbf.v4.new_code_cell(
        """view=offline[['seed','test_nrmse','terminal_test_nrmse','test_bias_a','terminal_test_bias_a']].copy()
view['global_test_nrmse_percent']=100*view.pop('test_nrmse')
view['terminal_test_nrmse_percent']=100*view.pop('terminal_test_nrmse')
display(view)
assert (view.global_test_nrmse_percent<1).all()
assert (view.terminal_test_nrmse_percent<1).all()"""
    ),
    nbf.v4.new_markdown_cell("## 6. 同模型闭环与硬斜率门槛"),
    nbf.v4.new_code_cell(
        """cols=['seed','mean_current_nrmse','mean_charge_time_gap_fraction','target_reach_fraction',
      'maximum_voltage_violation_v','maximum_slew_violation_a','maximum_current_step_a','speedup']
view=closed[cols].copy()
view['current_nrmse_percent']=100*view.pop('mean_current_nrmse')
view['charge_time_gap_percent']=100*view.pop('mean_charge_time_gap_fraction')
display(view)
display(Image(filename=str(OUT/'figures'/'five_seed_validation.png')))
assert (view.current_nrmse_percent<1).all()
assert (view.charge_time_gap_percent<2).all()
assert (view.target_reach_fraction==1).all()
assert (view.maximum_voltage_violation_v==0).all()
assert (view.speedup>100).all()
assert (view.maximum_slew_violation_a>0.001).all()"""
    ),
    nbf.v4.new_markdown_cell("## 7. 阶段判定"),
    nbf.v4.new_code_cell(
        """display(pd.DataFrame([metrics['decision']['checks']]))
display(pd.DataFrame([metrics['closed_loop']['checks']]))
print(metrics['decision']['conclusion'])
assert metrics['decision']['level3_success'] is False
assert metrics['decision']['proceed_to_level4'] is False
assert metrics['closed_loop']['checks']['zero_slew_violation'] is False"""
    ),
    nbf.v4.new_markdown_cell(
        """## 结论

Level 3 的教师确定性、双冻结离线拟合、闭环电流误差、终端到达、安全电压和
计算速度均通过；唯一失败项是 pure DNN 无法严格保证新增的硬斜率约束。

因此硬斜率约束是本消融序列中首次确认的 pure DNN 严格失效因素。依据预注册
停止规则，不进入 Level 4。后续若继续研究，应单独立项比较安全投影、分区 DNN
或 ANN 辅助 MPC，而不能把这些方法记为原始 pure DNN。"""
    ),
]
for index, cell in enumerate(notebook.cells):
    cell["id"] = f"level3-{index:02d}"
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
