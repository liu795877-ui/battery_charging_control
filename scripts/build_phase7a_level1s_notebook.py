"""填充并执行 Phase 7A Level 1S 结果 Notebook。"""

from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).parents[1]
PATH = ROOT / "notebooks" / "phase7a_level1s_training_stability_results.ipynb"
nb = nbf.read(PATH, as_version=4)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.cells = [
    nbf.v4.new_markdown_cell("""# Phase 7A Level 1S：训练稳定性消融

**目标**：不增加教师数据或模型复杂度，只比较三种网络结构与 Adam/LBFGS 的训练稳定性。数据、双冻结测试、MPC、1RC、约束、闭环初态和原验收门槛全部冻结。

只有选定方案的五个原种子全部通过双冻结测试与完整同模型闭环，才允许进入 Level 2。"""),
    nbf.v4.new_markdown_cell("## 1. 读取冻结结果"),
    nbf.v4.new_code_cell("""from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image

ROOT = Path.cwd()
OUT = ROOT / 'outputs' / 'phase7a_level1s_training_stability'
DATA = ROOT / 'data' / 'phase7a_level1s_training_stability'
metrics = json.loads((OUT / 'metrics.json').read_text(encoding='utf-8'))
candidates = pd.read_csv(OUT / 'candidate_metrics.csv')
schemes = pd.read_csv(OUT / 'scheme_validation_summary.csv')
selected = pd.read_csv(OUT / 'selected_scheme_five_seed_metrics.csv')
closed = pd.read_csv(DATA / 'closed_loop_metrics.csv')
diagnostics = pd.read_csv(DATA / 'closed_loop_diagnostics_per_seed.csv')
print(metrics['study_name'], metrics['status'])"""),
    nbf.v4.new_markdown_cell(r"""## 2. 诊断量定义

充电电流为正，采样周期为 \(\Delta t=5\,\mathrm{s}\)。累计电荷误差定义为

\[
\Delta Q=\frac{\Delta t}{3600}\left(\sum_{k=0}^{N_{\mathrm{DNN}}-1}I_k^{\mathrm{DNN}}-\sum_{k=0}^{N_{\mathrm{MPC}}-1}I_k^{\mathrm{MPC}}\right)\ \mathrm{Ah}.
\]

有符号步数差为

\[
\Delta N=N_{\mathrm{DNN}}-N_{\mathrm{MPC}},
\]

正值表示 DNN 更慢。离散到达时间差为 \(\Delta t_{\mathrm{disc}}=\Delta N\Delta t\)。若阈值 \(SOC_{\mathrm{th}}=0.7995\) 位于第 \(k\) 步的两个状态之间，连续穿越时间为

\[
t_{\mathrm{cross}}=k\Delta t+\Delta t\frac{SOC_{\mathrm{th}}-SOC_k}{SOC_{k+1}-SOC_k}.
\]

连续指标只诊断 5 s 采样量化，不替代原离散 2% 门槛。"""),
    nbf.v4.new_markdown_cell("## 3. 冻结合同"),
    nbf.v4.new_code_cell("""frozen = metrics['frozen_contract']
display(pd.DataFrame([{
    'all_sources_preserved': frozen['all_sources_preserved'],
    'closed_loop_initial_states_preserved': frozen['closed_loop_initial_states_preserved'],
    'no_new_teacher_data': frozen['no_new_teacher_data'],
}]))
assert frozen['all_sources_preserved']
assert frozen['closed_loop_initial_states_preserved']
assert frozen['no_new_teacher_data']"""),
    nbf.v4.new_markdown_cell("## 4. 六种训练方案的验证集联合选择"),
    nbf.v4.new_code_cell("""display(schemes[['scheme','validation_nrmse','validation_abs_bias_a','validation_low_current_abs_bias_a','validation_rank_sum','selected']])
print('Selected:', metrics['selection']['selected_scheme'])
assert len(candidates) == 30
assert metrics['selection']['selection_data'] == 'validation_only'
assert metrics['selection']['selected_scheme'] == 'deep_32_32_16__lbfgs'
display(Image(filename=str(OUT / 'figures' / 'scheme_selection.png')))"""),
    nbf.v4.new_markdown_cell("## 5. 选定方案的双冻结测试"),
    nbf.v4.new_code_cell("""view = selected[['seed','test_nrmse','terminal_test_nrmse','terminal_test_bias_a','terminal_test_low_current_bias_a']].copy()
view['original_test_nrmse_percent'] = 100 * view.pop('test_nrmse')
view['terminal_test_nrmse_percent'] = 100 * view.pop('terminal_test_nrmse')
display(view)
assert (view.original_test_nrmse_percent < 1).all()
assert (view.terminal_test_nrmse_percent < 1).all()"""),
    nbf.v4.new_markdown_cell("## 6. 固定初态完整闭环"),
    nbf.v4.new_code_cell("""view = closed[['seed','mean_current_nrmse','mean_charge_time_gap_fraction','target_reach_fraction','maximum_voltage_violation_v','speedup']].copy()
view['current_nrmse_percent'] = 100 * view.pop('mean_current_nrmse')
view['discrete_charge_time_gap_percent'] = 100 * view.pop('mean_charge_time_gap_fraction')
display(view)
display(diagnostics)
display(Image(filename=str(OUT / 'figures' / 'selected_scheme_closed_loop.png')))
assert (view.current_nrmse_percent < 1).all()
assert (view.discrete_charge_time_gap_percent < 2).all()"""),
    nbf.v4.new_markdown_cell("## 7. 最终判定"),
    nbf.v4.new_code_cell("""display(pd.DataFrame([metrics['decision']['checks']]))
print(metrics['decision']['conclusion'])
print('进入 Level 2:', metrics['decision']['proceed_to_level2'])
assert metrics['decision']['level1s_success']
assert metrics['decision']['proceed_to_level2']"""),
    nbf.v4.new_markdown_cell("""## 结论

Level 1R 不是模型不可学习的实质失败：策略拟合、闭环电流、安全性和计算速度均已通过，剩余问题是训练随机性造成的终端到达时间一致性。Level 1S 在不改变数据和控制问题的前提下，以深层网络加 LBFGS 消除了该不稳定性，五个原种子全部通过原离散门槛。"""),
]
for index, cell in enumerate(nb.cells):
    cell["id"] = f"level1s-{index:02d}"
nbf.write(nb, PATH)
executed = nbf.read(PATH, as_version=4)
NotebookClient(executed, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
nbf.write(executed, PATH)
print(PATH)
