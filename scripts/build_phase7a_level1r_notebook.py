"""填充并执行 Phase 7A Level 1R 结果 Notebook。"""

from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).parents[1]
PATH = ROOT / "notebooks" / "phase7a_level1r_terminal_coverage_results.ipynb"
nb = nbf.read(PATH, as_version=4)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.cells = [
    nbf.v4.new_markdown_cell("""# Phase 7A Level 1R：末端覆盖修复

**目标**：保持 Level 1 的 1RC、MPC、DNN 结构、五个随机种子和门槛，只修复 SOC 0.74–0.799 的末端降流标签覆盖。

本 Notebook 只读取冻结结果，不重新训练或求解。只有教师确定性、双冻结离线测试和完整同模型闭环全部通过，才允许进入 Level 2。"""),
    nbf.v4.new_markdown_cell("## 1. 读取独立 Level 1R 产物"),
    nbf.v4.new_code_cell("""from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image

ROOT = Path.cwd()
OUT = ROOT / 'outputs' / 'phase7a_level1r_terminal_coverage'
DATA = ROOT / 'data' / 'phase7a_level1r_terminal_coverage'
metrics = json.loads((OUT / 'metrics.json').read_text(encoding='utf-8'))
offline = pd.read_csv(OUT / 'dnn_offline_metrics.csv')
closed = pd.read_csv(DATA / 'closed_loop_metrics.csv')
terminal = pd.read_csv(DATA / 'terminal_teacher_dataset.csv')
tail = pd.read_csv(DATA / 'tail_training_teacher_dataset.csv')
print(metrics['study_name'], metrics['status'])"""),
    nbf.v4.new_markdown_cell("## 2. 冻结测试保护和数据覆盖合同"),
    nbf.v4.new_code_cell("""display(pd.DataFrame([metrics['frozen_test_contract']]))
display(pd.DataFrame([metrics['terminal_teacher']]))
assert metrics['frozen_test_contract']['preserved']
assert metrics['terminal_teacher']['attempted_trajectories'] == 160
assert metrics['terminal_teacher']['sample_count'] == 3840
assert metrics['terminal_teacher']['success']"""),
    nbf.v4.new_code_cell("display(Image(filename=str(OUT / 'figures' / 'terminal_coverage_repair.png')))"),
    nbf.v4.new_markdown_cell("## 3. 末端 100×15 多起点确定性审计"),
    nbf.v4.new_code_cell("""audit = metrics['terminal_teacher_audit']
display(pd.DataFrame([{
    'states': audit['state_count'],
    'warm_starts_per_state': audit['warm_starts_per_state'],
    'multivalued_fraction': audit['near_optimal_multivalued_fraction'],
    'action_range_p95_A': audit['near_optimal_first_action_range_p95_a'],
    'passed': audit['success'],
}]))
assert audit['success']"""),
    nbf.v4.new_markdown_cell("## 4. 原始与末端双冻结测试"),
    nbf.v4.new_code_cell("""view = offline[['seed','test_nrmse','terminal_test_nrmse']].copy()
view['original_test_nrmse_percent'] = 100 * view.pop('test_nrmse')
view['terminal_test_nrmse_percent'] = 100 * view.pop('terminal_test_nrmse')
display(view)
assert (view.original_test_nrmse_percent < 1).all()
assert (view.terminal_test_nrmse_percent < 1).all()"""),
    nbf.v4.new_markdown_cell("## 5. 完整同模型闭环"),
    nbf.v4.new_code_cell("""view = closed[['seed','mean_current_nrmse','mean_charge_time_gap_fraction','target_reach_fraction','maximum_voltage_violation_v','speedup']].copy()
view['current_nrmse_percent'] = 100 * view.pop('mean_current_nrmse')
view['charge_time_gap_percent'] = 100 * view.pop('mean_charge_time_gap_fraction')
display(view)
display(Image(filename=str(OUT / 'figures' / 'dnn_seed_stability.png')))"""),
    nbf.v4.new_markdown_cell("""### 闭环解释

覆盖修复把五种子闭环电流 NRMSE 从 Level 1 的约 8.9% 降至约 0.28%–0.32%，并保持目标到达、安全和高加速。但充电时间是目标阈值首次穿越的离散步数指标，五种子中仍有三个超过 2% 门槛，因此严格判定不能通过。"""),
    nbf.v4.new_markdown_cell("## 6. 阶段判定"),
    nbf.v4.new_code_cell("""display(pd.DataFrame([metrics['decision']['checks']]))
print(metrics['decision']['conclusion'])
print('进入 Level 2:', metrics['decision']['proceed_to_level2'])
assert metrics['status'] == 'completed'
assert metrics['decision']['checks']['frozen_test_preserved']
assert metrics['decision']['checks']['terminal_teacher_passed']
assert metrics['decision']['checks']['terminal_determinism_passed']
assert metrics['decision']['checks']['dual_offline_tests_passed']
assert not metrics['decision']['checks']['same_model_closed_loop_passed']
assert not metrics['decision']['proceed_to_level2']"""),
    nbf.v4.new_markdown_cell("""## 最终记录

- Level 1：严格门槛未通过；教师确定性与 pure DNN 局部逼近能力通过；失败源于末端降流区域覆盖不足。
- Level 1R：覆盖缺口已显著修复，闭环电流逼近通过，但五种子充电时间稳定性仍未达到严格门槛。
- 决策：保持停止在 Level 1，不进入 Level 2。"""),
]
nbf.write(nb, PATH)
executed = nbf.read(PATH, as_version=4)
NotebookClient(executed, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
nbf.write(executed, PATH)
print(PATH)
