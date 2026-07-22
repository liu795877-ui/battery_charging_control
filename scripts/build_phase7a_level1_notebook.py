"""填充并执行 Phase 7A Level 1 结果 Notebook。"""

from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).parents[1]
PATH = ROOT / "notebooks" / "phase7a_level1_1rc_results.ipynb"
notebook = nbf.read(PATH, as_version=4)
notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
notebook.cells = [
    nbf.v4.new_markdown_cell("""# Phase 7A Level 1：1RC pure DNN 层级复杂度消融

**问题**：在仅含 `(SOC, Vp)`、电流边界和端电压上限的项目参数 1RC 控制问题中，pure DNN 能否稳定逼近 MPC？

成功标准：教师确定性、五种子冻结离线 NRMSE `<1%`、五种子同模型闭环同时通过。Notebook 只读取冻结产物，不重新求解或训练。"""),
    nbf.v4.new_markdown_cell("""## 1. 设置与可复现性

所有数据、配置、模型与指标来自独立的 `phase7a_level1_1rc` 目录。"""),
    nbf.v4.new_code_cell("""from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image

ROOT = Path.cwd()
OUT = ROOT / 'outputs' / 'phase7a_level1_1rc'
DATA = ROOT / 'data' / 'phase7a_level1_1rc'
metrics = json.loads((OUT / 'metrics.json').read_text(encoding='utf-8'))
offline = pd.read_csv(OUT / 'dnn_offline_metrics.csv')
closed = pd.read_csv(DATA / 'closed_loop_metrics.csv')
audit = pd.read_csv(DATA / 'multistart_state_summary.csv')
print(metrics['study_name'], metrics['status'])"""),
    nbf.v4.new_markdown_cell("## 2. 教师合同与 100×15 多起点审计"),
    nbf.v4.new_code_cell("""display(pd.DataFrame([metrics['teacher']]))
display(pd.DataFrame([{
    'states': metrics['teacher_audit']['state_count'],
    'warm_starts/state': metrics['teacher_audit']['warm_starts_per_state'],
    'multivalued_fraction': metrics['teacher_audit']['near_optimal_multivalued_fraction'],
    'action_range_p95_A': metrics['teacher_audit']['near_optimal_first_action_range_p95_a'],
    'passed': metrics['teacher_audit']['success'],
}]))
assert len(audit) == 100
assert metrics['teacher_audit']['success']"""),
    nbf.v4.new_code_cell("display(Image(filename=str(OUT / 'figures' / 'teacher_and_audit.png')))"),
    nbf.v4.new_markdown_cell("## 3. 五种子冻结离线测试"),
    nbf.v4.new_code_cell("""columns = ['seed', 'test_mae_a', 'test_rmse_a', 'test_nrmse', 'test_r2']
view = offline[columns].copy()
view['test_nrmse_percent'] = 100 * view.pop('test_nrmse')
display(view)
assert len(view) == 5
assert (view.test_nrmse_percent < 1.0).all()"""),
    nbf.v4.new_markdown_cell("## 4. 25 ℃同模型闭环"),
    nbf.v4.new_code_cell("""view = closed[['seed','mean_current_nrmse','mean_charge_time_gap_fraction','target_reach_fraction','maximum_voltage_violation_v','maximum_current_violation_a','speedup']].copy()
view['mean_current_nrmse_percent'] = 100 * view.pop('mean_current_nrmse')
view['mean_charge_time_gap_percent'] = 100 * view.pop('mean_charge_time_gap_fraction')
display(view)
display(Image(filename=str(OUT / 'figures' / 'dnn_seed_stability.png')))"""),
    nbf.v4.new_markdown_cell("""## 5. 失败定位

冻结离线测试通过但闭环失败，属于计划中的情况 C。训练轨迹每条仅展开 8 个 5 s 步骤，原始状态上界为 SOC 0.75；其接受样本未覆盖闭环目标 SOC 0.80 附近。DNN 因此在末端区域发生分布外外推，虽然仍满足安全边界并到达目标，却无法复现 MPC 的末端降流策略。"""),
    nbf.v4.new_code_cell("""teacher = pd.read_csv(DATA / 'teacher_dataset.csv')
trajectories = pd.read_csv(DATA / 'closed_loop_trajectories.csv')
pd.DataFrame({
    'teacher_state_soc_max': [teacher.state_soc.max()],
    'closed_loop_state_soc_max': [trajectories.soc.max()],
    'teacher_polarization_max_V': [teacher.state_polarization_v.max()],
    'closed_loop_polarization_max_V': [trajectories.polarization_v.max()],
})"""),
    nbf.v4.new_markdown_cell("## 6. 阶段判定"),
    nbf.v4.new_code_cell("""display(pd.DataFrame([metrics['decision']['checks']]))
print(metrics['decision']['conclusion'])
print('进入 Level 2:', metrics['decision']['proceed_to_level2'])
assert metrics['status'] == 'completed'
assert metrics['decision']['checks']['teacher_determinism_passed']
assert metrics['decision']['checks']['offline_test_passed']
assert not metrics['decision']['checks']['same_model_closed_loop_passed']
assert not metrics['decision']['proceed_to_level2']"""),
    nbf.v4.new_markdown_cell("""## 下一步

依据预注册停止条件，本轮不进入 Level 2。若继续研究 Level 1，可单独注册一次仅使用训练/验证轨迹的末端覆盖增广；冻结测试集不得参与样本选择，并且该变体须标记为闭环增广 pure DNN，而非原始 Level 1。"""),
]
nbf.write(notebook, PATH)
executed = nbf.read(PATH, as_version=4)
NotebookClient(executed, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
nbf.write(executed, PATH)
print(PATH)
