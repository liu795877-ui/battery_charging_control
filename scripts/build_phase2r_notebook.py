"""用冻结产物构建 Phase 2R 中文结果 notebook。"""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "phase2r_model_and_state_sufficiency_results.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


notebook = nbf.read(NOTEBOOK, as_version=4)
notebook["cells"] = [
    md(r"""
# Phase 2R：Chen2020 模型与控制状态充分性审计

本 notebook 只读取已冻结的审计产物，不重新运行 DFN、MPC 或 ANN 训练。

**结论：**固定参数 2RC+双节点热模型通过本轮预设筛查，但尚不能证明全运行域充分；SOC/温度相关电热耦合模型未通过；当前五状态 DNN 输入下的教师控制律未达到局部单值性门槛。暂不训练新 ANN。
"""),
    md("## 1. 加载冻结产物"),
    code("""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

ROOT = Path.cwd()
if not (ROOT / "outputs" / "phase2r_sufficiency_audit").exists():
    ROOT = ROOT.parent
OUT = ROOT / "outputs" / "phase2r_sufficiency_audit"
metrics = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
horizon = pd.read_csv(OUT / "model_horizon_metrics.csv")
thermal = pd.read_csv(OUT / "thermal_structure_metrics.csv")
state = pd.read_csv(OUT / "state_conditional_variance_metrics.csv")
availability = pd.read_csv(OUT / "candidate_variable_availability.csv")
print(f"结果目录：{OUT.resolve()}")
print(metrics["decision"])
"""),
    md(r"""
## 2. 2R-A 审计合同

- 初始 SOC：60%、65%、70%、75%、80%；其中 60/70/80% 为参数锚点，65/75% 为 SOC 插值留出。
- 温度：15/25/30 ℃；脉冲：0.5/1/2 C，持续 300 s。
- 预测时域：5/25/300 s。
- 物理约束分类边界：4.2 V、35 ℃。
- 比较固定参数/相关参数 2RC，以及单温度/双节点热结构。
"""),
    code("""
holdout = pd.DataFrame(metrics["model_audit"]["holdout_summary"])
display(holdout.style.format({
    "voltage_rmse_mv_mean": "{:.2f}", "voltage_rmse_mv_max": "{:.2f}",
    "temperature_rmse_c_mean": "{:.3f}", "temperature_rmse_c_max": "{:.3f}"
}))
assert metrics["model_audit"]["pulse_profile_count"] == 45
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name, group in holdout.groupby("variant"):
    axes[0].plot(group.horizon_s, group.voltage_rmse_mv_mean, marker="o", label=name)
    axes[1].plot(group.horizon_s, group.temperature_rmse_c_mean, marker="o", label=name)
axes[0].axhline(50, color="black", linestyle="--", linewidth=1)
axes[1].axhline(1.5, color="black", linestyle="--", linewidth=1)
axes[0].set(xlabel="预测时域 [s]", ylabel="平均电压 RMSE [mV]")
axes[1].set(xlabel="预测时域 [s]", ylabel="平均温度 RMSE [℃]")
axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
"""),
    code("""
classes = (horizon[horizon.holdout_soc]
           .groupby(["variant", "horizon_s", "voltage_classification"])
           .size().unstack(fill_value=0))
display(classes)
print("温度真实不可行样本数：", int((~horizon.true_temperature_feasible).sum()))
"""),
    md("""
固定参数模型在留出 SOC 上没有电压 false-safe，但 300 s 最大电压误差约 52 mV；本批样本没有真实过温阳性。因此“通过筛查”不等于完整 MPC 域已得到验证。相关参数模型在 300 s 出现 3 个电压 false-safe；其独立热结构拟合很好，而耦合预测明显变差，问题应定位到当前电热耦合辨识/热源传播实现。
"""),
    md("## 3. 单温度与双节点热结构"),
    code("""
thermal_holdout = (thermal[thermal.holdout_soc]
                   .groupby("thermal_structure")[["rmse_c", "maximum_absolute_error_c"]]
                   .mean().sort_values("rmse_c"))
display(thermal_holdout.style.format("{:.4f}"))
"""),
    md("""
DFN 数据只给出体积平均温度，没有独立核心/表面温度真值。因而这里能比较平均温度预测，不能证明双节点内部状态分别正确；单温度结构误差较小也不能单独证明它在约束控制中更可靠。
"""),
    md(r"""
## 4. 2R-B：控制律局部单值性

对每个输入集合标准化后取 25 个近邻，估计

\[
\operatorname{Var}(I_k^\star\mid\phi_k).
\]

门槛为平均局部标准差不超过 0.25 A，最近邻标签差 P95 不超过 0.50 A。该方法是数据邻域诊断，不是解析条件分布证明。
"""),
    code("""
cols = ["feature_set", "mean_local_standard_deviation_a",
        "nearest_neighbor_label_difference_p95_a",
        "variance_reduction_from_current_dnn_5_fraction",
        "significant_reduction_vs_current_dnn"]
display(state[cols].style.format({
    "mean_local_standard_deviation_a": "{:.3f}",
    "nearest_neighbor_label_difference_p95_a": "{:.3f}",
    "variance_reduction_from_current_dnn_5_fraction": "{:.1%}"
}))
"""),
    code("""
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.barh(state.feature_set, state.mean_local_standard_deviation_a)
ax.axvline(0.25, color="black", linestyle="--", label="单值性门槛")
ax.set(xlabel="平均局部标准差 [A]")
ax.invert_yaxis(); ax.legend(); plt.tight_layout(); plt.show()
"""),
    md("""
上一最优序列摘要带来 37.6% 的平均局部方差下降，是唯一超过 20% 判据的候选变量；但扩充后仍为 0.413 A / 0.745 A，未通过两项门槛。控制相位和事后 MPC 模式没有稳定改善，且存在维数增加引起的近邻稀疏效应，不能据此认定它们无关。
"""),
    md("## 5. 回放可信度与变量可用性"),
    code("""
display(availability)
replay = metrics["state_audit"]["replay"]
display(pd.DataFrame([{
    "严格逐点成功": replay["exact_success"],
    "差异P95 [A]": replay["p95_replay_current_difference_a"],
    ">0.01 A比例": replay["large_difference_fraction"],
    "统计诊断可用": replay["memory_summary_usable"],
    "最大差异 [A]": replay["maximum_replay_current_difference_a"],
}]).style.format({"差异P95 [A]": "{:.6f}", ">0.01 A比例": "{:.2%}", "最大差异 [A]": "{:.3f}"}))
assert not replay["exact_success"] and replay["memory_summary_usable"]
"""),
    md("""
CSV 状态舍入和 SLSQP 在约束切换点的局部解导致严格重放失败；P95 差异为约 0.30 mA，超过 10 mA 的比例为 1.63%。所以历史序列摘要只可用于本轮统计诊断，下一批教师数据必须在生成时直接记录，不能由事后重放替代。
"""),
    md("""
## 6. 阶段决策

1. **暂不训练新 ANN，也不进入 A1。**
2. 新教师数据应原生记录上一可行/最优序列摘要，并做相同邻域方差审计；若仍高，再考虑更完整的控制记忆或模式分区。
3. 修正相关参数模型的电热耦合后，复核 15 ℃、2 C、300 s 与 3 个电压 false-safe；补充可产生真实过温阳性的受控测试，才能评价温度约束分类。
4. 环境温度和可辨识参数在当前 Phase 6R 数据中为常量，其必要性不可识别；需要跨温度/参数数据后再审计。

这将 Chen2020 的优先问题从“基础 DNN 管线错误”进一步收敛为：**控制记忆缺失导致的标签非单值性，以及降阶模型在长时域/约束边界的电热耦合误差。**
"""),
]
notebook["metadata"].setdefault("kernelspec", {"display_name": "Python 3", "language": "python", "name": "python3"})
nbf.write(notebook, NOTEBOOK)
print(NOTEBOOK)
