# Phase 2R：Chen2020 模型与控制状态充分性审计

## 总结

- 2R-A 固定参数 2RC+双节点热模型：**通过本轮筛查**（筛查通过不等同于已证明全域充分）。
- 2R-A SOC/温度相关参数模型：**未通过本轮筛查**。
- 2R-B 当前五状态 DNN 输入的局部单值性：**不充分**。
- 下一步：先在新教师数据中精确记录并验证上一最优序列摘要等控制记忆，再决定 Chen2020 DNN 的扩充输入；暂不训练新 ANN。

## 审计合同

2R-A 使用 60%、65%、70%、75%、80% 初始 SOC，15/25/30 ℃，0.5/1/2 C 的 300 s DFN 脉冲，并在 5/25/300 s 统计误差。60/70/80% 用于局部参数辨识，65/75% 作为 SOC 插值留出验证。约束真假分类使用 4.2 V 与 35 ℃物理边界。

2R-B 严格重放冻结的 222 条、1776 个 Phase 6R 首动作标签。对嵌套输入集合用 25 近邻估计：

\[
\operatorname{Var}\!\left(I_k^\star\mid\phi_k\right).
\]

该估计是局部邻域诊断，不等同于解析条件分布。MPC 活跃模式和 target-cap 状态由求解结果构造，只能用于解释，不自动视为可部署的求解前输入。

## 2R-A 留出 SOC 结果

| 模型 | 时域 [s] | 平均电压 RMSE [mV] | 平均温度 RMSE [℃] | 电压 false-safe | 温度 false-safe |
|---|---:|---:|---:|---:|---:|
| fixed_2rc_dual | 5 | 18.33 | 0.0267 | 0 | 0 |
| fixed_2rc_dual | 25 | 18.49 | 0.0986 | 0 | 0 |
| fixed_2rc_dual | 300 | 28.72 | 0.5113 | 0 | 0 |
| related_2rc_dual | 5 | 18.61 | 0.0069 | 0 | 0 |
| related_2rc_dual | 25 | 20.41 | 0.0473 | 0 | 0 |
| related_2rc_dual | 300 | 29.68 | 1.5886 | 3 | 0 |
| related_2rc_single | 5 | 18.61 | 0.0067 | 0 | 0 |
| related_2rc_single | 25 | 20.41 | 0.0457 | 0 | 0 |
| related_2rc_single | 300 | 29.68 | 1.5641 | 3 | 0 |

## 2R-B 条件方差结果

| 输入集合 | 平均局部标准差 [A] | 最近邻标签差 P95 [A] | 相对当前五状态方差下降 |
|---|---:|---:|---:|
| electrothermal_4 | 0.8496 | 2.0000 | -167.3% |
| current_dnn_5_plus_previous_current | 0.5024 | 1.0208 | 0.0% |
| plus_thermal_split | 0.4937 | 1.6362 | -2.6% |
| plus_control_block_phase | 0.5882 | 1.2400 | -23.6% |
| plus_mpc_mode | 0.5296 | 1.2822 | -12.8% |
| plus_previous_plan_summary | 0.4128 | 0.7454 | 37.6% |
| plus_target_cap_state | 0.5024 | 1.0208 | 0.0% |
| all_online_or_diagnostic_variables | 0.4958 | 0.8264 | 12.3% |

严格逐点重放**未通过**：最大教师电流差为 `2.240e-01 A`，超过 `1e-6 A` 的样本数为 `1222`。差异的 P95 为 `3.011e-04 A`，超过 `0.01 A` 的比例为 `1.63%`；因此本轮只把上一最优序列摘要判为**可用于统计诊断的近似重放量**，不能把它当作精确历史真值。

## 候选变量可审计性

| 候选变量 | 状态 | 说明 |
|---|---|---|
| I_k_minus_1 | available | state_previous_current_a 已在当前五状态 DNN 中 |
| control_block_phase | recoverable_proxy | 使用 step_index mod control_block_steps；滚动 MPC 每步重算，物理含义需谨慎 |
| ambient_temperature | constant_not_identifiable | Phase 6R 教师数据全部为 25 ℃ |
| current_mpc_mode | diagnostic_only | 由求解后的活跃约束定义，不能直接作为求解前在线输入 |
| previous_optimal_sequence_summary | replayed_approximate | 记录上一控制块首项/均值/末项；CSV 状态重启在少数切换点不能严格复现 |
| identifiable_parameters | constant_not_identifiable | Phase 6R 使用同一组固定 2RC/热参数 |
| target_cap_state | replayed_diagnostic | 由预测终端 SOC 是否接近目标构造 |

## 结论边界

1. 参数相关模型在 65/75% SOC 上是插值验证，但温度仅在 15/25/30 ℃锚点上评价，不是温度留出外推。
2. Chen2020 DFN 使用集总热模型，只提供体积平均温度；双节点核心/表面温度仍没有独立真值，因此不能宣称两个内部温度状态分别得到验证。
3. 若所有温度样本都未达到 35 ℃，温度分类只能证明真可行区域，不能证明对过温的识别能力。
4. 可辨识参数和环境温度在 Phase 6R 教师集中为常量，其条件方差贡献无法由该数据识别。
5. 固定参数模型在留出 SOC 上通过预设均值门槛且没有 false-safe，但 300 s 最大电压误差达到约 52 mV，且本批样本没有真实过温阳性；所以不能据此宣称它足以覆盖完整 MPC 运行域。
6. 参数相关模型的独立热结构重放误差很小，但电热耦合预测在 15 ℃、2 C、300 s 出现明显温度偏差，并产生 3 个电压 false-safe。该结果说明当前“参数相关辨识 + 电热耦合”实现尚未形成可靠替代模型，不说明 SOC/温度相关参数思想本身无效。
7. 上一最优序列摘要使平均局部条件方差下降 37.6%，但加入后仍未达到 0.25 A / 0.50 A 两项单值性门槛；它是下一轮应精确记录并验证的必要候选信息，不是已经充分的状态扩充方案。
