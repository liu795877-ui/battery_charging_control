# Phase 5B-0：MPC 可行性边界结果

## 实验边界

本阶段复用了 Phase 5A 冻结的 69 个降阶场景、同一 60 min 截止、80% SOC 目标和
既有物理约束。没有运行 DFN 温度锚点、Phase 5A ANN 扰动重跑、跨电池验证或更换电池。

名义 MPC 使用名义辨识参数和带偏差/相关噪声的状态估计；oracle MPC 仅知道当前场景
的真实容量与电热参数，但使用完全相同的估计误差序列。两者都使用 Phase 3 的斜率感知
控制块执行规则。

教师可行性要求同时满足：

\[
\text{真实目标完成}\land\text{真实物理安全}\land
\text{优化成功率}\ge 95\%\land\text{fallback}=0.
\]

真实安全边界为 4.20 V、35 ℃、10 A，以及每 5 s 最大电流变化 2 A；真实终端 SOC
误差容差为 ±0.015。

## 总体结果

| 控制器 | 场景数 | 完成率 | 物理安全率 | 完整教师可行数 | 可行率 | 平均求解时间 | 最大求解时间 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 名义 MPC | 69 | 86.96% | 7.25% | 5 | 7.25% | 1110.6 ms | 9634.2 ms |
| oracle MPC | 69 | 85.51% | 1.45% | 1 | 1.45% | 1190.4 ms | 10075.4 ms |

两种教师都大量出现斜率违约和 MPC fallback。oracle 并未比名义 MPC 更可行，说明在
当前约束、状态估计误差和参数扰动定义下，问题不只是“ANN 不会模仿”，而是教师本身
在大部分压力场景中没有形成可接受的完整控制轨迹。oracle 只表示参数知识上界，不能
解释为真实 BMS 可获得的控制器。

## 场景分类

| 类别 | 数量 | 解释 |
|---|---:|---|
| teacher_and_ann_infeasible | 35 | 教师和 Phase 5A ANN 都未形成完整安全轨迹 |
| teacher_and_ann_feasible | 5 | 名义教师、oracle 教师和 Phase 5A ANN 均满足摘要级完成/安全条件 |
| ann_feasible_teachers_failed_unresolved | 29 | Phase 5A ANN 使用了安全层；两种无安全层 MPC 教师失败，不能直接归因于 ANN |
| nominal_teacher_failed_oracle_teacher_feasible | 0 | 未观察到名义失败而 oracle 成功的场景 |

因此，本轮没有足够证据把大多数 Phase 5A ANN 失败归因于模仿误差；同时也不能把
29 个 ANN “可行”场景计作教师可行，因为它们依赖 Phase 5A 安全层且教师基准失败。

## 约束激活

名义/oracle MPC 的平均决策激活比例分别为：电压约 35.8%/36.2%，温度约 74.9%/75.2%，
电流上限约 2.8%/2.9%，斜率约 26.0%/26.9%。温度约束是最常见的主动约束，斜率约束
是最主要的求解失败/回退关联区域。

## Phase 5B-1 决策

Phase 5B-0 证明当前 Phase 5A 扰动域对“无安全层完整 MPC 教师”过于苛刻，不能直接
进入 ANN v3 全域训练。下一步应先：

1. 使用 5 个名义教师可行场景建立可审计的教师训练/验证掩码；
2. 单独分析 35 个教师与 ANN 都不可行场景，确认是温度预算、斜率约束、状态估计误差
   还是优化器回退导致；
3. 将 29 个 unresolved 场景保留为“安全层/混合控制器”研究对象，不纳入 pure ANN
   的教师模仿结论；
4. Phase 5B-1 优先设计 ANN 候选 + 约束检查 + 有限迭代 MPC 修正，而不是训练无保护
   ANN 替代完整 MPC；
5. 在此之前不运行跨电池实验，也不把本阶段的降阶教师边界外推到 DFN 或真实电芯。

## 产物

- 场景级表：`data/phase5b_mpc_feasibility/scenario_feasibility_table.csv`
- 教师运行表：`data/phase5b_mpc_feasibility/controller_run_summary.csv`
- 教师可行掩码：`data/phase5b_mpc_feasibility/teacher_feasible_scenario_mask.csv`
- 约束激活：`data/phase5b_mpc_feasibility/teacher_constraint_activity.csv`
- 原因分类：`data/phase5b_mpc_feasibility/infeasibility_reason_counts.csv`
- 指标 JSON：`outputs/metrics/phase5b0_metrics.json`
- 可查看 notebook：`notebooks/10_phase5b0_mpc_feasibility.ipynb`
