# Phase 5B-0：MPC 可行性边界实施计划

## 当前状态

本文件只冻结下一阶段的实验契约，`configs/phase5b0_mpc_feasibility_envelope.yaml`
的状态为 `planned_not_run`。本次 Phase 6R 没有运行 15/30 ℃、Phase 5A 扰动或跨电池实验。

## 研究问题

在训练 ANN v3 前，先区分“控制目标本身不可行”和“ANN 模仿失败”。对 Phase 5A
冻结的 69 个降阶场景以及 15/25/30 ℃ DFN 锚点，比较：

1. 使用名义辨识参数的完整 MPC；
2. 知道场景真实参数的 oracle MPC。

两者采用同一 60 min 时限、80% SOC 目标和既有物理约束，不删除困难场景，也不降低门槛。

## 场景分类

每个场景只能归入以下可审计类别之一：

1. 教师可行、ANN 失败；
2. 教师与 ANN 都不可行；
3. 教师与 ANN 都可行；
4. 名义教师失败、oracle 教师可行。

第 1 类是 Phase 5B-1/2 的首要数据增强对象；第 2 类不得计为 ANN 模仿失败；第 4 类
用于量化参数失配造成的可行性损失。

## 冻结输出

- `scenario_feasibility_table.csv`：逐场景完成、安全、时间、最大约束值和类别；
- `teacher_constraint_activity.csv`：电压、温度、电流与斜率约束激活统计；
- `teacher_feasible_scenario_mask.csv`：后续 ANN 公平验收掩码；
- `phase5b0_metrics.json`：名义/oracle 教师完成率、安全率和时间分布；
- `phase5b0_report.md`：不可行原因及进入 Phase 5B-1 的采样边界。

## 启动前检查

- 保持 Phase 5A 的 69 个场景、三个温度锚点及阈值原封不动；
- 明确名义 MPC 与 oracle MPC 的状态、参数和可观测信息边界；
- 为长时间求解加入逐场景检查点，失败场景保存优化器状态和原因；
- 先完成降阶场景，再对关键类别运行 DFN；
- Phase 5B-0 只评价教师可行性，不训练 ANN v3。
