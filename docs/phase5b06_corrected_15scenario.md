# Phase 5B-0.6 修正合同下的 15 场景复评

## 目标

本阶段不训练新 ANN，不运行完整 69 场景。目标是在与 Phase 5B-0 完全一致的回放合同下，判断 Recovery MPC 是否：

1. 在原始教师可行场景上无安全回归；
2. 能在 unresolved 场景中通过非 emergency 候选恢复扩大可行域；
3. 能把所有失败分解为数值失败、预测域不可行或硬安全—斜率冲突。

## 冻结合同

原始 MPC 与 Recovery MPC 对每个场景共享：

- Phase 5B-0 随机种子与完整 69 场景中的原始索引；
- 相同噪声创新序列；
- 相同初始状态、真实模型参数和控制器模型参数；
- 相同控制更新时间；
- 相同目标电流 cap；
- 相同轨迹截止长度和完成判定。

最终硬安全标准保持为电压 4.2 V、温度 35 ℃、电流 0–10 A，以及每 5 s 最大电流变化 2 A。

## 场景组成

从冻结的 `representative_scenarios.csv` 自动选择：

- 5 个原始教师可行场景；
- 5 个 unresolved 场景；
- 5 个教师与 ANN 均不可行场景。

## 统一可行性合同

CSV、指标 JSON、中文报告和 notebook 均使用字段 `operational_feasible`。自动测试要求：

```text
paired_summary Recovery feasible_count
== metrics JSON recovery_feasible_count
== report recovery feasible_count
```

## 结果

| 场景组 | 原始 MPC | Recovery MPC |
|---|---:|---:|
| 原始教师可行 | 5/5 | 5/5 |
| unresolved | 0/5 | 0/5 |
| 教师与 ANN 均不可行 | 0/5 | 0/5 |
| 合计 | 5/15 | 5/15 |

Recovery 在原始可行域内没有回归，但没有扩大可行域。三类候选的实际使用次数均为 0；困难场景只有 emergency 动作，不能计为恢复成功。

## 决策

停止 pure ANN 完整替代和全压力域模仿路线。后续采用以下混合结构：

\[
\text{ANN 初值/参考/活跃约束预测}
\rightarrow
\text{约束与制动可行性检查}
\rightarrow
\text{MPC 安全修正}.
\]

ANN 只在已验证可行域内考虑直接输出；边界和域外压力场景必须交还 MPC。
