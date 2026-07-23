# Phase 5B-0.6 修正合同下的 15 场景复评

<!-- canonical_feasibility_field: operational_feasible -->
<!-- recovery_operational_feasible_count: 5 -->

## 冻结合同

本次使用 Phase 5B-0 的随机种子、完整场景索引、噪声序列、初始状态、模型参数、控制更新时间、目标电流 cap 与轨迹截止规则。未训练新 ANN，也未运行完整 69 场景。

## 可行性结果

统一可行性字段为 `operational_feasible`。

| 场景组 | 原始 MPC | Recovery MPC |
|---|---:|---:|
| 原始教师可行 | 5/5 | 5/5 |
| unresolved | 0/5 | 0/5 |
| 教师与 ANN 均不可行 | 0/5 | 0/5 |
| 合计 | 5/15 | 5/15 |

## 候选恢复与失败分类

- `shifted_previous_feasible`：0 次；
- `projected_ann_sequence`：0 次；
- `conservative_slew_down`：0 次；
- emergency fallback：1067 次，不计为恢复成功；
- 预测域不可行：750 次；
- 硬安全—斜率冲突：317 次。

## 两层门槛

- 第一层无回归：通过。原始可行组 Recovery 为 5/5，电压、温度、电流和斜率均满足。
- 第二层恢复能力：失败。unresolved 组没有非 emergency 候选恢复。

## 决策

Recovery 没有扩大可行域。停止 pure ANN 完整替代与全压力域模仿路线；后续采用 ANN 提供 MPC 初值、参考电流或活跃约束预测，MPC 负责硬约束与安全修正。ANN 直接输出仅限已验证可行域。
