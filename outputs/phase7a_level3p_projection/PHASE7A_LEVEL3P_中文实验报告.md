# Phase 7A Level 3P：最小输出投影验证报告

## 结论

Level 3P 判定：**Level 3P 最小投影严格通过，研究停止在 Level 3P**。本实验明确不进入 Level 4。

## 冻结合同

- Level 3 教师数据、双冻结测试、MPC 实现、闭环初态、离线指标和五种子模型共 13 个工件哈希全部匹配。
- 未新增教师数据，未重新训练网络，未改变 2RC 模型、MPC、初态、五个种子或验收门槛。
- 唯一变化为将 DNN 原始输出裁剪到 `[max(0,I_previous-2), min(10,I_previous+2)]`。

## 投影介入

- 冻结 Level 3 原始斜率风险动作：48 步。
- Level 3P 实际介入：48 步，占全部动作 0.3596%。
- 精确位置重合：48 步；位于原风险动作 ±1 步内：48 步。
- ±1 步邻域外新增介入：0 步。
- 最大投影修正：0.132667 A。

## 五种子闭环

- 电流 NRMSE：0.2282%–0.3382%。
- 最大平均充电时间偏差：0.8690%。
- 最低目标到达率：100.0%。
- 最大电压违约：0.000000e+00 V。
- 最大斜率违约：4.440892e-16 A；最大单步变化 2.000000 A。
- 最低在线加速：626.3×。

## 严格门槛

```json
{
  "all_frozen_hashes_match": true,
  "frozen_raw_violation_count_is_48": true,
  "zero_slew_violation": true,
  "all_seed_current_nrmse": true,
  "all_seed_charge_time_gap": true,
  "all_seed_target_reach": true,
  "voltage_constraint_satisfied": true,
  "current_bounds_satisfied": true,
  "all_interventions_near_frozen_raw_violations": true,
  "all_seed_speedup": true
}
```

Level 3P 只验证最小安全投影能否修复 Level 3 已确认的硬斜率失效，不包含温度、DFN、扰动或 Level 4 内容。
