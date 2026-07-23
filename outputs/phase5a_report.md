# 第五阶段 A 报告：有界鲁棒性验证

## 结论

阶段5A验收：未通过。

## 方法边界

- 降阶层使用名义ANN和名义安全过滤器控制参数扰动对象，并注入有偏、相关的状态估计误差；
- 场景是有界压力样本，不代表真实制造概率分布，也不能换算失效率；
- DFN层只复核15、25、30 ℃三个温度锚点。

## 降阶压力测试

- 场景数：69；完成率：68.12%；物理安全率：65.22%；
- 最长时间：60.00 min；最坏实质介入：55.83%；

| scenario_id | completion_success | physical_safe | charge_time_min | maximum_voltage_v | maximum_temperature_c | terminal_true_soc_error |
|---|---|---|---|---|---|---|
| corner_hot_resistive_optimistic | False | False | 60.0000 | 3.9406 | 34.7744 | -0.2817 |
| corner_cold_resistive | True | True | 58.1667 | 4.1703 | 24.1337 | 0.0021 |
| lhs_007 | True | False | 47.5000 | 4.1494 | 33.5363 | 0.0006 |

## Chen2020 DFN温度锚点

| anchor_temperature_c | reached_target_soc | charge_time_min | maximum_voltage_v | maximum_temperature_c | material_safety_filter_intervention_fraction | success |
|---|---|---|---|---|---|---|
| 15.0000 | True | 56.0833 | 4.1592 | 24.3631 | 0.1293 | True |
| 25.0000 | True | 52.7500 | 4.1426 | 33.5024 | 0.0158 | True |
| 30.0000 | False | nan | 3.9634 | 33.5038 | 0.9139 | False |

## 下一步

- 若本阶段未通过，应针对明确失败模式调整训练域、安全余量或状态估计方案，禁止直接进入BMS接口；
- 即使通过，仍需析锂/老化约束、观测器实现和HIL验证。
