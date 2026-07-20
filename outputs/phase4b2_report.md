# 第四阶段 B-2 报告：主动数据聚合与ANN v2

## 结论

阶段4B-2验收：通过。

## 数据与标签

- 候选状态：489；接受标签：486；教师接受率：99.39%；
- 训练/验证/测试：{'train': 364, 'validation': 64, 'test': 58}；
- 教师模式：{'terminal_reference_governor': 402, 'startup_reference_governor': 44, 'thermal_budget_mpc': 40}；
- 原168个可达状态全部由新混合教师重标；新增状态来自旧ANN周围12条受约束轨迹、降阶在策略轨迹和DFN在策略轨迹；
- 标准化和权重只拟合训练轨迹，验证轨迹选模，测试轨迹只做最终评价。

## 离线模仿

- 网络结构：[5, 16, 16, 1]；参数量：385；
- 选中L2：0.01；种子：33；在策略训练权重：3.0；
- 测试MAE：0.2015 A；RMSE：0.3679 A；最大误差：1.2007 A；

## Chen2020 DFN闭环

| controller | charge_time_min | material_intervention_fraction | mean_filter_correction_a | maximum_voltage_v | maximum_temperature_c | success |
|---|---|---|---|---|---|---|
| ANN v1 + safety filter | 55.6667 | 0.1302 | 0.0610 | 4.1425 | 33.5012 | True |
| Hybrid thermal-budget teacher | 52.6667 | 0.0000 | 0.0000 | 4.1425 | 33.5024 | True |
| ANN v2 + safety filter | 52.7500 | 0.0158 | 0.0166 | 4.1426 | 33.5024 | True |

- ANN v2实质安全过滤介入：1.58%（阈值0.10 A）；
- 平均过滤修正：0.0166 A；
- 相对混合教师时间差：0.16%；

## 边界

- 主动数据改善只在当前仿真域内成立，安全过滤器仍保留；
- 即使本轮某条轨迹零介入，也不能据此声明裸ANN可部署；
- 下一步仍需多温度、参数扰动、老化和观测器误差验证。
