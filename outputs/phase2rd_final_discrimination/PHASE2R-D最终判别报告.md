# Phase 2R-D：pure DNN 路线最终判别报告

## D1 相同状态多起点

- 状态数：100；每状态 warm start：15。
- 近最优第一动作多值状态：57（57.0%）。
- warm-start 敏感状态：58。
- 最大近最优第一动作极差：0.3659 A。
- 判定：存在不可忽略的近最优多值性。

## D2/D3 完整序列、邻域与模式

- 完整上一序列，K=25：局部标准差 0.3699 A；最近邻标签差 P95 0.5613 A。
- K=25 达标：False；K=5/10/25/50 全部达标：False。
- K=25 可审计模式：temperature, temperature+slew, voltage, voltage+slew, voltage+temperature。
- K=25 样本不足模式：interior, slew, voltage+temperature+slew。

## 最终决策

**停止 pure DNN 直接替代路线**

理由：相同完整状态下仍存在不可忽略的近最优第一动作多值性。 本阶段不训练 ANN；求解后活跃模式只用于诊断，不作为 pure DNN 的在线输入。
