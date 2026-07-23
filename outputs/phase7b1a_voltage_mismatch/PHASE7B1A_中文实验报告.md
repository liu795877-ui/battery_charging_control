# Phase 7B-1A：25 ℃ DFN 电压失配与制动可行性审计

## 问题与符号

充电电流取正，电压单位为 V，电流单位为 A，采样周期为 5 s。冻结轨迹中每个控制步的电压残差定义为：

\[
e_{V,k+1}
=
V_{\mathrm{DFN},k+1}
-
\hat V_{\mathrm{2RC},k+1}(I_k).
\]

安全层在决策时刻只使用上一时刻已经测得的残差，并加入冻结回归轨迹中的最大正向一步增长：

\[
\Delta V_{\mathrm{guard}}
=
\max_k\max(e_{V,k+1}-e_{V,k},0).
\]

## 冻结合同

- Phase 7B-0 工件哈希：4/4 匹配。
- 审计样本：16024 个控制步，72 条闭环。
- 独立确认集：24 个新初态；它们不是 ANN 教师数据，且与 12 个回归初态无重合。

## 电压残差

- 最大正向残差：29.117 mV。
- 正向残差 P95 / P99：27.093 / 28.672 mV。
- 一步正向增长最大值：11.306 mV。
- 一步正向增长 P95 / P99：2.143 / 5.444 mV。
- 固定裕量诊断基线建议值（P99残差＋最大增长）：39.978 mV。

## 电压—斜率可行性

在每一步求满足修正后下一步电压不超过 4.2 V 的最大电流：

\[
I_{V,k}^{\max}
=
\max\left\{
I\in[0,10]:
\hat V_{k+1}^{\mathrm{2RC}}(I)
+e_{V,k}
+\Delta V_{\mathrm{guard}}
\leq 4.2
\right\}.
\]

并与斜率下界

\[
I_k^-=\max(0,I_{k-1}-2)
\]

比较。结果：

- 电压—斜率空区间：0 次。
- 最小可行性裕量：0.840223 A。
- 4.15、4.18、4.19 V 三个提前阈值中“最大斜率制动仍来不及”：0 次。
- 已经进入 4.20 V 后才检测到的事后时刻：72 次；该项只说明 4.20 V 不能作为提前触发阈值，不用于否决一步残差修正。
- 4.15 / 4.18 / 4.19 / 4.20 V 的首次进入时刻已逐轨迹保存在 `threshold_timing.csv`。

## 判定

**7B-1A 通过：最大残差增长裕量下的一步电压限制始终与 2 A/步斜率约束相容，可进入 7B-1B 一步残差修正**

```json
{
  "all_frozen_hashes_match": true,
  "confirmation_set_frozen_before_safety_layer": true,
  "confirmation_set_is_independent": true,
  "zero_voltage_slew_empty_intervals": true,
  "maximum_slew_braking_is_never_late_before_voltage_limit": true
}
```

本阶段只决定一步残差修正是否具有斜率可行性，不宣称安全层已经通过 DFN 闭环。独立确认集在安全层参数冻结后才可用于最终验收。
