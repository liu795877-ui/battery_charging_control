# Phase 7B-0：Level 3P 控制器的 25 ℃ DFN 跨模型审计

## 结论

**Phase 7B-0 仅电压安全失败：进入电压感知安全层，不重新训练 ANN**

本阶段没有训练新 ANN、没有增加教师数据、没有加入温度状态或参数扰动。唯一变化是把闭环被控对象由 2RC 换成固定 25 ℃ 的 Chen2020 DFN；控制器仍使用冻结的 2RC 状态接口，其中 SOC 由 DFN 反馈校正，两个极化状态按冻结 2RC 模型传播。

## 冻结合同

- 冻结工件哈希：10/10 匹配。
- 固定网络种子：[22, 42, 73, 101, 137]。
- 初始状态：12 条。
- 采样周期：5.0 s。
- 电流/斜率投影未修改。

## DFN 上的 MPC 基线

- 目标到达率：100.0%。
- 最大端电压：4.214054 V。
- 最大电压越界：1.405428e-02 V。
- 优化器成功率：100.00%。

## 冻结 ANN＋投影五种子

- 平均电流 NRMSE 范围：0.2282%–0.3382%。
- 最大平均离散充电时间偏差：0.8690%。
- 最低目标到达率：100.0%。
- 最大 DFN 端电压：4.216401 V。
- 最大电压越界：1.640107e-02 V。
- 最大单步电流变化：2.000000 A。
- 最大电流/斜率越界：0.000e+00 A / 4.441e-16 A。
- 最低在线控制器加速：545.3×（不计 DFN 被控对象求解时间）。
- 投影介入率范围：0.2640%–0.4506%。
- 异常提前降流计数最大值：41。
- 相对 MPC 的提前降流计数最大增量：1（五种子均值差 -0.833）。
- 电流方向反转计数最大值：0。
- 连续目标穿越时间最大绝对偏差：9.572 s。

## 严格门槛

```json
{
  "all_frozen_hashes_match": true,
  "no_new_training_or_teacher_data": true,
  "mpc_target_reach_100_percent": true,
  "mpc_dfn_voltage_safe": false,
  "all_seed_current_nrmse_below_1_percent": true,
  "all_seed_charge_time_gap_below_2_percent": true,
  "all_seed_target_reach_100_percent": true,
  "ann_dfn_voltage_safe": false,
  "current_bounds_strictly_satisfied": true,
  "slew_bound_strictly_satisfied": true,
  "all_seed_speedup_above_100": true
}
```

若仅电压门槛失败，应把原因判定为跨模型状态约束失配：电流/斜率投影只保证输入约束，不能保证 DFN 端电压。下一步应优先评估电压感知安全修正，而不是重新扩大 pure ANN。
