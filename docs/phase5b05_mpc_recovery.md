# Phase 5B-0.5：MPC 恢复与可行性复核

## 目的

本阶段不进入 Phase 5B-1，也不立即重跑全部 69 个场景。它先修复 Phase 5B-0
暴露的 fallback 斜率违约和可行序列丢失问题，再在代表场景上判断是否值得全域复核。

## 斜率安全动作区间

对上一控制电流 (I_{k-1})，正常动作必须位于

\[
\mathcal I_k=
[I_{k-1}-2,I_{k-1}+2]\cap[0,10]
=\left[\max(0,I_{k-1}-2),\min(10,I_{k-1}+2)\right]\ \mathrm{A}.
\]

所有移位序列、ANN 候选、保守下降和普通 emergency fallback 均保持在该区间。
如果区间内不存在满足下一步 4.20 V 和 35 ℃硬安全的电流，但区间下方存在硬安全
电流，则执行硬安全优先动作并记录 `hard_safety_slew_conflict`。它不再计作普通数值失败。

## 候选优先级

SLSQP 没有同时满足“成功退出且预测序列可行”时，依次审计：

1. `shifted_previous_feasible`：按已执行步数移位并补尾的上一可行序列；
2. `projected_ann_sequence`：ANN 分块候选逐块投影到电流/斜率区间；
3. `conservative_slew_down`：每个控制块最多下降 2 A 的保守序列；
4. `slope_safe_emergency`：预测域没有可行候选时的一步斜率安全硬约束动作；
5. `hard_safety_emergency`：只有硬安全与斜率真正冲突时使用。

选中的完整可行序列会被保存，供下一次求解移位复用。

## 失败分类

- `numerical_optimization_failure_feasible_alternative`：SLSQP 失败，但前三类候选中存在
  完整预测可行序列；
- `prediction_domain_infeasible_under_candidate_audit`：SLSQP 与三类完整候选都不可行，
  但存在斜率安全的一步硬安全动作；这是候选审计下的预测域不可行，不能当作数学上的
  全局不可行证明；
- `hard_safety_slew_conflict`：斜率安全区间内不存在硬安全动作，必须在硬安全与斜率间
  做显式冲突处理。

## 代表场景

运行器从 Phase 5B-0 冻结表中选取并去重：

- 5 个当前名义教师可行场景；
- 5 个 unresolved 场景；
- 5 个教师与 ANN 都失败场景；
- nominal、热极端和冷极端场景。

名义恢复 MPC 与参数 oracle 恢复 MPC 都会运行。代表场景通过只允许进入“全 69 场景
恢复复核”，不能直接授权 Phase 5B-1。

## 正式运行

该实验耗时较长，应由用户在本地运行：

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
$env:PYBAMM_DISABLE_TELEMETRY='true'
python -m battery_fast_charge.phase5b05_cli `
  --config configs\phase5b05_mpc_recovery.yaml `
  --project-root .
```

结果逐控制器–场景写入 `data/phase5b05_mpc_recovery/recovery_run_summary.csv`，中断后
再次执行同一命令会跳过已完成键。完成后请返回控制台输出，以及该汇总 CSV 和
`outputs/metrics/phase5b05_metrics.json`。随后再生成已执行结果 notebook。
