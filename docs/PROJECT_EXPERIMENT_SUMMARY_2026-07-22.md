# 动力电池 AI 充电控制项目：完整实验与结果总结

更新时间：2026-07-22
项目目录：`C:\Users\LENOVO\Documents\动力电池AI`

## 1. 项目目标与研究边界

本项目研究在动力电池快速充电过程中，能否利用人工神经网络（ANN）降低约束模型预测控制（MPC）的在线计算量，同时保持充电性能以及电流、电压、温度和电流变化率约束安全。

研究对象和固定边界如下：

- 电池参数集：PyBaMM Chen2020。
- 高保真虚拟电池：DFN 电化学模型＋集总热模型。
- 标称容量：5 Ah。
- 充电范围：10%–80% SOC。
- 名义环境温度：25 ℃。
- 物理电压上限：4.20 V。
- 研究性平均温度上限：35 ℃。
- 最大充电电流：10 A。
- 控制周期：5 s。
- MPC 预测时域：300 s。
- 当前全部结果均为仿真结果，不构成真实电芯安全认证或可直接部署的 BMS 控制器。

项目最初希望证明：

$$
\text{ANN/DNN}\approx\text{MPC},\qquad
\text{并直接替代在线 MPC}
$$

经过多阶段实验，研究问题已经细化为：

1. 在什么模型和约束复杂度下，pure ANN 可以直接逼近并替代 MPC？
2. 哪些因素会破坏状态到 MPC 第一动作的确定映射？
3. 当直接替代不成立时，ANN 能否作为初值、参考或结构预测器加速 MPC？

## 2. 当前总体结论

本项目目前得到的最重要结论是：

> 在低维、状态充分、教师确定、约束较简单且训练数据覆盖完整的控制问题中，ANN 可以准确逼近 MPC；在包含双极化、电热耦合、硬斜率约束、多约束切换和求解器分支的复杂控制问题中，静态 pure ANN 直接回归 MPC 第一动作不再可靠，此时更适合采用 ANN 辅助 MPC。

简化表示为：

$$
\begin{aligned}
\text{简单控制问题}
&\rightarrow \text{ANN 可直接替代 MPC},\\
\text{复杂多约束问题}
&\rightarrow \text{ANN 提供初值/参考，MPC 负责安全修正}.
\end{aligned}
$$

现有证据同时表明：

- 不能把此前失败简单归因于 Chen2020 电池参数选错。
- 不能归因于 DNN/MPC 数据管线整体错误。
- 网络容量和总样本数增加不能自动解决控制律分支和标签歧义。
- 训练数据覆盖，特别是目标 SOC 附近的末端降流覆盖，对闭环结果至关重要。
- 输出投影和安全过滤器可以恢复部分可行性，但不能代替策略本体的准确学习。

## 3. 项目实验阶段总表

| 阶段 | 核心问题 | 主要结果 | 判定 |
|---|---|---|---|
| Phase 1 | 建立 Chen2020 DFN 充电基线 | 1C/1.5C/2C 均达到 80% SOC，但均超过 35 ℃ | 完成 |
| Phase 2 | 辨识 MPC 降阶模型 | 2RC 电压 RMSE 23.01 mV；平均温度 RMSE 0.318 ℃ | 初版通过，有边界 |
| Phase 3 | 约束 MPC 能否安全闭环 | 降阶和 DFN 均 53.58 min 到达目标，无 fallback | 通过 |
| Phase 3B | MPC 教师数据质量 | 168/168 标签接受，按轨迹隔离 | 通过 |
| Phase 4A | 小型 ANN 模仿 MPC | 离线准确，但安全层介入 50.3%，时间偏差 3.89% | 工程可行，非 pure ANN |
| Phase 4B | 改进热预算教师 | 52.67 min，比安全 1C 改善 1.25% | 通过 |
| Phase 4B-2 | 主动数据聚合 ANN v2 | 时间偏差 0.16%，安全层实质介入 1.58% | 通过 |
| Phase 5A | 多温度/参数压力测试 | 69 场景完成率 68.12%，30 ℃ DFN 失败 | 未通过 |
| Phase 6A | 论文式 pure DNN 迁移 | 离线 5.63%，DFN 闭环 7.73%，斜率违约 | 未通过 |
| Phase 6B | 数据、容量、投影诊断 | pure 5.23%，projected 7.02%；投影修约束不修拟合 | 未通过 |
| Phase 6C | 约束分区与数据增广 | 45 次消融＋6400 新样本，所有方法 0/5 通过 | 未通过 |
| Phase 6R | 修正教师时序 | 离线平均 1.257%，闭环约 2.6%，仍未过门槛 | 未通过但显著改善 |
| Phase 5B-0 | MPC 压力域可行性 | 名义 MPC 仅 5/69 完整可行 | 压力域不适合直接模仿 |
| Phase 5B-0.5 | Recovery 初次复核 | 初始结果受回放合同不一致影响 | 结果作废/修正 |
| Phase 5B-0.6 | 严格配对合同审计 | 已知可行场景原始/Recovery 均 5/5 | 无回归通过 |
| 修正 15 场景复评 | Recovery 能否扩大可行域 | 合计仍 5/15，未恢复 unresolved 场景 | 恢复能力失败 |
| Phase 6P-0 | 论文 NDC 阳性对照 | 离线 0.0369%，闭环 0.0326%，30/30 到达 | 通过 |
| Phase 2R | 模型与状态充分性 | 固定 2RC 暂时可用；五状态局部标准差 0.502 A | 状态不充分 |
| Phase 2R-C | 原生控制记忆 | 方差下降 32.2%，但仍未过局部门槛 | 记忆有效但不足 |
| Phase 2R-D | 多起点最终判别 | 57% 状态存在近最优第一动作多值性 | 停止完整系统 pure DNN |
| Phase 7A Level 1 | 自有参数 1RC 两状态验证 | 离线通过，因末端数据缺失闭环约 9% | 覆盖受限 |
| Phase 7A Level 1R | 修复末端覆盖 | 闭环 0.281%–0.315%，仅时间稳定性未全过 | 接近通过，仍停 Level 1 |
| Phase 7A Level 1S | 训练稳定性消融 | 深层 LBFGS 五种子严格通过，时间偏差不超过 0.565% | 通过，可进入 Level 2 |

## 4. Phase 1：Chen2020 高保真基准充电

### 4.1 目的

建立可重复运行的 Chen2020 DFN＋热模型虚拟电池，并生成 CC–CV 基准。

### 4.2 结果

| 工况 | 达到 80% SOC | 充电时间 | 最高电压 | 最高温度 | 最大电流 |
|---|---:|---:|---:|---:|---:|
| 1C | 是 | 42.81 min | 4.20 V | 37.51 ℃ | 5.00 A |
| 1.5C | 是 | 31.05 min | 4.20 V | 46.50 ℃ | 7.50 A |
| 2C | 是 | 26.47 min | 4.20 V | 54.26 ℃ | 10.00 A |

### 4.3 结论

- 即使 1C 也超过 35 ℃研究性温度上限。
- 更高倍率虽然缩短充电时间，但带来明显温升。
- CC–CV 曲线只能作为基线，不能称为满足本项目全部约束的最优方案。

主要文件：

- `outputs/phase1_report.md`

## 5. Phase 2：2RC＋双节点热模型辨识

### 5.1 电气参数

| 参数 | 数值 |
|---|---:|
| $R_0$ | 0.021341 Ω |
| $R_1$ | 0.009118 Ω |
| $\tau_1$ | 17.24 s |
| $C_1$ | 1891.10 F |
| $R_2$ | 0.015567 Ω |
| $\tau_2$ | 150.51 s |
| $C_2$ | 9668.18 F |

### 5.2 热参数

| 参数 | 数值 |
|---|---:|
| 总热容量 | 71.15 J/K |
| 核心—表面热阻 | 0.0100 K/W |
| 表面—环境热阻 | 10.9292 K/W |
| 热增益 | 1.0991 |

### 5.3 独立验证

- 电压 RMSE：23.01 mV。
- 电压最大绝对误差：59.30 mV。
- 平均温度 RMSE：0.318 ℃。
- 平均温度最大绝对误差：0.913 ℃。

### 5.4 边界

- Chen2020 集总热模型只提供体积平均温度。
- 核心温度和表面温度没有独立空间真值。
- 核心—表面热阻命中辨识下限，双节点结构接近集总热模型。
- Phase 2R 后续发现 300 s 最大电压误差约 52 mV，几乎消耗完 4.20 V 与 4.14 V 之间的 60 mV 裕量。

主要文件：

- `outputs/phase2_report.md`
- `outputs/metrics/phase2_validation_metrics.json`
- `outputs/metrics/phase2_identified_parameters.json`

## 6. Phase 3/3B：约束 MPC 与教师数据

### 6.1 MPC 约束

- 电流：0–10 A。
- 物理电压上限：4.20 V。
- MPC 内部电压上限：4.14 V。
- 物理平均温度上限：35 ℃。
- MPC 内部平均温度上限：33.5 ℃。
- 最大电流变化：2 A/5 s。

### 6.2 名义闭环结果

| 被控对象 | 到达 80% | 时间 | 最高电压 | 最高平均温度 | 优化成功率 | fallback |
|---|---:|---:|---:|---:|---:|---:|
| 降阶模型 | 是 | 53.58 min | 4.1400 V | 33.500 ℃ | 100% | 0 |
| Chen2020 DFN | 是 | 53.58 min | 4.1425 V | 33.517 ℃ | 100% | 0 |

### 6.3 教师数据

- 候选状态：168。
- 接受标签：168。
- 拒绝：0。
- 训练/验证/测试：112/28/28。
- 按整条轨迹隔离。
- 活跃约束样本：电压 48、温度 144、斜率 16。
- 平均教师求解时间约 496.5 ms。

### 6.4 结论

- 名义 MPC 在降阶模型和 DFN 上均可行。
- MPC 比安全过滤后的 1C 基线慢约 0.25 min，不能声称已经改善充电时间。
- 只有优化成功、预测可行且未使用 fallback 的样本可以进入监督学习数据。

主要文件：

- `outputs/phase3_report.md`
- `outputs/phase3b_report.md`
- `outputs/metrics/phase3_mpc_metrics.json`
- `outputs/metrics/phase3b_metrics.json`

## 7. Phase 4A：小型 ANN 模仿 MPC

### 7.1 网络

$$
5-8-8-1
$$

- 参数量：129。
- 测试 RMSE：0.0691 A。
- 测试最大误差：0.3440 A。
- 测试 $R^2=0.997787$。

### 7.2 DFN 闭环

| 控制器 | 充电时间 | 最高电压 | 最高温度 | 结果 |
|---|---:|---:|---:|---|
| 安全过滤 1C | 53.33 min | 4.1425 V | 33.502 ℃ | 通过 |
| MPC | 53.58 min | 4.1425 V | 33.517 ℃ | 通过 |
| Tiny ANN＋安全过滤 | 55.67 min | 4.1425 V | 33.501 ℃ | 通过 |

- ANN 相对 MPC 时间偏差：3.89%。
- 安全过滤器介入：336 次，占 50.30%。
- 推理相对 MPC 加速约 5831 倍。

### 7.3 结论

ANN 离线误差很低，但闭环安全主要依赖外部安全过滤器，不能将该结果解释为裸 ANN 已经替代 MPC。

主要文件：

- `outputs/phase4a_report.md`
- `outputs/metrics/phase4a_metrics.json`

## 8. Phase 4B/4B-2：热预算教师与主动数据聚合

### 8.1 热预算混合教师

控制阶段包括：

1. 启动参考调节器。
2. 热预算 MPC。
3. 终端参考调节器。

结果：

- DFN 充电时间：52.67 min。
- 相对安全 1C 改善：1.25%。
- 最高电压：4.1425 V。
- 最高平均温度：33.502 ℃。

### 8.2 ANN v2 主动数据聚合

- 候选状态：489。
- 接受标签：486。
- 网络：$5-16-16-1$。
- 测试 RMSE：0.3679 A。
- DFN 充电时间：52.75 min。
- 相对混合教师时间差：0.16%。
- 安全过滤器实质介入率：1.58%。
- 平均过滤修正：0.0166 A。

### 8.3 结论

- 主动数据聚合明显改善了 ANN 闭环分布覆盖。
- ANN v2 在名义域内接近混合教师。
- 安全过滤器仍然保留，因此该结果属于工程混合控制，而不是 pure ANN。

主要文件：

- `outputs/phase4b_report.md`
- `outputs/phase4b2_report.md`
- `outputs/metrics/phase4b_metrics.json`
- `outputs/metrics/phase4b2_metrics.json`

## 9. Phase 5A：有界鲁棒性压力测试

### 9.1 降阶压力域

- 场景数：69。
- 完成率：68.12%。
- 物理安全率：65.22%。
- 最长时间：60 min。
- 最坏实质安全过滤介入率：55.83%。

### 9.2 DFN 温度锚点

| 环境温度 | 达到目标 | 充电时间 | 安全过滤介入率 | 结果 |
|---:|---:|---:|---:|---|
| 15 ℃ | 是 | 56.08 min | 12.93% | 通过 |
| 25 ℃ | 是 | 52.75 min | 1.58% | 通过 |
| 30 ℃ | 否 | 未完成 | 91.39% | 失败 |

### 9.3 结论

ANN v2 只在 25 ℃名义域表现良好，不能直接推广到多温度、参数扰动和状态估计误差场景。

主要文件：

- `outputs/phase5a_report.md`
- `outputs/metrics/phase5a_metrics.json`

## 10. Phase 6A：论文式 pure DNN 迁移验证

### 10.1 目标

按照 2025 年论文的思路，使用重复求解约束 MPC 生成状态—第一动作数据，再训练 pure DNN 直接输出充电电流。

### 10.2 结果

- MPC 初态接受：434/500。
- 监督样本：3472。
- 离线测试 NRMSE：5.63%。
- 25 ℃ DFN 闭环电流 NRMSE：7.73%。
- DNN 充电时间：50.25 min。
- MPC 充电时间：53.58 min。
- 时间偏差：6.22%。
- 最大单步电流变化：2.6167 A，违反 2 A/5 s 约束。
- 电压和温度没有越界。
- 推理加速约 8856 倍。

### 10.3 结论

DNN 更快到达目标不能解释为性能提升，因为它没有准确复现 MPC 电流并违反了斜率约束。按照停止规则，没有运行 15/30 ℃及压力测试。

主要文件：

- `outputs/phase6_report.md`
- `outputs/metrics/phase6_metrics.json`
- `docs/phase6_paper_method_validation.md`

## 11. Phase 6B：DNN 失败诊断

### 11.1 数据和网络

- 初态：1000。
- 可行轨迹：878。
- 样本：7024。
- 选中网络：$5-32-32-16-1$。
- 网络达到 2500 次迭代上限。

### 11.2 结果

- 离线测试 NRMSE：11.494%。
- pure DNN 闭环 NRMSE：5.228%。
- projected DNN 闭环 NRMSE：7.023%。
- pure DNN 最大斜率违约：2.581 A。
- projected DNN 斜率违约：0 A。

### 11.3 误差区域

- 上一电流 0–2 A：NRMSE 约 14.95%。
- 上一电流 2–5 A：NRMSE 约 15.14%。
- 斜率约束激活或临界区域：NRMSE 约 14.75%。

### 11.4 结论

- 输出投影能消除斜率违约，但不能修复控制律拟合。
- 网络容量和总样本增加没有自动解决问题。
- 主要困难集中在约束切换和低/中等上一电流区域。

主要文件：

- `outputs/phase6b_report.md`
- `outputs/metrics/phase6b_metrics.json`
- `docs/phase6b_dnn_failure_diagnosis.md`

## 12. Phase 6C：约束区域学习和定向数据增广

### 12.1 实验

- Phase 6C-1：完成 45 次网络容量、正则化和种子消融。
- Phase 6C-2：新增 800 条 MPC 轨迹、6400 个样本。
- Phase 6C-3：比较 pure、projected 和结构化增量 DNN。

### 12.2 五种子结果

| 控制器 | 冻结测试 NRMSE | 25 ℃ DFN 闭环 NRMSE | 时间偏差 | 严格通过 |
|---|---:|---:|---:|---:|
| pure DNN | 5.551% ± 0.086% | 4.969% ± 0.340% | 3.701% | 0/5 |
| projected DNN | 5.522% ± 0.071% | 5.179% ± 0.207% | 3.608% | 0/5 |
| 结构化增量 DNN | 7.990% ± 0.072% | 3.511% ± 0.456% | 3.795% | 0/5 |

### 12.3 结论

- 结构化增量输出保证了每步电流变化不超过 2 A，但仍未达到策略逼近和时间门槛。
- 主要问题不是单纯网络容量或 LBFGS 未收敛。
- 按停止规则不进入 Phase 6D。

主要文件：

- `outputs/phase6c_report.md`
- `docs/phase6c_constraint_regime_learning.md`
- `outputs/metrics/phase6c1_metrics.json`
- `outputs/metrics/phase6c2_metrics.json`
- `outputs/metrics/phase6c3_metrics.json`

## 13. Phase 6R：教师时序修正

### 13.1 修正

每个 5 s 状态重新求解一次滚动 MPC，只保留当前求解的第一控制动作，修复此前教师动作与状态时序可能不一致的问题。

### 13.2 数据

- 接受轨迹：222。
- 样本：1776。
- 训练/验证/测试：1232/272/272。
- 教师第一动作一致性最大误差：0 A。

### 13.3 结果

- 5状态 pure DNN 离线平均 NRMSE：1.257%，最佳 1.100%。
- 7状态 pure DNN：平均 1.990%。
- 可行区间 DNN：平均 3.092%。
- 25 ℃降阶闭环：约 2.54%–2.63%。
- 代表性 DFN 闭环：约 3.03%–3.36%。
- 充电时间偏差：约 6%–8%。

### 13.4 结论

教师时序修正显著改善了结果，但 pure DNN 仍未达到严格替代门槛。可行区间输出消除了严重越界，却没有改善策略拟合。

主要文件：

- `outputs/phase6r_report.md`
- `docs/phase6r_corrected_policy_distillation.md`
- `outputs/metrics/phase6r_nominal_validation_metrics.json`

## 14. Phase 5B-0：MPC 压力域可行性

### 14.1 结果

- 冻结场景：69。
- 名义/oracle MPC 轨迹：138。
- 名义 MPC 完整可行：5/69。
- oracle MPC 完整可行：1/69。
- 名义完成率：86.96%。
- oracle 完成率：85.51%。
- 名义物理安全率：7.25%。
- oracle 物理安全率：1.45%。

场景分类：

- 教师与 ANN 都不可行：35。
- 教师与 ANN 都可行：5。
- ANN 带安全层可行、两种无安全层教师失败：29，标记为 unresolved。
- 名义教师失败而 oracle 成功：0。

### 14.2 结论

Phase 5A 压力域不能直接作为 pure ANN 模仿学习域，因为大部分场景中教师 MPC 自身都未通过完整可行性合同。

主要文件：

- `outputs/phase5b0_report.md`
- `outputs/metrics/phase5b0_metrics.json`
- `data/phase5b_mpc_feasibility/scenario_feasibility_table.csv`

## 15. Phase 5B-0.5/0.6：Recovery 与回放合同审计

### 15.1 初次 Recovery 结果

初次代表场景复核显示 Recovery 只有 2/15 可行，但随后发现回放使用了不一致的：

- 随机种子。
- 原始场景索引。
- 噪声序列。
- 截止长度。
- 目标电流 cap 逻辑。

因此该退化结果不能作为 Recovery 失效证据。

### 15.2 严格配对审计

修正合同后，在已知可行的 5 个场景中：

- 原始 MPC：5/5 可行。
- Recovery MPC：5/5 可行。
- 电流和约束松弛仅有浮点误差差异。
- 最大电流变化均为 2 A/5 s。

### 15.3 修正合同下 15 场景复评

| 场景组 | 原始 MPC | Recovery MPC |
|---|---:|---:|
| 原始教师可行 | 5/5 | 5/5 |
| unresolved | 0/5 | 0/5 |
| 教师与 ANN 均不可行 | 0/5 | 0/5 |
| 合计 | 5/15 | 5/15 |

诊断：

- 非 emergency 候选实际使用次数：0。
- emergency fallback：1067 次。
- 预测域不可行：750 次。
- 硬安全—斜率冲突：317 次。

### 15.4 结论

Recovery 保持了已知可行域，但没有扩大可行域。后续不应继续以 ANN 直接输出为核心，而应考虑 ANN 初值/参考＋MPC 修正。

主要文件：

- `outputs/phase5b05_report.md`
- `outputs/phase5b06_report.md`
- `outputs/metrics/phase5b05_metrics.json`
- `outputs/metrics/phase5b06_metrics.json`

## 16. Phase 6P-0：论文 NDC 阳性对照

### 16.1 目的

验证论文方法在论文原始低维模型条件下能否工作，从而区分“方法/代码管线错误”和“Chen2020 迁移问题”。

### 16.2 设置

- NDC 状态：$V_s,V_b$。
- 初态：400。
- 训练样本：2000。
- 测试轨迹：30。
- 测试样本：4500。
- 网络：$2-7-5-3-1$。

### 16.3 结果

- 冻结离线 NRMSE：0.0369%。
- 30条闭环平均电流 NRMSE：0.0326%。
- 最差轨迹 NRMSE：0.2297%。
- 目标到达率：30/30。
- 最大约束违约约 0.00528 A/V。
- 本地规范结果文件中的在线加速约 78.25 倍。
- 五种子中 4/5 低于 1%。

### 16.4 关键发现

如果首个电流增量惩罚依赖未进入 DNN 输入的历史电流，则相同 $(V_s,V_b)$ 会对应不同标签，误差会升至约 3%–5%。恢复状态到动作的单值映射后，阳性对照通过。

### 16.5 结论

- 论文方法在低维、状态充分的条件下确实可以工作。
- 本项目 DNN 训练、MPC 标签和闭环验证管线没有整体错误。
- Chen2020 迁移失败必须从状态充分性、教师确定性、约束复杂度和数据覆盖中解释。

主要文件：

- `outputs/phase6p0_ndc_paper/PHASE6P0_NDC论文原位复现报告.md`
- `outputs/phase6p0_ndc_paper/metrics.json`
- `notebooks/phase6p0_ndc_paper_reproduction_results.ipynb`

## 17. Phase 2R：模型与控制状态充分性审计

### 17.1 模型审计

固定参数 2RC＋双节点热模型：

- 5 s平均电压 RMSE：18.33 mV。
- 25 s平均电压 RMSE：18.49 mV。
- 300 s平均电压 RMSE：28.72 mV。
- 300 s最大电压误差：51.82 mV。
- 留出 SOC 中没有 voltage false-safe。

SOC/温度相关模型：

- 300 s平均温度 RMSE约 1.59 ℃。
- 最大温度误差约 4.9 ℃。
- 出现 3 个电压 false-safe。
- 当前实现未通过。

### 17.2 状态充分性

- 当前五状态平均局部标准差：0.502 A。
- 最近邻标签差 P95：1.021 A。
- 上一最优序列摘要使条件方差降低 37.6%。
- 加入摘要后仍未达到 0.25 A/0.50 A 门槛。

### 17.3 结论

- 固定参数模型可以继续用于名义筛查，但不能宣称全域充分。
- 当前五状态输入未形成足够稳定、易学习的局部控制律。
- 暂不训练新的扩充输入 ANN。

主要文件：

- `outputs/phase2r_sufficiency_audit/PHASE2R模型与控制状态充分性审计报告.md`
- `outputs/phase2r_sufficiency_audit/metrics.json`

## 18. Phase 2R-C：前瞻式原生控制记忆

### 18.1 设置

在 MPC 教师数据生成过程中，求解前原生记录上一最优控制序列及其摘要，不再依赖事后重放。

### 18.2 结果

- 教师轨迹：222/240 接受。
- 样本：1776。
- 五状态基线局部标准差：0.5026 A。
- 加入原生上一序列摘要：0.4205 A。
- 条件方差下降：32.2%。
- 最近邻标签差 P95：0.8099 A。

### 18.3 结论

控制记忆确实属于缺失信息，但上一最优序列摘要仍不足以达到局部充分性门槛。

主要文件：

- `outputs/phase2rc_prospective_memory_audit/PHASE2R-C前瞻式控制记忆审计报告.md`
- `outputs/phase2rc_prospective_memory_audit/metrics.json`

## 19. Phase 2R-D：pure DNN 最终判别

### 19.1 完全相同状态多起点实验

- 状态数：100。
- 每状态 warm start：15。
- 1500 次求解全部可行。
- fallback：0。
- 近最优多值状态：57/100。
- 近最优第一动作极差中位数：约 0.058 A。
- P95：约 0.270 A。
- 最大：0.3659 A。

### 19.2 完整控制记忆

加入完整上一控制序列后，在 $K=25$ 时：

- 平均局部标准差：0.3699 A，高于 0.25 A 门槛。
- 最近邻标签差 P95：0.5613 A，高于 0.50 A 门槛。
- $K=5,10,25,50$ 均未完整通过。
- 温度单一模式通过，但电压、斜率和复合约束区域失败。

### 19.3 结论

当前完整 Chen2020 MPC 的第一动作不仅依赖可观察状态，还受到近最优解集合、warm start 和求解器分支影响。继续增加 pure DNN 输入或网络容量不再具有充分依据。

停止的是：

$$
\text{静态 pure DNN 直接模仿任意 MPC 第一动作}
$$

并未停止所有 ANN 控制研究。

主要文件：

- `outputs/phase2rd_final_discrimination/PHASE2R-D最终判别报告.md`
- `outputs/phase2rd_final_discrimination/metrics.json`

## 20. Phase 7A Level 1：自有参数 1RC 两状态消融

### 20.1 模型

$$
x_k=
\begin{bmatrix}
SOC_k & V_{p,k}
\end{bmatrix}^{T}
$$

$$
SOC_{k+1}=SOC_k+\frac{I_k\Delta t}{3600Q}
$$

$$
V_{p,k+1}=aV_{p,k}+R_1(1-a)I_k,qquad
a=\exp\left(-\frac{\Delta t}{\tau_1}\right)
$$

只保留：

- 0–10 A电流边界。
- 4.20 V端电压上限。

不包含：

- 第二极化状态。
- 硬斜率约束。
- 温度约束。
- DFN 被控对象。
- 扰动与压力域。

### 20.2 教师和离线结果

- 教师轨迹：240/240 接受。
- 样本：1920。
- 100×15 多起点全部成功。
- 多值状态比例：0%。
- 第一动作极差 P95：约 $1.54\times10^{-11}$ A。
- 五种子离线 NRMSE：0.067%–0.125%。

### 20.3 初次闭环结果

- 闭环电流 NRMSE：8.88%–9.07%。
- 充电时间差：约 38%。
- 所有种子安全到达目标。
- 无电流或电压违约。
- 最低加速约 2311 倍。

### 20.4 失败定位

- 教师最大 SOC：0.7596。
- 闭环运行至约 0.80 SOC。
- 教师没有覆盖目标附近的末端降流区域。
- 在教师覆盖范围内，闭环电流 NRMSE仍约为 0.088%–0.155%。

### 20.5 结论

Level 1 初次闭环失败属于末端状态覆盖不足，不是 1RC 模型或 ANN 不可学习。

主要文件：

- `outputs/phase7a_level1_1rc/PHASE7A_LEVEL1_中文实验报告.md`
- `outputs/phase7a_level1_1rc/metrics.json`
- `notebooks/phase7a_level1_1rc_results.ipynb`

## 21. Phase 7A Level 1R：末端覆盖修复

### 21.1 数据修复

- 原36条冻结测试轨迹哈希保持不变。
- 独立末端冻结测试集哈希保持不变。
- 新增末端轨迹：160。
- 每条轨迹：24步。
- 训练轨迹：120，其中20条加密 SOC 0.795–0.799。
- 验证轨迹：20。
- 独立末端测试轨迹：20。
- 新增末端样本：3840。
- 低电流标签（不超过0.25 A）：由99增至308。

### 21.2 教师确定性

- 160/160条轨迹接受。
- 末端100×15多起点全部成功。
- fallback：0。
- 多值状态比例：0%。
- 第一动作极差P95：0.0165 A。

### 21.3 离线结果

- 原冻结测试五种子 NRMSE：0.213%–0.291%。
- 独立末端冻结测试五种子 NRMSE：0.307%–0.343%。
- 两套冻结测试均全部低于1%。

### 21.4 同模型闭环

- 五种子电流 NRMSE：0.281%–0.315%。
- 目标到达率：100%。
- 电流违约：0。
- 电压违约：0。
- 最低加速：约973倍。
- 充电时间偏差：1.41%–3.11%。
- 五个种子中有三个超过2%时间门槛。

具体表现：

| 种子 | 平均时间偏差 | 相对 MPC 控制步数差 | 判定 |
|---:|---:|---:|---|
| 22 | 2.04% | 慢约3–4步 | 略超门槛 |
| 42 | 2.26% | 快4步 | 略超门槛 |
| 73 | 3.11% | 快5–6步 | 未通过 |
| 101 | 1.41% | 快2–3步 | 通过 |
| 137 | 1.60% | 慢2–3步 | 通过 |

### 21.5 结论

Level 1R 已经显著修复末端覆盖：闭环 NRMSE 从约9%下降至约0.3%，安全性、目标到达和在线速度全部通过。

严格合同仍判定未通过，因为要求五个种子的充电时间偏差全部低于2%。但从科学结论看，已经证明：

> 使用本项目电池参数的1RC两状态MPC可以被ANN准确学习；当前剩余问题是训练随机性引起的终端时间稳定性，而不是电池模型不可学习。

主要文件：

- `outputs/phase7a_level1r_terminal_coverage/PHASE7A_LEVEL1R_中文实验报告.md`
- `outputs/phase7a_level1r_terminal_coverage/metrics.json`
- `notebooks/phase7a_level1r_terminal_coverage_results.ipynb`
- `data/phase7a_level1r_terminal_coverage/closed_loop_metrics.csv`

## 22. Phase 7A Level 1S：训练稳定性消融与严格验收

### 22.1 冻结合同与方案选择

- 未新增教师数据。
- 原始数据、末端数据、两套冻结测试、MPC、1RC、约束和闭环初态哈希全部保持一致。
- 完成3种网络结构、2种优化器、5个随机种子，共30次候选训练。
- 仅根据验证集 NRMSE、总体绝对 bias 和低电流绝对 bias 的等权秩和选择方案，没有使用冻结测试集或闭环结果选模。
- 最终选择：$2-32-32-16-1$ 深层网络与 LBFGS 优化器。

### 22.2 冻结测试结果

- 原始冻结测试五种子 NRMSE：0.158%–0.372%。
- 末端冻结测试五种子 NRMSE：0.155%–0.284%。
- 末端测试电流 bias：$-0.00120$ 至 $+0.00050\,\mathrm{A}$。
- 低电流区 bias：$-0.00153$ 至 $+0.00438\,\mathrm{A}$。

### 22.3 同模型闭环结果

- 五种子闭环电流 NRMSE：0.167%–0.352%。
- 最大平均离散到达时间偏差：0.565%，低于2%门槛。
- 连续插值穿越时间最大偏差：0.663%，作为采样量化诊断。
- 有符号平均控制步数差 $\Delta N=N_{\mathrm{DNN}}-N_{\mathrm{MPC}}$：$-1$ 至 $+1$ 步。
- 最大平均累计电荷误差绝对值：$5.39\times10^{-5}\,\mathrm{Ah}$。
- 目标到达率：100%。
- 电流和电压违约：0。
- 最低在线加速：约1671倍。

### 22.4 结论

Level 1S 在不修改数据和控制问题的条件下消除了 Level 1R 的随机种子终端时间不稳定性。策略拟合、闭环电流、到达时间、安全性和计算速度在五个种子上均严格通过，因此 Level 1 完整证据链闭合，具备进入 Level 2 的条件。

主要文件：

- `outputs/phase7a_level1s_training_stability/PHASE7A_LEVEL1S_中文实验报告.md`
- `outputs/phase7a_level1s_training_stability/metrics.json`
- `notebooks/phase7a_level1s_training_stability_results.ipynb`
- `outputs/phase7a_level1s_training_stability/scheme_validation_summary.csv`
- `outputs/phase7a_level1s_training_stability/candidate_metrics.csv`
- `outputs/phase7a_level1s_training_stability/selected_scheme_five_seed_metrics.csv`
- `data/phase7a_level1s_training_stability/closed_loop_diagnostics_per_seed.csv`

## 23. 完整证据链

本项目当前形成了如下因果链：

$$
\begin{aligned}
\text{论文NDC复现通过}
&\Rightarrow \text{DNN/MPC管线没有整体错误},\\
\text{项目1RC Level 1S五种子严格通过}
&\Rightarrow \text{Chen2020参数并非天然不可学习},\\
\text{完整2RC电热MPC出现多起点分支}
&\Rightarrow \text{复杂控制律存在标签歧义},\\
\text{投影修复约束但不修复拟合}
&\Rightarrow \text{问题不只是输出越界},\\
\text{压力域MPC仅5/69可行}
&\Rightarrow \text{教师不可行域不能用于pure ANN模仿},\\
\text{因此}
&\Rightarrow \text{简单域用ANN，复杂域用ANN辅助MPC}.
\end{aligned}
$$

## 24. 已经证明、已经排除和尚未证明

### 24.1 已经证明

1. Chen2020 DFN、降阶模型、MPC、ANN和闭环验证管线已经打通。
2. 论文NDC低维案例可以使用DNN逼近MPC。
3. 本项目电池参数的一阶1RC两状态MPC也可以被ANN准确逼近。
4. 末端数据覆盖不足可以造成离线测试通过而完整闭环失败。
5. 在数据和控制问题冻结的条件下，深层ANN配合LBFGS可消除Level 1的多种子终端时间不稳定性，并使五种子全部严格通过。
6. 完整2RC电热MPC中存在近最优多解和warm-start分支依赖。
7. 输出投影能改善斜率约束，但不能修复策略本体误差。
8. 安全层能保持已知可行域，但没有证明能够扩大MPC可行域。
9. 多数Phase 5A压力场景中，教师MPC自身尚未通过完整可行性合同。

### 24.2 已经排除或否定

1. 不能将失败简单归因于Chen2020电池参数选错。
2. 不能将失败归因于DNN训练代码或MPC数据管线整体错误。
3. 不能认为只要增大网络就一定能解决问题。
4. 不能认为只增加总样本数而不改善状态覆盖就能解决闭环问题。
5. 不能把DNN更快到达目标自动解释为性能提升。
6. 不能把projected DNN或带安全过滤器ANN当作pure ANN。
7. 不能在教师不可行域内宣称ANN已经替代MPC。

### 24.3 尚未证明

1. 增加第二个极化状态后pure ANN是否仍能通过。
2. 硬斜率约束是否是pure ANN首次失效的明确复杂度边界。
3. 单节点温度和双节点温度分别增加多少控制复杂度。
4. 降阶ANN在Chen2020 DFN跨模型闭环中的最终适用边界。
5. ANN warm start是否能够显著降低MPC迭代次数和在线求解时间。
6. 多温度、参数扰动、老化、状态估计误差、HIL和真实电芯条件下的可部署性。

## 25. 下一阶段建议

### 25.1 进入 Level 2：只增加第二个极化状态

保持以下内容不变：

- 仍只使用电流和电压约束。
- 不加入硬斜率、温度、DFN、参数扰动或压力场景。
- 保持五种子和 Level 1 的严格验收门槛。
- 固定深层 $2-32-32-16-1$ 与 LBFGS 作为训练基线，仅按三状态输入调整输入层。

唯一新增复杂度是第二个极化状态：

$$
x_k=
\begin{bmatrix}
\mathrm{SOC}_k & V_{1,k} & V_{2,k}
\end{bmatrix}^{\mathsf T}.
$$

Level 2应从一开始覆盖目标 SOC 附近的末端降流区，并建立普通冻结测试与独立末端冻结测试。执行顺序为：

1. 生成并冻结三状态MPC教师数据。
2. 完成100×15多起点教师确定性审计。
3. 完成两套冻结离线测试。
4. 完成同模型闭环、安全性、到达时间和在线速度验证。
5. 仅在严格通过后进入硬斜率约束 Level 3。

### 25.2 继续模型—约束复杂度阶梯

建议顺序：

1. Level 2：增加第二个极化状态，只保留电流和电压约束。
2. Level 3：增加上一电流和硬斜率约束。
3. Level 4：增加单节点温度。
4. Level 5：增加双节点温度。
5. Level 6：接入Chen2020 DFN。
6. Level 7：增加多温度、参数、噪声和压力域。

每一级均遵循：

$$
\text{教师确定性审计}
\rightarrow
\text{冻结离线测试}
\rightarrow
\text{同模型闭环}
\rightarrow
\text{DFN闭环}
$$

### 25.3 复杂系统转向 ANN 辅助 MPC

推荐结构：

$$
x_k
\xrightarrow{\mathrm{ANN}}
\hat{\mathbf I}_{k:k+N}
\xrightarrow{\mathrm{MPC\ finite\ correction}}
\mathbf I_k^*
$$

ANN负责：

- MPC初始控制序列。
- 参考电流。
- 活跃约束预测。
- 可行域或风险预测。

MPC负责：

- 最终可行性。
- 电流、电压、温度和斜率约束。
- 多个近最优分支的在线选择。
- 安全修正和fallback。

后续验收指标应重点转向：

- MPC迭代次数和函数评估次数。
- 在线求解时间。
- 可行率和fallback比例。
- 目标函数相对差。
- 充电时间。
- 物理约束违约。

## 26. 可用于论文或汇报的最终研究叙事

本项目首先建立了Chen2020 DFN高保真电池、2RC＋热降阶模型和多约束MPC基线，并训练ANN模仿MPC以降低在线计算量。名义域内，主动数据聚合和安全过滤器能够使ANN接近混合教师，但多温度和参数压力测试暴露出明显的分布外与约束风险。

随后，本项目对2025年论文提出的DNN显式MPC方法进行了功能性阳性复现，证明该方法在低维、状态充分的NDC模型中能够实现低误差和显著在线加速。将该方法迁移到Chen2020电热耦合控制问题后，pure DNN未能通过严格离线、闭环和斜率约束门槛。

通过教师时序修正、模型充分性审计、原生控制记忆记录和完全相同状态的多起点求解，项目发现完整MPC在电压、温度和斜率复合约束区域存在近最优第一动作多值性及warm-start分支依赖，因此单一第一动作并不是稳定的监督目标。

进一步的层级消融表明，使用本项目电池参数的1RC两状态MPC能够被ANN准确逼近；初次闭环失败由目标SOC附近末端降流数据缺失导致，补齐覆盖后闭环电流NRMSE下降到约0.3%。随后在数据与控制合同完全冻结的条件下，深层网络配合LBFGS消除了随机种子造成的终端时间不稳定性，使五种子在策略拟合、闭环电流、到达时间、安全性和在线速度上严格通过。这说明pure ANN的成败由状态充分性、教师确定性、约束复杂度、训练域覆盖和训练稳定性共同决定，而不能简单归因于电池参数或网络容量。

因此，本项目后续将一方面继续识别pure ANN的模型—约束复杂度适用边界，另一方面在复杂电热系统中采用ANN提供初值或参考、MPC负责约束和安全修正的学习增强MPC路线。

## 27. 版本管理与当前工作区状态

本证据链准备在分支 `codex/phase7a-level1-evidence-chain` 提交。提交前基线为：

```text
405c376 feat: 完成 Phase 2R-D pure DNN 最终判别
```

本次提交范围包括：

- Level 1配置、实现、测试、数据、Notebook和报告。
- Level 1R配置、实现、测试、数据、Notebook和报告。
- Level 1S配置、实现、测试、数据、Notebook和报告。
- `docs/phase7a_hierarchical_pure_dnn_ablation_plan.md`。
- 本项目完整实验总结文档。
- `pyproject.toml`相关修改。

旧 Phase 5B 和 Phase 6B 的临时运行日志不属于本证据链，不纳入本次提交。完整测试已在提交前通过；完成提交和推送后，可在独立后续任务中开始 Level 2。

## 28. 核心文件索引

- 项目交接：`HANDOFF.md`
- Phase 7A层级计划：`docs/phase7a_hierarchical_pure_dnn_ablation_plan.md`
- Phase 6P-0论文阳性对照：`outputs/phase6p0_ndc_paper/PHASE6P0_NDC论文原位复现报告.md`
- Phase 2R模型与状态审计：`outputs/phase2r_sufficiency_audit/PHASE2R模型与控制状态充分性审计报告.md`
- Phase 2R-C原生记忆：`outputs/phase2rc_prospective_memory_audit/PHASE2R-C前瞻式控制记忆审计报告.md`
- Phase 2R-D最终判别：`outputs/phase2rd_final_discrimination/PHASE2R-D最终判别报告.md`
- Level 1报告：`outputs/phase7a_level1_1rc/PHASE7A_LEVEL1_中文实验报告.md`
- Level 1R报告：`outputs/phase7a_level1r_terminal_coverage/PHASE7A_LEVEL1R_中文实验报告.md`
- Level 1S报告：`outputs/phase7a_level1s_training_stability/PHASE7A_LEVEL1S_中文实验报告.md`
