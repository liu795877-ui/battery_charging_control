# 动力电池 ANN–MPC 快速充电项目交接文档

更新时间：2026-07-23

项目目录：`C:\Users\LENOVO\Documents\动力电池AI`

当前分支：`codex/phase7b1-voltage-safety-layer`

当前分支基线：`68f7aa8`（Phase 7B-0 PR #5 已合并到 `main`）

工作区状态：**Phase 7B-1A 已完成并通过。一步残差修正下的电压—斜率区间在 16,024 个冻结控制步中从未为空，可以进入 7B-1B；独立 24 初态确认集已在安全层实现前冻结。**

---

## 0. 新对话先读这里

本项目研究：

> 使用 ANN 学习受约束 MPC 的动力电池快速充电策略，以降低 BMS 在线计算量，同时保持电流、电压、充电时间和安全约束。

当前最重要的结论是：

1. 1RC 和 2RC 简单电模型中，pure DNN 可以稳定逼近 MPC。
2. 加入上一时刻电流和硬电流斜率约束后，pure DNN 的离线与闭环精度仍然通过，但无法天然保证每一步严格满足硬约束。
3. Level 3P 增加一个解析输出投影，仅修正原有 48 个风险动作，占全部动作的 0.3596%，即可严格消除斜率违约，同时基本不损害闭环性能。
4. Phase 7A **停止在 Level 3P，不进入 Level 4**。Level 4 的温度状态、温度约束和参数相关性本轮明确不运行。
5. 下一步优先级不是训练新网络，而是：
   - 先提交、推送 Level 3 与 Level 3P 的完整证据链；
   - 再新开独立阶段，冻结 Level 3P 控制器，做 25 ℃ DFN 跨模型闭环审计。

不要把最新结论错误概括为“ANN不能替代MPC”。准确表述是：

> 无保护的 pure DNN 可以高精度模仿含斜率约束的 MPC 策略，但不能提供逐步硬约束保证；ANN 加最小解析投影在当前 2RC 同模型域内严格通过，并可替代在线 MPC 求解。

---

## 1. 研究任务和技术架构

### 1.1 长期目标

最终目标是形成适合 BMS 实时部署的快速充电控制器：

$$
\text{电池状态估计}
\rightarrow
\text{ANN 快速控制决策}
\rightarrow
\text{轻量安全保证}
\rightarrow
\text{电池}
$$

当前全部工作仍属于仿真研究，尚未完成 HIL、嵌入式部署或真实电芯实验，不能声称已经获得可直接上车的 BMS 控制器。

### 1.2 电池模型、MPC 与 ANN 的分工

| 模块 | 作用 | 当前定位 |
|---|---|---|
| DFN 高保真模型 | 模拟更真实的电化学动态 | 虚拟电池与最终跨模型验证对象 |
| 1RC/2RC 降阶模型 | 以较低计算量预测 SOC、电压等状态 | MPC 在线预测模型 |
| MPC | 在模型和约束下滚动求解最优电流 | 离线教师与性能基准 |
| ANN/DNN | 学习 MPC 的状态—第一动作映射 | 在线快速策略 |
| 解析投影/安全层 | 对 ANN 输出提供硬约束保证 | Level 3P 已验证电流与斜率投影 |

ANN 当前主要替代的是 MPC 的**在线重复优化求解**，不是替代电池模型、状态估计或全部安全逻辑。

### 1.3 当前验收合同

主要严格门槛：

| 指标 | 门槛 |
|---|---:|
| 离线冻结测试 NRMSE | 小于 1% |
| 同模型闭环电流 NRMSE | 小于 1% |
| 平均充电时间偏差 | 小于 2% |
| 目标到达率 | 100% |
| 电流、电压和硬斜率严重违约 | 0 |
| 在线加速 | 大于 100 倍 |

只有这些指标同时通过，才可以说当前层级严格通过。

---

## 2. 已完成实验路线

## 2.1 Phase 1–4：基础链路

已完成：

- Chen2020 电池参数与 CC–CV 基线；
- DFN 高保真虚拟电池；
- 1RC/2RC 与电热降阶模型；
- 受约束 MPC；
- MPC 教师轨迹；
- 小型 ANN 模仿；
- ANN 安全层与 DFN 闭环。

Phase 4B-2 在名义域内表明“ANN候选＋安全层”具有工程可行性，但不能证明 pure ANN 可完全替代 MPC。

主要入口：

- `docs/phase2_model.md`
- `docs/phase3_mpc.md`
- `docs/phase3b_teacher_data.md`
- `docs/phase4_tiny_ann.md`
- `docs/phase4b2_active_learning.md`

## 2.2 Phase 5：压力域和 MPC 可行性

Phase 5A 在温度、参数扰动和 DFN 压力场中暴露了名义域外失效。

Phase 5B 系列进一步证明：

- 复杂压力域中 MPC 教师本身大量不可行；
- Recovery 可以保持已知可行域，但没有扩大可行域；
- 不能把教师不可行场景直接当作 pure ANN 模仿学习域；
- 回放合同若不一致，会制造虚假的控制器回归结论。

关键教训见本文件“踩坑”章节。

主要入口：

- `outputs/phase5a_report.md`
- `outputs/phase5b0_report.md`
- `outputs/phase5b06_report.md`
- `docs/phase5b0_mpc_feasibility_envelope.md`
- `docs/phase5b06_corrected_15scenario.md`

## 2.3 Phase 6：论文方法迁移与失败根因诊断

### Phase 6A/6B/6C/6R

直接把论文式 pure DNN 迁移到 Chen2020 复杂控制问题时没有严格通过。

已排查：

- 输出投影；
- 网络容量；
- 优化器；
- 数据增广；
- 结构化增量输出；
- 教师时序一致性；
- 控制记忆；
- 约束区域覆盖。

修正教师时序后结果显著改善，但仍未通过严格门槛。

### Phase 6P-0：NDC 阳性对照

NDC 功能性原位复现通过：

- 冻结测试 NRMSE：0.0369%；
- 30 条闭环平均电流 NRMSE：0.0326%；
- 目标到达：30/30；
- 在线加速：96.6 倍。

这排除了“基础 ANN 管线整体错误”。

### Phase 2R/2R-C/2R-D：控制律可学习性审计

最终发现：

- 控制器记忆是缺失信息之一；
- Chen2020 复杂 MPC 在部分区域存在近最优第一动作多值性；
- 完整控制序列仍不能让局部单值性全面通过；
- pure DNN 直接替代复杂 MPC 缺少充分依据。

主要入口：

- `outputs/phase6r_report.md`
- `outputs/phase6p0_ndc_paper/PHASE6P0_NDC论文原位复现报告.md`
- `outputs/phase2r_sufficiency_audit/PHASE2R模型与控制状态充分性审计报告.md`
- `outputs/phase2rc_prospective_memory_audit/PHASE2R-C前瞻式控制记忆审计报告.md`
- `outputs/phase2rd_final_discrimination/PHASE2R-D最终判别报告.md`

---

## 3. Phase 7A：逐级增加复杂度

Phase 7A 的目的不是继续盲目扩大网络，而是每次只增加一个复杂因素，定位 pure DNN 首次失效点。

实验阶梯：

$$
\text{1RC}
\rightarrow
\text{2RC}
\rightarrow
\text{上一电流与硬斜率}
\rightarrow
\text{电热约束}
\rightarrow
\text{DFN/扰动}
$$

本轮实际停止在第三层的投影修复，不进入电热 Level 4。

## 3.1 Level 1：1RC 初始实验

教师确定、离线测试通过，但闭环失败：

- 闭环电流 NRMSE 约 9%；
- 充电时间偏差约 38%。

根因是教师样本最大 SOC 仅约 0.7596，没有覆盖目标 SOC 0.80 附近的末端降流区域。

## 3.2 Level 1R：修复末端覆盖

增加末端轨迹后：

- 闭环电流 NRMSE：0.281%–0.315%；
- 安全与到达率通过；
- 部分种子的充电时间偏差仍超过 2%。

## 3.3 Level 1S：训练稳定性

在不修改数据、MPC和控制问题的情况下完成 3 种结构 × 2 种优化器 × 5 种子，共 30 次训练。

选择深层网络加 LBFGS 后严格通过：

- 原始冻结测试 NRMSE：0.158%–0.372%；
- 末端冻结测试 NRMSE：0.155%–0.284%；
- 闭环电流 NRMSE：0.167%–0.352%；
- 最大平均到达时间偏差：0.565%；
- 目标到达率：100%；
- 电流、电压违约：0；
- 最低加速：约 1671 倍。

主要入口：

- `outputs/phase7a_level1s_training_stability/PHASE7A_LEVEL1S_中文实验报告.md`
- `outputs/phase7a_level1s_training_stability/metrics.json`

## 3.4 Level 2：2RC

状态：

$$
x_k=
\begin{bmatrix}
SOC_k & V_{1,k} & V_{2,k}
\end{bmatrix}^{\mathsf T}
$$

仍然只有电流和端电压约束。

严格通过：

- 教师接受：399/400；
- 100×15 多起点全部成功；
- 多值状态比例：0%；
- 第一动作极差 P95：0.0169 A；
- 全域冻结测试 NRMSE：0.192%–0.274%；
- 末端冻结测试 NRMSE：0.205%–0.462%；
- 闭环电流 NRMSE：0.148%–0.366%；
- 最大时间偏差：1.343%；
- 目标到达率：100%；
- 电流、电压违约：0；
- 最低加速：约 1342 倍。

结论：第二个极化状态不是失效点。

主要入口：

- `outputs/phase7a_level2_2rc/PHASE7A_LEVEL2_中文实验报告.md`
- `outputs/phase7a_level2_2rc/metrics.json`

---

## 4. 当前最新结果：Level 3 与 Level 3P

## 4.1 Level 3：pure DNN 首次严格失败

Level 3 相对 Level 2 只增加：

1. ANN 输入中的上一时刻电流；
2. 每 5 s 最大 2 A 的硬电流斜率约束。

状态：

$$
x_k=
\begin{bmatrix}
SOC_k & V_{1,k} & V_{2,k} & I_{k-1}
\end{bmatrix}^{\mathsf T}
$$

硬约束：

$$
\left|I_k-I_{k-1}\right|
\leq
2\ \mathrm{A}/5\mathrm{s}
$$

本层级不包含温度、DFN、参数扰动或 Phase 5A 压力场。

结果：

- 教师轨迹：400/400 接受；
- 教师样本：5760；
- 低电流标签：183；
- 100×15 审计全部成功；
- 多值状态比例：0%；
- 第一动作极差 P95：0.0161 A；
- 全域离线 NRMSE：0.331%–0.521%；
- 末端离线 NRMSE：0.329%–0.507%；
- 闭环电流 NRMSE：0.228%–0.340%；
- 最大充电时间偏差：0.869%；
- 目标到达率：100%；
- 电压违约：0 V；
- 最低加速：约 1035 倍。

唯一失败项：

$$
\max_k\left|I_k-I_{k-1}\right|
=
2.128929\ \mathrm{A}
$$

超过 2 A 硬门槛：

$$
2.128929-2
=
0.128929\ \mathrm{A}
$$

逐步数据进一步显示：

- DNN 闭环动作总数：13349；
- 风险动作：48 步；
- 风险比例：0.3596%；
- 五个种子全部出现少量斜率违约；
- 违约不只发生在初始步，也发生在后续控制步。

结论：

> Level 3 不是策略拟合失败，而是无约束 pure DNN 无法天然保证逐步硬约束。

主要入口：

- `outputs/phase7a_level3_slew/PHASE7A_LEVEL3_中文实验报告.md`
- `outputs/phase7a_level3_slew/metrics.json`
- `data/phase7a_level3_slew/closed_loop_trajectories.csv`
- `notebooks/phase7a_level3_slew_results.ipynb`
- `configs/phase7a_level3_slew.yaml`

## 4.2 Level 3P：最小解析投影严格通过

Level 3P 冻结了 Level 3 的 13 项工件：

- 模型与MPC实现；
- 配置；
- 教师数据；
- 两套冻结测试；
- 闭环初态与原始轨迹；
- 离线指标；
- 五个 ANN 模型。

13 项 SHA-256 哈希全部匹配；没有重新训练网络，也没有重新生成教师数据。

唯一变化是把 ANN 原始输出投影到当前可行电流区间。

定义：

$$
I_k^{-}
=
\max\left(0,I_{k-1}-2\right)
$$

$$
I_k^{+}
=
\min\left(10,I_{k-1}+2\right)
$$

$$
I_k^{\mathrm{safe}}
=
\operatorname{clip}
\left(
I_k^{\mathrm{raw}},
I_k^{-},
I_k^{+}
\right)
$$

这从结构上保证：

$$
0
\leq
I_k^{\mathrm{safe}}
\leq
10\ \mathrm{A}
$$

以及：

$$
\left|I_k^{\mathrm{safe}}-I_{k-1}\right|
\leq
2\ \mathrm{A}/5\mathrm{s}
$$

投影结果：

- Level 3 原始风险动作：48；
- Level 3P 实际介入：48；
- 精确位置重合：48/48；
- 风险动作 ±1 步邻域外新增介入：0；
- 介入比例：0.3596%；
- 介入时平均修正：0.0439 A；
- 最大修正：0.1327 A。

闭环结果：

- 最大斜率违约：$4.44\times10^{-16}$ A，属于浮点误差，视为 0；
- 最大单步电流变化：2.000000 A；
- 电流 NRMSE：0.228%–0.338%；
- 最大时间偏差：0.869%；
- 目标到达率：100%；
- 电压违约：0 V；
- 电流绝对边界违约：0 A；
- 最低加速：约 626 倍。

严格结论：

> Level 3P 最小投影严格通过。策略拟合原本已经足够准确；解析投影仅修复少量边界风险动作，即可恢复硬电流和硬斜率可行性。

注意：

- Level 3P 已经不是“无保护 pure DNN”，应称为“ANN策略＋解析输出投影”。
- 它在当前 2RC 同模型控制域内可以替代在线 MPC 求解。
- 投影只对电流边界和斜率约束提供数学保证；本轮电压无违约是实验结果，不是投影对所有状态的普遍保证。

主要入口：

- `outputs/phase7a_level3p_projection/PHASE7A_LEVEL3P_中文实验报告.md`
- `outputs/phase7a_level3p_projection/metrics.json`
- `data/phase7a_level3p_projection/projection_interventions.csv`
- `notebooks/phase7a_level3p_projection_results.ipynb`
- `configs/phase7a_level3p_projection.yaml`

---

## 5. 当前卡在哪里

当前没有算法运行或代码实现层面的硬阻塞。真正的待办是版本封存和下一阶段边界选择。

### 5.1 Level 3/3P 已形成两个独立提交

当前分支：

`codex/phase7b0-dfn-cross-model-audit`

当前证据提交：

- `933ab5e`：Level 3 硬斜率失败证据；
- `812441c`：Level 3P 最小输出投影修复；
- `ad30633`：Phase 7A 总结、HANDOFF 与核心结论图；
- `e50c04c`：Phase 7A PR #4 合并提交。

Level 3、Level 3P 已经合并到 `main`。Phase 7B-0 在独立分支运行，未重新训练 ANN，也未新增教师数据。

不要执行：

- `git reset --hard`
- `git checkout -- .`
- 未核对工作区就 `pull`
- 删除 Level 3/3P 数据或模型后重算

### 5.2 工作区包含额外未跟踪产物

除 Level 3/3P 外，还存在：

- 导师汇报 PPT 与渲染预览；
- PPT 填充素材；
- 项目总结 Markdown；
- 旧 Phase 5B/6B 运行日志；
- `outputs/phase7a_level1s_training_stability/PHASE7A_LEVEL1S_中文实验报告.md` 的行尾状态变化；
- `pyproject.toml` 新增 Level 3/3P CLI 入口。

提交前必须逐项确认范围，不要把所有未跟踪文件无差别加入一次实验提交。

### 5.3 研究边界

Phase 7A 已明确停止在 Level 3P，不进入 Level 4。

当前尚未证明：

- 25 ℃ DFN 跨模型闭环仍通过；
- 温度状态和温度约束下仍通过；
- 参数扰动或模型失配下仍通过；
- 跨温度、跨电池泛化；
- HIL 或真实 BMS 实时部署。

---

## 6. 下一步计划

## 6.1 第一优先级：提交并推送 Level 3/3P

建议保留“失败—修复”两段证据，分成两个提交：

### 提交一：Level 3

建议包含：

- `configs/phase7a_level3_slew.yaml`
- `src/battery_fast_charge/phase7a_level3_*.py`
- `scripts/build_phase7a_level3_notebook.py`
- `notebooks/phase7a_level3_slew_results.ipynb`
- `tests/test_phase7a_level3.py`
- `data/phase7a_level3_slew/`
- `outputs/phase7a_level3_slew/`
- `pyproject.toml` 中对应 CLI 入口

建议提交信息：

`Add Phase 7A Level 3 hard slew boundary`

### 提交二：Level 3P

建议包含：

- `configs/phase7a_level3p_projection.yaml`
- `src/battery_fast_charge/phase7a_level3p_*.py`
- `scripts/build_phase7a_level3p_notebook.py`
- `notebooks/phase7a_level3p_projection_results.ipynb`
- `tests/test_phase7a_level3p.py`
- `data/phase7a_level3p_projection/`
- `outputs/phase7a_level3p_projection/`
- `pyproject.toml` 中对应 CLI 入口
- 更新后的 `HANDOFF.md`

建议提交信息：

`Add Phase 7A Level 3P minimal output projection`

提交前运行完整测试并检查：

```powershell
git status --short --branch
git diff --check
$env:PYTHONPATH=(Resolve-Path 'src').Path
python -m pytest -q
```

完成后推送当前分支。

## 6.2 第二优先级：更新项目总结和导师汇报

需要把原先“Level 3计划”更新为“Level 3失败＋Level 3P修复”。

推荐核心表：

| 阶段 | 新增因素 | 策略精度 | 硬斜率 | 结论 |
|---|---|---:|---:|---|
| Level 2 | 第二极化状态 | 通过 | 无 | pure DNN通过 |
| Level 3 | 上一电流＋硬斜率 | 通过 | 失败 | pure DNN无硬保证 |
| Level 3P | 最小输出投影 | 通过 | 通过 | ANN＋投影严格通过 |

推荐核心表述：

> 模型复杂度和策略拟合不是当前失效原因；无约束 ANN 缺少硬约束保证，而解析投影可以仅介入 0.3596% 的风险动作完成修复。

现有汇报材料：

- `outputs/导师汇报_ANN_MPC充电控制研究思路_7页简版_2026-07-23.pptx`
- `outputs/PPT填充素材_电池模型_MPC_ANN_2026-07-23/`
- `docs/当前实验思路与已完成实验路线_2026-07-23.md`

这些文件当前未必都适合与实验代码放入同一提交，应单独决定是否纳入版本管理。

## 6.3 Phase 7B-0 跨模型审计结果

已按独立阶段执行，没有进入 Level 4。

建议名称：

**Phase 7B-0：冻结 Level 3P 控制器的 25 ℃ DFN 跨模型验证**

唯一变化：

$$
\text{ANN＋投影}\rightarrow\text{2RC}
$$

变为：

$$
\text{ANN＋投影}\rightarrow\text{25 ℃ DFN}
$$

必须冻结：

- Level 3 五个 ANN 模型；
- Level 3P 投影；
- MPC和2RC参数；
- 初始状态；
- 采样周期；
- SOC目标；
- 电流、电压和斜率门槛；
- 不重新训练；
- 不新增教师数据。

检查：

- 电流与斜率投影仍应严格满足；
- DFN端电压是否越界；
- 目标到达率；
- 充电时间偏差；
- ANN与MPC电流NRMSE；
- 是否出现闭环振荡；
- 在线加速。

若 DFN 电压失败，不要立刻重训更大网络。先增加：

$$
\text{ANN输出}
\rightarrow
\text{电流/斜率投影}
\rightarrow
\text{一步电压可行性检查}
$$

若一步检查仍不足，再转向：

$$
\text{ANN候选/参考}
\rightarrow
\text{有限迭代MPC安全修正}
$$

不要在 Phase 7B-0 同时加入温度、参数扰动和跨电池因素，否则无法归因。

实际完成 12 个冻结初态、1 个 MPC 基线和 5 个冻结 ANN 种子，共 72 条 25 ℃ Chen2020 DFN 闭环。结果为：

- 冻结哈希 10/10 匹配；
- 五种子电流 NRMSE：0.2282%–0.3382%；
- 最大平均离散充电时间偏差：0.8690%；
- 到达率：100%；
- 最大电流越界：0 A；
- 最大斜率越界：$4.44\times10^{-16}$ A，视为 0；
- 最大单步电流变化：2.000000 A；
- 最低控制器加速：545.3 倍；
- 电流方向反转：0；
- 投影介入率：0.2640%–0.4506%；
- MPC→DFN 最大电压：4.214054 V，越界 14.054 mV；
- ANN＋投影→DFN 最大电压：4.216401 V，越界 16.401 mV。

因此 Phase 7B-0 **仅电压安全门槛失败**。ANN 拟合、到达、输入约束、斜率约束、稳定性和速度均通过；MPC 基线自身也在 DFN 上越压，说明根因是 2RC 预测电压与 DFN 端电压之间的跨模型失配，而不是 ANN 模仿误差。

下一步停止多温度和扰动，优先建立不重新训练 ANN 的电压感知安全修正。

---

## 6.4 Phase 7B-1A：电压残差与制动可行性

Phase 7B-1A 只读取 Phase 7B-0 的 72 条冻结轨迹，不训练 ANN、不生成教师数据。

定义：

$$
e_{V,k+1}
=
V_{\mathrm{DFN},k+1}
-
\hat V_{\mathrm{2RC},k+1}(I_k)
$$

审计结果：

- 正向电压残差最大值：29.117 mV；
- 正向残差 P95 / P99：27.093 / 28.672 mV；
- 一步正向残差增长最大值：11.306 mV；
- 一步正向增长 P95 / P99：2.143 / 5.444 mV；
- 以最大一步增长作为保守裕量后，16,024 个控制步的电压—斜率可行区间空集次数为 0；
- 最小电压—斜率可行裕量为 0.840 A；
- 4.15、4.18、4.19 V 三个提前阈值均可在斜率约束内及时制动；
- 4.20 V 只能作为事后诊断阈值，不能作为提前触发阈值；
- 24 个独立 25 ℃ 初态已冻结，SHA-256 为 `738ae9eb52e2d7edbd598f9a2231e595743da920a9ae1b884ff8e5b5d5ecaab5`。

判定：**7B-1A 通过，可以进入 7B-1B 的“当前测量残差＋最大一步增长裕量＋一步电压电流上限”实现。**

## 7. 已踩过的坑，不要再踩

### 7.1 不要把低 NRMSE 当作硬约束保证

Level 3 已经证明：

- 闭环 NRMSE 低于 0.35%；
- 但仍有 48 个动作违反 2 A 硬斜率。

平均误差和最坏逐步约束是两类不同指标，必须分别检查。

### 7.2 不要混淆 pure DNN 与 projected DNN

- pure DNN：原始网络输出直接控制；
- projected DNN：输出经过可行域投影。

Level 3P 的成功不能写成“pure DNN严格通过”，必须写成“ANN＋解析投影严格通过”。

### 7.3 不要改动冻结工件后仍声称单因素验证

Level 3P 的因果证据来自 13 项哈希全部匹配。若修改模型、数据、初态、ANN或MPC，就不再是投影的单因素对照。

### 7.4 不要无故重新生成教师数据

教师数据生成慢，且重新运行会引入随机性和合同漂移。已有数据、模型和哈希应优先复用。

### 7.5 不要一次加入多个复杂因素

此前 Chen2020 复杂域同时包含：

- 热状态；
- 参数相关性；
- 控制记忆；
- 斜率约束；
- 模型失配；
- 求解器分支。

这导致失败难以归因。后续必须继续单因素推进。

### 7.6 不要忽略教师可行性与单值性

Phase 2R-D 和 Phase 5B 已证明：

- 教师可能存在近最优动作多值性；
- MPC在压力域可能本身不可行；
- 不可行教师域不能直接用于证明 ANN 模仿失败。

训练前先审计教师。

### 7.7 不要遗漏控制器记忆

当 MPC 目标或约束依赖上一电流、上一控制序列或 warm start 时，这些信息若不进入 ANN 输入，同一表观状态可能对应不同标签。

Level 3 已把上一电流明确加入状态，不能回退到不含 $I_{k-1}$ 的输入后仍比较硬斜率策略。

### 7.8 不要破坏轨迹级数据隔离

训练、验证和冻结测试必须按轨迹隔离，不能把同一轨迹的不同时间步分到不同集合，否则离线指标会虚高。

### 7.9 不要忽略末端覆盖

Level 1 的失败来自没有覆盖 0.80 SOC 附近的末端降流区。离线全局测试通过不代表闭环末端行为正确。

### 7.10 不要让回放合同不一致

Phase 5B-0.5 的错误结论来自：

- 随机种子不一致；
- 场景索引不一致；
- 噪声序列不一致；
- 截止时间不一致；
- 目标电流 cap 逻辑不一致。

配对比较必须共享完整回放合同。

### 7.11 不要混淆电池模型角色

- DFN更真实，但计算量大，主要作为虚拟电池；
- 1RC/2RC适合控制预测，但不是比DFN更真实；
- Level 1–3P主要是同模型电控制验证，不等于电热和DFN已经验证。

### 7.12 不要把仿真结论直接外推到真实BMS

真实部署还需要：

- 在线状态估计；
- 参数辨识与老化；
- 传感器噪声；
- 故障与降级；
- 实时硬件测试；
- HIL；
- 真实电芯验证。

### 7.13 不要覆盖未提交工作区

当前工作区包含完整 Level 3/3P 证据链。任何清理、重置、切换或拉取前必须先核对并保存。

### 7.14 不要把行尾提示当成内容修改

Windows 下可能出现 LF/CRLF 提示。当前 Level 1S 中文报告在 `git status` 中显示修改，但文本 diff 为空，可能只是行尾状态。不要未经核对将其混入 Level 3/3P 提交。

### 7.15 不要自动提交所有日志和PPT预览

工作区有旧运行日志、PPT渲染目录和 `.inspect.ndjson`。提交前决定哪些属于正式证据，避免把临时预览和大批无关文件混入实验提交。

---

## 8. 新对话启动检查

进入项目后先执行：

```powershell
git status --short --branch
git log --oneline --decorate -5
git diff --check
```

优先阅读：

1. `HANDOFF.md`
2. `outputs/phase7a_level3_slew/PHASE7A_LEVEL3_中文实验报告.md`
3. `outputs/phase7a_level3_slew/metrics.json`
4. `outputs/phase7a_level3p_projection/PHASE7A_LEVEL3P_中文实验报告.md`
5. `outputs/phase7a_level3p_projection/metrics.json`
6. `data/phase7a_level3p_projection/projection_interventions.csv`

检查测试环境：

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
python -m pytest -q
```

若系统 `python` 不是项目环境，应先查找项目已有虚拟环境或使用 Codex 工作区提供的 Python，不要随意安装或升级依赖。

---

## 9. 给新对话的直接任务

建议新对话按以下顺序工作：

1. 先提交 Phase 7B-1A 诊断与独立确认集冻结证据；
2. 保持五个 ANN、MPC、2RC、Level 3P 投影和 11.306 mV 残差增长裕量冻结；
3. 实现 7B-1B 一步电压感知安全层，不加入紧急硬裁剪；
4. 先回归现有 12 个初态，再运行冻结的 24 初态确认集；
5. 若任何时刻出现电压—斜率空区间，停止一步方案并转 2–5 步短时域修正；
6. 只有回归集和确认集同时严格通过，才允许规划多温度验证。

当前最需要保护的研究结论是：

> Level 3/3P 证明解析投影可以补上输入硬约束保证；Phase 7B-0 证明它不能自动保证跨模型状态约束；Phase 7B-1A 又证明基于在线电压残差的一步限制在现有回归域内与 2 A/步斜率约束相容。下一步是验证该安全层能否在独立初态上把 DFN 越压严格降为 0。
