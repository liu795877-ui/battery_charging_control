# 项目交接说明

更新时间：2026-07-21
项目目录：`C:\Users\LENOVO\Documents\动力电池AI`
当前分支：`agent/phase2-reduced-model-identification`
当前提交：`592a7e5 Add Phase 6 paper-style DL-E-MPC validation`

## 1. 我们在做什么

本项目研究锂离子电池的仿真快速充电控制。当前总体路线是：

1. 使用公开、参数较完整的 Chen2020 电化学参数集，以单体电池为对象；
2. 充电范围固定为 10%–80% SOC，第一目标是缩短充电时间；
3. 用带电流、电压、温度和电流变化率约束的 MPC 生成可行的最优/近最优充电轨迹；
4. 用 ANN/DNN 学习 MPC 的状态到控制量映射，以较低在线计算量逼近 MPC；
5. 在 DFN 高保真模型闭环中比较 DNN、MPC 和 CC–CV，不预设 DNN 一定更快；
6. 目前正在迁移验证 2025 年论文 *Health-aware optimal charging of lithium-ion batteries using deep-neural networks-based explicit constrained model predictive control* 的“反复求解约束 MPC 生成数据，再训练显式 DNN 控制器”路线（DOI：10.1016/j.compchemeng.2025.109096）。

研究边界：现阶段只做纯仿真验证，后期才考虑接入 BMS。当前不是在训练真实电池硬件上的最终控制器，也不能把仿真结果直接当作可部署的安全结论。

## 2. 已完成的阶段

### Phase 1：基础充电仿真

- 建立 Chen2020 电池模型下的 CC–CV 基线和 10%–80% SOC 终止逻辑。
- 修正过 2C 工况不能正确进入/保持恒压、目标 SOC 终止不正确的问题。
- Git 标签：`v0.1.0`、`v0.1.1`。

### Phase 2：降阶模型辨识

- 为 MPC 建立计算量较低的预测模型，并用 Chen2020/DFN 结果校核。
- 相关说明：`docs/phase2_model.md`。
- 提交：`8ce8d4a`。

### Phase 3 / 3B：约束 MPC 与教师数据

- 实现电压、电流、温度和电流斜率约束下的 MPC。
- 在 Chen2020 DFN 闭环中验证 MPC，并生成可达的 MPC 教师数据集。
- 相关说明：`docs/phase3_mpc.md`、`docs/phase3b_teacher_data.md`。
- 提交：`0d03ceb`、`029e761`。

### Phase 4 / 4B / 4B-2：小型 ANN 模仿与主动增广

- 训练过小型 ANN 模仿 MPC。
- 做过热预算教师验证和针对困难区域的主动数据增广。
- 相关说明：`docs/phase4_tiny_ann.md`、`docs/phase4b_teacher.md`、`docs/phase4b2_active_learning.md`。
- 提交：`3756938`、`8001fd1`、`b4d4097`。

### Phase 5A：鲁棒性压力测试

- 对 ANN 控制器做过温度变化等压力测试，用于暴露分布外和约束风险。
- 相关说明：`docs/phase5a_robustness.md`。
- 提交：`fb4a1fd`。

### Phase 6A：论文式纯 DNN 方法验证

- 独立建立 `phase6_paper_method_validation`，没有覆盖 Phase 1–5。
- 按论文思路完成状态采样、MPC 标签生成、纯 DNN 训练、25 ℃ 名义闭环比较。
- 预设通过门槛包括：闭环电流 NRMSE < 1%、无严重物理约束违约、充电时间相对 MPC 偏差 < 2%、推理显著快于 MPC，并最终检查 15/25/30 ℃。
- Phase 6A 没有通过纯 DNN 方法验证，因此继续做 Phase 6B 诊断。
- 相关说明：`docs/phase6_paper_method_validation.md`。
- 已推送提交：`592a7e5`。

### Phase 6B：解释纯 DNN 为什么没有学好 MPC

Phase 6B 已实现、已运行、已出报告并通过测试，但**尚未提交 Git**。它包含三项实验：

1. 按 SOC、温度、上一时刻电流、约束激活/临界状态分区诊断误差，重点检查电流变化率 `ΔI` 约束附近；
2. 将数据扩大到 1000 个初始状态，并比较 `5-32-32-16-1` 和 `5-64-64-32-1` 两个更大网络；
3. 将 pure DNN 和 projected DNN 分开比较。pure DNN 完全不裁剪；projected DNN 只把输出投影到 `I_max` 和 `ΔI_max` 可行区间，作为独立对照组。

最终生成 878 条被接受的教师轨迹、7024 个展开样本。选择的网络是 `[5, 32, 32, 16, 1]`，但训练达到 2500 次迭代上限，尚未收敛。

关键结果：

| 指标 | pure DNN | projected DNN |
|---|---:|---:|
| 25 ℃ DFN 闭环电流 NRMSE | 5.228% | 7.023% |
| 10%–80% 充电时间 | 52.17 min | 52.42 min |
| 相对 MPC 充电时间偏差 | 2.644% | 2.177% |
| 最大 `ΔI` 违约 | 2.581 A | 0 A |
| 严重物理违约判定 | 是 | 否 |

离线测试集 NRMSE 为 11.494%，R² 为 0.659。最差误差集中在：

- 上一时刻电流 2–5 A：NRMSE 15.138%；
- 上一时刻电流 0–2 A：NRMSE 14.953%；
- `ΔI` 约束临界附近：NRMSE 14.762%；
- `ΔI` 约束激活样本：NRMSE 14.751%。

当前最可靠的结论是：输出投影能够消除电流斜率违约，但没有改善控制轨迹拟合，反而使闭环 NRMSE 从 5.228% 上升到 7.023%。所以主要问题不是“网络输出偶尔越界”这么简单，而是 DNN 本体没有准确学到 MPC 在约束切换附近的映射。更大的数据集和更大的网络在这次实验中也没有自动解决问题。

Phase 6B 入口与结果：

- 配置：`configs/phase6b_dnn_failure_diagnosis.yaml`
- 运行入口：`src/battery_fast_charge/phase6b_cli.py`
- 核心流程：`src/battery_fast_charge/phase6b_runner.py`
- 总报告：`outputs/phase6b_report.md`
- 完整指标：`outputs/metrics/phase6b_metrics.json`
- 分区诊断表：`data/phase6b_dnn_failure_diagnosis/error_partition_diagnostics.csv`
- 方法说明：`docs/phase6b_dnn_failure_diagnosis.md`
- 图：`outputs/figures/phase6b_*.png`
- 模型：`outputs/models/phase6b_paper_dnn.npz`

## 3. 当前状态与“卡点”

当前没有代码执行层面的硬阻塞；Phase 6B 已完整跑通。真正的研究卡点是：纯前馈 DNN 对 MPC 的分段、约束切换映射拟合不足，尤其是 `I_previous` 较低以及 `ΔI` 约束激活/临界区域。

同时存在一个版本管理待办：工作区当前有 Phase 6B 的全部未提交改动。新对话开始后应先运行 `git status --short --branch` 核对，不要清理、重置或重新拉取覆盖这些文件。当前已知未提交内容包括：

- 修改：`README.md`、`pyproject.toml`、`src/battery_fast_charge/phase6_plotting.py`；
- 新增：Phase 6B 配置、源代码、测试、数据、模型、图、指标、报告和本交接文档；
- `outputs/phase6b_run_stdout.txt` 和 `outputs/phase6b_run_stderr.txt` 是运行日志。stderr 主要是 scikit-learn 达到迭代上限的收敛警告；提交前可决定是否保留日志，其他 Phase 6B 产物应保留。

最近一次完整测试结果：`37 passed`。`git diff --check` 通过，仅出现 Windows 下 LF/CRLF 转换提示。

## 4. 建议的下一步

### 第一步：先保存 Phase 6B 版本

1. 查看 `git diff` 和 `git status`，确认只有上述 Phase 6B 工作；
2. 保留源代码、配置、测试、报告、指标、诊断 CSV、模型和图片；
3. 运行完整测试；
4. 建议提交信息：`Add Phase 6B DNN failure diagnosis`；
5. 推送当前分支。不要在没有核对工作区的情况下执行 pull、reset 或 checkout。

### 第二步：开展 Phase 6C，但不要只盲目增大网络

建议新建独立 `phase6c_constraint_regime_learning`，保留 Phase 6A/6B 原始结果。优先按以下顺序做：

1. **确认误差机制**：画出 MPC 标签相对 `SOC、T、I_previous` 的局部切片，明确 `ΔI` 激活边界处是否存在折点、不连续或多解/求解器抖动；同时检查同类状态是否因为未纳入 DNN 输入的隐状态而对应不同标签。
2. **定向数据增广/加权**：增加 `I_previous=0–5 A`、`ΔI` 激活和临界区域样本，在损失中提高这些样本权重；保留统一的独立测试集，不能把增广样本泄漏到测试集。
3. **把预测目标改为 `ΔI` 做严格对照**：预测下一步电流增量而不是绝对电流，并保持 pure 模型和 projected 模型分组；这利用了问题的自然约束结构，但不能把投影模型冒充纯论文式 DNN。
4. **约束工况分区/混合专家对照**：可按“斜率约束激活、其他约束激活、内部自由区”训练分类器加回归器，检验单一连续网络是否难以覆盖所有控制区域。
5. **再处理优化器收敛**：当前两个候选网络都撞到 2500 次迭代上限。先标准化和检查标签几何，再尝试更高迭代数、不同初始化或 Adam/早停；不能把未收敛结果解释成网络容量的最终上限。
6. **沿用同一评价门槛**：先过 25 ℃ 名义闭环，再做 15/30 ℃ 与 Phase 5A 压力测试。不要因为误差下降就放宽原有 1% NRMSE、2% 时间偏差和物理约束门槛。

建议 Phase 6C 的最小实验矩阵：

| 组别 | 输入/目标 | 数据处理 | 输出处理 | 用途 |
|---|---|---|---|---|
| A | 原 5 输入 → 绝对电流 | 原始分布 | 无 | Phase 6B 基线 |
| B | 原 5 输入 → 绝对电流 | 边界增广/加权 | 无 | 判断数据覆盖是否主因 |
| C | 原 5 输入 → `ΔI` | 边界增广/加权 | 无 | 判断目标参数化是否更合适 |
| D | 与最佳纯模型相同 | 相同 | `I_max + ΔI_max` 投影 | 只评估可行性后处理贡献 |

只有纯模型先在离线独立测试和 25 ℃ 闭环达到可接受水平，才值得继续扩大到多温度。若定向增广后约束切换区仍明显失败，应转向分区模型、可微约束结构或保留在线优化修正，而不是继续宣称单一 DNN 已学到“最优控制器”。

## 5. 已踩过的坑，勿重复

1. **不要把 ANN 说成自动得到“最优控制器”**。ANN 只是模仿有限状态域内的 MPC 教师；最优性来自模型、目标、约束和求解精度，而且只能在验证范围内讨论。
2. **不要预设 ANN 一定快充得比 CC–CV 更快**。本项目只预设在线推理比反复解 MPC 快，充电时间优劣必须由仿真结果决定。
3. **不要把 projected DNN 混入 pure DNN 结论**。投影是安全/可行性后处理，必须是独立对照组，否则无法判断网络本体是否学好。
4. **不要只看总体 RMSE**。总体均值会掩盖 `ΔI` 激活边界和低 `I_previous` 区域的严重错误，必须保留分区诊断。
5. **不要仅靠扩大网络和数据量得出结论**。Phase 6B 已表明 1000 初态、两个更大网络仍失败，而且均未在 2500 次内收敛；需先处理数据覆盖、标签结构和优化收敛。
6. **不要把达到迭代上限的模型叫作已充分训练**。当前选择网络 `selected_optimizer_reached_iteration_limit=True`，这是明确限制，报告中必须保留。
7. **教师数据生成很慢**。1000 个初态的平均 MPC 求解约 0.90 s，最大约 5.29 s。Phase 6B 已实现 CSV 缓存复用；不要无故删除 `data/phase6b_dnn_failure_diagnosis/initial_state_audit.csv` 和 `paper_teacher_dataset.csv` 后重算。
8. **此前尝试过过大的训练网格**：2 个结构 × 3 个正则化 × 3 个种子、8000 次迭代，实际过慢且网络撞迭代上限，后来收缩为每结构一个正则化和一个种子、2500 次。若恢复完整网格，应当有明确算力/时间预算，并保存中间结果。
9. **不要覆盖旧阶段**。Phase 6A、6B 以及未来 6C 应保持独立目录、配置、报告和产物，便于论文式方法的可追溯对照。
10. **不要强行补齐 Chen2020 没有的空间热参数**。该参数集缺少极耳尺寸和多个表面对流参数；人为补齐会把假设误当事实。它主要威胁空间温度分布、热点和局部约束结论。当前单温度/集中热模型阶段可继续，但不能据此声称已验证电芯内部热点安全或真实 BMS 可部署性。
11. **不要将仿真模型直接部署至 BMS**。后续至少需要参数辨识、模型失配分析、传感器噪声/状态估计、实时性、故障降级、HIL 和真实电芯验证。
12. **Windows 工具注意事项**：本环境中 `rg` 曾被系统拒绝，可用 PowerShell 的 `Get-ChildItem`/`Select-String`；运行测试和脚本时需设置 `PYTHONPATH` 指向 `src`，并使用项目已配置的虚拟环境。换行符出现 LF/CRLF 提示并不等同于内容错误。
13. **不要覆盖未提交工作区**。用户之前曾因新任务覆盖项目后从 GitHub 恢复；当前 Phase 6B 尚未提交，若此时直接重新拉取或重置，可能再次丢失工作。

## 6. 新对话的建议启动检查

进入项目后依次确认：

```powershell
git status --short --branch
git log --oneline --decorate -5
$env:PYTHONPATH=(Resolve-Path 'src').Path
& 'D:\CodexData\.codex\.chatgpt-projects\g-p-6a5a0bf7034481918b38c28982065456\battery_charging_control\.venv\Scripts\python.exe' -m pytest -q
```

Phase 6B 可通过项目脚本入口重新运行，但默认应优先复用现有教师数据缓存。运行前先阅读：

1. `outputs/phase6b_report.md`
2. `outputs/metrics/phase6b_metrics.json`
3. `docs/phase6b_dnn_failure_diagnosis.md`
4. `data/phase6b_dnn_failure_diagnosis/error_partition_diagnostics.csv`

最后，研究表述应保持克制：目前已经证明的是“纯 DNN 在本配置下没有充分逼近 MPC，输出投影能恢复斜率可行性但不能修复拟合误差”；尚未证明的是“论文方法普遍无效”或“已获得可部署的最优充电控制器”。
