# 电池约束快充控制仿真

本项目使用 PyBaMM `Chen2020` 参数集建立高保真虚拟电芯。第一阶段完成 25 ℃下 1C、1.5C 和 2C CC–CV 基线；第二阶段使用 OCV、充电脉冲和热响应虚拟试验辨识面向 MPC 的二阶 RC＋双节点热模型。

## 已冻结的研究配置

- 高保真对象：PyBaMM DFN＋集总热模型；
- 参数集：Chen2020，标称容量 5 Ah；
- SOC：10%→80%；
- 环境温度与初始温度：25 ℃；
- 约束：4.2 V、10 A、35 ℃、每 5 s 电流变化不超过 2 A；
- 控制周期：5 s；
- 后续 MPC 预测时域：60 步，即 300 s；
- 研究目标：求解约束下的最短可行充电轨迹，不预设 MPC 必须优于公平设置的 CC–CV。

项目统一采用“充电电流为正”的符号约定；PyBaMM 原始结果的电流符号会在导出时转换。

## 建议的代码阅读顺序

如果你暂时看不懂全部代码，不需要从头逐行硬读。建议按下面顺序：

1. `configs/phase1.yaml`：先看实验参数、单位和约束；
2. `notebooks/01_chen2020_baseline.ipynb`：按实验过程运行并观察结果；
3. `src/battery_fast_charge/runner.py`：看一次完整实验如何组织；
4. `src/battery_fast_charge/high_fidelity.py`：看 DFN 模型、CC–CV 和结果导出；
5. `src/battery_fast_charge/checks.py` 与 `plotting.py`：看数据检查和绘图；
6. `src/battery_fast_charge/config.py`：最后再看配置如何转换成 Python 对象。

第二阶段建议继续按下面顺序：

1. `docs/phase2_model.md`：先理解2RC和热模型方程、符号与限制；
2. `configs/phase2.yaml`：看OCV、脉冲、热训练和独立验证协议；
3. `notebooks/02_reduced_model_identification.ipynb`：先从图表理解实验结果；
4. `src/battery_fast_charge/reduced_model.py`：看两个降阶模型如何计算；
5. `identification.py` 与 `phase2_runner.py`：最后看参数优化和全流程组织。

代码注释重点解释物理意义、单位、正负号和设计原因。像“给变量赋值”这类可
直接从代码读出的动作不逐行重复注释，以免真正重要的信息被淹没。

## 运行

克隆仓库后，在项目根目录创建独立虚拟环境并安装：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m battery_fast_charge --config configs\phase1.yaml --project-root .
```

运行 Notebook：

```powershell
jupyter-lab notebooks\01_chen2020_baseline.ipynb
```

运行第二阶段：

```powershell
python -m battery_fast_charge.phase2_cli --config configs\phase2.yaml --project-root .
jupyter-lab notebooks\02_reduced_model_identification.ipynb
```

## 输出

- `data/processed/`：各 C-rate 的轨迹 CSV；
- `outputs/metrics/`：逐工况 JSON 和汇总 CSV；
- `outputs/figures/`：SOC、电压、电流、温度对比图；
- `notebooks/01_chen2020_baseline.ipynb`：可重复运行的阶段演示；
- `data/phase2/`：OCV、脉冲、热训练和独立验证数据；
- `outputs/metrics/phase2_*.json`：辨识参数和验证误差；
- `notebooks/02_reduced_model_identification.ipynb`：第二阶段实验记录；
- `docs/phase2_model.md`：完整状态方程、符号、单位和适用边界。

## 当前边界

第二阶段双节点热模型只用 Chen2020 集总平均温度进行约束辨识；核心和表面温度是潜在状态，尚未得到独立空间温度数据验证。MPC 和 ANN 将在后续阶段逐步加入。当前的 35 ℃是研究性控制上限，不代表制造商绝对安全极限。

# 第三阶段 A

第三阶段使用第二阶段已经验证的2RC＋平均热模型建立约束MPC教师，并把每次控制动作逐步施加到Chen2020 DFN虚拟电池。MPC内部以4.14 V和33.5 ℃收紧约束，为DFN的4.20 V和35 ℃边界预留模型误差余量。

建议按以下顺序阅读：

1. `docs/phase3_mpc.md`：控制状态、目标函数、约束收紧和验证边界；
2. `configs/phase3.yaml`：第一版数值设置；
3. `notebooks/03_mpc_teacher_validation.ipynb`：结果图、基线对比和验收判断；
4. `src/battery_fast_charge/mpc.py`：降阶预测和在线优化；
5. `closed_loop.py` 与 `phase3_runner.py`：DFN闭环和结果落盘。

运行第三阶段：

```powershell
python -m battery_fast_charge.phase3_cli --config configs\phase3.yaml --project-root .
jupyter-lab notebooks\03_mpc_teacher_validation.ipynb
```

主要输出为 `data/phase3/`、`outputs/metrics/phase3_*`、`outputs/figures/phase3_mpc_closed_loop.png` 和 `outputs/phase3_report.md`。

# 第三阶段 B

第三阶段 B 不训练 DNN，而是先建立可审计的 MPC 教师数据协议。状态只从 12 条受约束、动力学可达的探索轨迹中抽取；样本按 SOC 分层，并按整条轨迹划分训练、验证和测试集，防止相邻状态泄漏。全部 MPC 求解尝试保存在审计表中，只有优化成功、预测可行且未使用回退动作的标签进入教师数据集。

建议按以下顺序阅读：

1. `docs/phase3b_teacher_data.md`：状态来源、标签规则、数据划分和验收闸门；
2. `configs/phase3b.yaml`：探索策略、采样数量和质量阈值；
3. `notebooks/04_mpc_teacher_dataset.ipynb`：数据覆盖、公平基线和进入 DNN 训练的判断；
4. `src/battery_fast_charge/teacher_data.py`：可达轨迹、分层采样和 MPC 批量标注；
5. `filtered_baseline.py` 与 `phase3b_runner.py`：同约束基线和全流程组织。

运行第三阶段 B：

```powershell
python -m battery_fast_charge.phase3b_cli --config configs\phase3b.yaml --project-root .
jupyter-lab notebooks\04_mpc_teacher_dataset.ipynb
```

正式结果包含 168 个已接受教师标签，按 112/28/28 个样本和 8/2/2 条整轨迹划分为训练/验证/测试集。数据质量闸门通过，可以开始第一版 DNN 实验；但当前约束 MPC 的 DFN 充电时间为 53.58 min，同约束过滤 1C 基线为 53.33 min，因此尚不能声称 MPC 缩短了充电时间。

主要输出为 `data/phase3b/`、`outputs/metrics/phase3b_*`、`outputs/figures/phase3b_*` 和 `outputs/phase3b_report.md`。

# 第四阶段 A

第四阶段A使用阶段3B的整轨迹数据训练一个 `5-8-8-1` 小型ANN，逼近MPC第一步电流。标准化只拟合训练集，验证集用于选择L2正则化和初始化，测试集只做最终评价。模型导出为只依赖NumPy的非可执行NPZ权重。

运行训练和闭环验证：

```powershell
python -m pip install -e ".[ann]"
python -m battery_fast_charge.phase4_cli --config configs\phase4a.yaml --project-root .
jupyter-lab notebooks\05_tiny_ann_imitation.ipynb
```

第一版测试MAE为0.0195 A、RMSE为0.0691 A。ANN加安全过滤器在Chen2020 DFN上用55.67 min完成10%到80% SOC，满足4.20 V、35 ℃、10 A和每5 s最大2 A变化限制；平均推理约0.10 ms。但安全过滤器介入约50.3%的控制周期，且ANN比MPC和过滤1C更慢，因此当前结果只证明低计算量模仿流程可行，不能称为裸ANN安全或更快的控制器。

主要输出为 `data/phase4a/`、`outputs/models/phase4a_tiny_ann.npz`、`outputs/metrics/phase4a_*`、`outputs/figures/phase4a_*`、`notebooks/05_tiny_ann_imitation.ipynb` 和 `outputs/phase4a_report.md`。

# 第四阶段 B-1

第四阶段B-1先改进教师，再继续扩充ANN数据。新教师由启动参考调节器、低SOC热预算MPC和终端一步可行参考调节器组成；预测到达80% SOC后停止施加有效电流，以表达“到达目标即结束”。

运行正式验证：

```powershell
python -m battery_fast_charge.phase4b_cli --config configs\phase4b.yaml --project-root .
jupyter-lab notebooks\06_thermal_budget_mpc.ipynb
```

在Chen2020、25 ℃、10%→80% SOC和同一组物理约束下，混合教师用52.67 min完成充电，比过滤1C的53.33 min缩短1.25%；最高电压4.1425 V、最高平均温度33.5024 ℃、MPC回退0次。阶段门槛通过，可以进入主动数据聚合，但这不是全局最优性或真实BMS安全性的证明。

建议先读 `docs/phase4b_teacher.md`，再看 `notebooks/06_thermal_budget_mpc.ipynb`。主要输出位于 `data/phase4b/`、`outputs/metrics/phase4b_*`、`outputs/figures/phase4b_*` 和 `outputs/phase4b_report.md`。
