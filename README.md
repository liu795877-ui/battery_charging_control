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
