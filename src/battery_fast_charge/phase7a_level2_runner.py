"""执行 Phase 7A Level 2 的 2RC 教师、审计、DNN 和同模型闭环。"""

from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ann_model import TinyANN
from .phase7a_level1_config import load_phase7a_level1_config
from .phase7a_level1_runner import _regression_metrics, _van_der_corput, _warm_starts
from .phase7a_level1s_runner import continuous_crossing_time_s
from .phase7a_level2_config import Level2DomainConfig, Phase7ALevel2Config
from .phase7a_level2_model import Level2MPC, Level2Model, Level2State

FEATURES = ("state_soc", "state_polarization_1_v", "state_polarization_2_v")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _validate_parameters(config: Phase7ALevel2Config, root: Path) -> None:
    identified = json.loads((root / config.source_identified_parameters).read_text(encoding="utf-8"))["electrical_2rc"]
    for key, value in (("r0_ohm", config.model.r0_ohm), ("r1_ohm", config.model.r1_ohm),
                       ("tau1_s", config.model.tau1_s), ("r2_ohm", config.model.r2_ohm), ("tau2_s", config.model.tau2_s)):
        if not np.isclose(value, identified[key], rtol=0.0, atol=1e-15):
            raise RuntimeError(f"Level 2 参数 {key} 未保持项目辨识值。")


def design_initial_states(config: Phase7ALevel2Config, model: Level2Model, domain: Level2DomainConfig,
                          prefix: str, test_split: str, seed_offset: int) -> pd.DataFrame:
    records = []; candidate = 1
    voltage_limit = model.inherited.mpc.terminal_voltage_max_v - config.data.initial_voltage_margin_v
    while len(records) < domain.trajectory_count:
        soc_unit = _van_der_corput(candidate, 2) ** domain.soc_sampling_power
        soc = domain.soc_bounds[0] + np.ptp(domain.soc_bounds) * soc_unit
        v1 = domain.v1_bounds_v[0] + np.ptp(domain.v1_bounds_v) * _van_der_corput(candidate, 3)
        v2 = domain.v2_bounds_v[0] + np.ptp(domain.v2_bounds_v) * _van_der_corput(candidate, 5)
        candidate += 1
        state = Level2State(soc, v1, v2)
        if model.terminal_voltage(state, 0.0) > voltage_limit: continue
        records.append({"trajectory_id": f"{prefix}_{len(records):03d}", "initial_soc": soc,
                        "initial_polarization_1_v": v1, "initial_polarization_2_v": v2})
    frame = pd.DataFrame(records)
    ids = frame.trajectory_id.to_numpy(object).copy()
    np.random.default_rng(config.data.random_seed + seed_offset).shuffle(ids)
    train_end = domain.train_trajectory_count; validation_end = train_end + domain.validation_trajectory_count
    split = {str(v): "train" for v in ids[:train_end]}
    split.update({str(v): "validation" for v in ids[train_end:validation_end]})
    split.update({str(v): test_split for v in ids[validation_end:]})
    frame["split"] = frame.trajectory_id.map(split)
    return frame


def _teacher_trajectory(row: pd.Series, steps: int, model: Level2Model) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    controller = Level2MPC(model)
    state = Level2State(float(row.initial_soc), float(row.initial_polarization_1_v), float(row.initial_polarization_2_v))
    records = []; rejection = ""
    for step in range(steps):
        result = controller.solve(state)
        if not result.optimizer_success or not result.prediction_feasible or result.used_fallback:
            rejection = result.status; break
        next_state = model.step(state, result.current_a)
        records.append({"trajectory_id": row.trajectory_id, "split": row.split, "step_index": step,
                        "state_soc": state.soc, "state_polarization_1_v": state.polarization_1_v,
                        "state_polarization_2_v": state.polarization_2_v, "teacher_current_a": result.current_a,
                        "terminal_voltage_v": model.terminal_voltage(state, result.current_a), "next_soc": next_state.soc,
                        "next_polarization_1_v": next_state.polarization_1_v, "next_polarization_2_v": next_state.polarization_2_v,
                        "teacher_objective": result.objective_value, "teacher_solve_time_s": result.solve_time_s,
                        "teacher_optimizer_success": result.optimizer_success, "teacher_prediction_feasible": result.prediction_feasible,
                        "teacher_used_fallback": result.used_fallback, "minimum_prediction_margin": result.minimum_constraint_margin,
                        **{f"plan_block_{i:02d}_a": float(v) for i, v in enumerate(result.plan_a)}})
        state = next_state
    accepted = len(records) == steps
    return (records if accepted else []), {"trajectory_id": row.trajectory_id, "split": row.split,
                                           "teacher_accepted": accepted, "completed_step_count": len(records),
                                           "rejection_reason": rejection}


def _generate_domain(config: Phase7ALevel2Config, model: Level2Model, domain: Level2DomainConfig,
                     prefix: str, test_split: str, seed_offset: int, data_dir: Path, resume: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_path = data_dir / f"{prefix}_teacher_dataset.csv"; audit_path = data_dir / f"{prefix}_trajectory_audit.csv"
    design = design_initial_states(config, model, domain, prefix, test_split, seed_offset)
    design.to_csv(data_dir / f"{prefix}_initial_state_design.csv", index=False)
    if resume and dataset_path.exists() and audit_path.exists():
        dataset, audit = pd.read_csv(dataset_path), pd.read_csv(audit_path)
        if len(audit) == domain.trajectory_count: return dataset, audit
    rows, audits = [], []
    for index, (_, initial) in enumerate(design.iterrows(), start=1):
        trajectory, audit = _teacher_trajectory(initial, domain.trajectory_steps, model)
        rows.extend(trajectory); audits.append(audit)
        if index % 10 == 0:
            pd.DataFrame(rows).to_csv(dataset_path, index=False); pd.DataFrame(audits).to_csv(audit_path, index=False)
            print(f"Level 2 {prefix} teacher {index}/{len(design)}", flush=True)
    dataset, audit = pd.DataFrame(rows), pd.DataFrame(audits)
    dataset.to_csv(dataset_path, index=False); audit.to_csv(audit_path, index=False)
    return dataset, audit


def _representative_states(dataset: pd.DataFrame, count: int) -> pd.DataFrame:
    ordered = dataset.sort_values(list(FEATURES)).reset_index(drop=True)
    indices = np.linspace(0, len(ordered)-1, count).round().astype(int)
    selected = ordered.iloc[indices][["trajectory_id", "step_index", *FEATURES]].copy().reset_index(drop=True)
    selected.insert(0, "state_id", [f"level2_audit_{i:03d}" for i in range(count)])
    return selected


def _teacher_audit(config: Phase7ALevel2Config, model: Level2Model, dataset: pd.DataFrame,
                   data_dir: Path, resume: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    states_path = data_dir / "representative_states_100.csv"; solutions_path = data_dir / "multistart_solutions.csv"
    selected = pd.read_csv(states_path) if resume and states_path.exists() else _representative_states(dataset, config.data.audit_state_count)
    selected.to_csv(states_path, index=False)
    solutions = pd.DataFrame()
    if resume and solutions_path.exists():
        candidate = pd.read_csv(solutions_path); counts = candidate.groupby("state_id").size()
        if len(counts) == config.data.audit_state_count and (counts == config.data.warm_starts_per_state).all(): solutions = candidate
    if solutions.empty:
        records = []
        for state_index, row in selected.iterrows():
            state = Level2State(float(row.state_soc), float(row.state_polarization_1_v), float(row.state_polarization_2_v))
            controller = Level2MPC(model)
            for warm_index, (kind, warm) in enumerate(_warm_starts(controller, state_index, config.data.warm_starts_per_state, config.data.random_seed)):
                controller.set_warm_start(warm); result = controller.solve(state)
                records.append({"state_id": row.state_id, "warm_start_index": warm_index, "warm_start_kind": kind,
                                "first_action_a": result.current_a, "objective_value": result.objective_value,
                                "optimizer_success": result.optimizer_success, "prediction_feasible": result.prediction_feasible,
                                "used_fallback": result.used_fallback, "status": result.status,
                                "minimum_constraint_margin": result.minimum_constraint_margin})
            if (state_index + 1) % 10 == 0:
                pd.DataFrame(records).to_csv(solutions_path, index=False)
                print(f"Level 2 audit {state_index+1}/{len(selected)}", flush=True)
        solutions = pd.DataFrame(records); solutions.to_csv(solutions_path, index=False)
    rows = []
    gates = model.inherited.gates
    for state_id, group in solutions.groupby("state_id", sort=True):
        valid = group[group.optimizer_success.astype(bool) & group.prediction_feasible.astype(bool) & ~group.used_fallback.astype(bool)]
        if valid.empty: near_range, near_count = np.nan, 0
        else:
            best = float(valid.objective_value.min()); tolerance = max(gates.absolute_objective_tolerance, abs(best)*gates.relative_objective_tolerance)
            near = valid[valid.objective_value <= best + tolerance]
            near_range, near_count = float(near.first_action_a.max()-near.first_action_a.min()), len(near)
        rows.append({"state_id": state_id, "successful_feasible_count": len(valid),
                     "optimizer_success_count": int(group.optimizer_success.astype(bool).sum()),
                     "prediction_feasible_count": int(group.prediction_feasible.astype(bool).sum()),
                     "fallback_count": int(group.used_fallback.astype(bool).sum()), "near_optimal_solution_count": near_count,
                     "near_optimal_first_action_range_a": near_range,
                     "near_optimal_multivalued": bool(np.isfinite(near_range) and near_range > gates.maximum_near_optimal_action_range_p95_a)})
    summary = pd.DataFrame(rows); summary.to_csv(data_dir / "multistart_state_summary.csv", index=False)
    p95 = float(summary.near_optimal_first_action_range_a.quantile(.95)); fraction = float(summary.near_optimal_multivalued.mean())
    checks = {"exact_audit_contract": len(solutions) == config.data.audit_state_count*config.data.warm_starts_per_state,
              "all_optimizer_success": bool((summary.optimizer_success_count == config.data.warm_starts_per_state).all()),
              "all_prediction_feasible": bool((summary.prediction_feasible_count == config.data.warm_starts_per_state).all()),
              "zero_fallback": bool((summary.fallback_count == 0).all()),
              "multivalued_fraction": fraction <= gates.maximum_multivalued_state_fraction,
              "action_range_p95": p95 <= gates.maximum_near_optimal_action_range_p95_a}
    metrics = {"state_count": len(summary), "warm_starts_per_state": config.data.warm_starts_per_state,
               "near_optimal_multivalued_fraction": fraction, "near_optimal_first_action_range_p95_a": p95,
               "maximum_near_optimal_first_action_range_a": float(summary.near_optimal_first_action_range_a.max()),
               "checks": checks, "success": bool(all(checks.values()))}
    return solutions, summary, metrics


def _fit_network(inherited: Any, train: pd.DataFrame, seed: int) -> tuple[TinyANN, dict[str, Any]]:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    xs = StandardScaler().fit(train[list(FEATURES)]); ys = StandardScaler().fit(train[["teacher_current_a"]])
    estimator = MLPRegressor(hidden_layer_sizes=inherited.network.hidden_layer_sizes,
                             activation=inherited.network.activation, solver="lbfgs",
                             alpha=inherited.network.regularization_alpha,
                             max_iter=inherited.network.maximum_iterations,
                             tol=inherited.network.convergence_tolerance, random_state=seed)
    started = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(xs.transform(train[list(FEATURES)]), ys.transform(train[["teacher_current_a"]]).ravel())
    network = TinyANN(FEATURES, xs.mean_, xs.scale_, float(ys.mean_[0]), float(ys.scale_[0]),
                      tuple(np.asarray(v) for v in estimator.coefs_), tuple(np.asarray(v) for v in estimator.intercepts_),
                      inherited.mpc.current_bounds_a[0], inherited.mpc.current_bounds_a[1])
    return network, {"fit_time_s": perf_counter()-started, "optimization_iterations": int(estimator.n_iter_),
                     "warning_count": len(caught), "warnings": sorted({str(v.message) for v in caught})}


def _train(config: Phase7ALevel2Config, model: Level2Model, dataset: pd.DataFrame, output: Path, resume: bool) -> tuple[pd.DataFrame, dict[int, TinyANN]]:
    path = output / "dnn_offline_metrics.csv"; model_dir = output / "models"; model_dir.mkdir(exist_ok=True)
    records = {int(v["seed"]): v for v in (pd.read_csv(path).to_dict("records") if resume and path.exists() else [])}
    networks = {}; train = dataset[dataset.split == "train"]
    frames = {name: dataset[dataset.split == name] for name in ("train", "validation", "test", "terminal_test")}
    for seed in model.inherited.network.initialization_seeds:
        model_path = model_dir / f"level2_deep_lbfgs_seed_{seed}.npz"
        if seed in records and model_path.exists(): networks[seed] = TinyANN.load(model_path); continue
        network, optimization = _fit_network(model.inherited, train, seed); network.save(model_path); networks[seed] = network
        record: dict[str, Any] = {"seed": seed, "architecture": "3-32-32-16-1", "optimizer": "lbfgs",
                                  "parameter_count": network.parameter_count, **optimization}
        for split, frame in frames.items():
            prediction = np.asarray(network.predict(frame[list(FEATURES)].to_numpy(float)))
            metrics = _regression_metrics(frame.teacher_current_a.to_numpy(float), prediction)
            error = prediction - frame.teacher_current_a.to_numpy(float); low = frame.teacher_current_a.to_numpy(float) <= config.data.low_current_threshold_a
            metrics.update({"bias_a": float(error.mean()), "low_current_sample_count": int(low.sum()),
                            "low_current_bias_a": float(error[low].mean()) if low.any() else float("nan")})
            record.update({f"{split}_{key}": value for key, value in metrics.items()})
        records[seed] = record; pd.DataFrame(records.values()).sort_values("seed").to_csv(path, index=False)
        print(f"Level 2 DNN seed {seed}: global={100*record['test_nrmse']:.3f}% terminal={100*record['terminal_test_nrmse']:.3f}%", flush=True)
    return pd.DataFrame(records.values()).sort_values("seed"), networks


def _closed_initial_states(config: Phase7ALevel2Config, model: Level2Model) -> pd.DataFrame:
    d = config.data; records = []; candidate = 11
    while len(records) < d.closed_loop_trajectory_count:
        soc = d.closed_loop_soc_bounds[0] + np.ptp(d.closed_loop_soc_bounds)*_van_der_corput(candidate,2)
        v1 = d.closed_loop_v1_bounds_v[0] + np.ptp(d.closed_loop_v1_bounds_v)*_van_der_corput(candidate,3)
        v2 = d.closed_loop_v2_bounds_v[0] + np.ptp(d.closed_loop_v2_bounds_v)*_van_der_corput(candidate,5); candidate += 1
        state = Level2State(soc,v1,v2)
        if model.terminal_voltage(state,0.0) > model.inherited.mpc.terminal_voltage_max_v-config.data.initial_voltage_margin_v: continue
        records.append({"trajectory_id":f"level2_closed_{len(records):02d}","initial_soc":soc,
                        "initial_polarization_1_v":v1,"initial_polarization_2_v":v2})
    return pd.DataFrame(records)


def _rollout(config: Phase7ALevel2Config, model: Level2Model, controller: Level2MPC | TinyANN,
             initial: pd.Series, kind: str, seed: int | None=None) -> list[dict[str,Any]]:
    state = Level2State(float(initial.initial_soc),float(initial.initial_polarization_1_v),float(initial.initial_polarization_2_v)); rows=[]
    for step in range(config.data.maximum_closed_loop_steps):
        started=perf_counter()
        if kind=="mpc":
            result=controller.solve(state)  # type: ignore[union-attr]
            if not result.optimizer_success or not result.prediction_feasible: break
            current,elapsed=result.current_a,result.solve_time_s
        else:
            current=float(controller.predict(np.asarray([state.soc,state.polarization_1_v,state.polarization_2_v])))  # type: ignore[union-attr]
            elapsed=perf_counter()-started
        voltage=model.terminal_voltage(state,current); next_state=model.step(state,current)
        rows.append({"controller":kind,"seed":seed,"trajectory_id":initial.trajectory_id,"step_index":step,
                     "soc":state.soc,"polarization_1_v":state.polarization_1_v,"polarization_2_v":state.polarization_2_v,
                     "current_a":current,"terminal_voltage_v":voltage,"next_soc":next_state.soc,"elapsed_s":elapsed})
        state=next_state
        if state.soc>=model.inherited.mpc.target_soc-5e-4: break
    return rows


def _closed_loop(config: Phase7ALevel2Config, model: Level2Model, networks: dict[int,TinyANN], data_dir:Path, resume:bool) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    path=data_dir/"closed_loop_trajectories.csv"; metrics_path=data_dir/"closed_loop_metrics.csv"; diag_path=data_dir/"closed_loop_diagnostics.csv"
    if resume and path.exists() and metrics_path.exists() and diag_path.exists(): return pd.read_csv(path),pd.read_csv(metrics_path),pd.read_csv(diag_path)
    initial=_closed_initial_states(config,model); initial.to_csv(data_dir/"closed_loop_initial_states.csv",index=False)
    rows=[]
    for index,(_,state) in enumerate(initial.iterrows(),start=1):
        rows.extend(_rollout(config,model,Level2MPC(model),state,"mpc")); print(f"Level 2 closed-loop MPC {index}/{len(initial)}",flush=True)
    for seed,network in networks.items():
        for _,state in initial.iterrows(): rows.extend(_rollout(config,model,network,state,"dnn",seed))
    trajectories=pd.DataFrame(rows); trajectories.to_csv(path,index=False); teacher=trajectories[trajectories.controller=="mpc"]
    metric_rows=[]; diag_rows=[]; dt=config.model.sample_period_s; threshold=model.inherited.mpc.target_soc-5e-4
    for seed in networks:
        dnn=trajectories[(trajectories.controller=="dnn")&(trajectories.seed==seed)]; per=[]
        for trajectory_id,mpc_group in teacher.groupby("trajectory_id"):
            dnn_group=dnn[dnn.trajectory_id==trajectory_id]
            paired=mpc_group[["step_index","current_a"]].merge(dnn_group[["step_index","current_a"]],on="step_index",suffixes=("_mpc","_dnn"))
            nrmse=_regression_metrics(paired.current_a_mpc.to_numpy(),paired.current_a_dnn.to_numpy())["nrmse"]
            gap=abs(len(dnn_group)-len(mpc_group))/len(mpc_group); mpc_cont=continuous_crossing_time_s(mpc_group,threshold,dt); dnn_cont=continuous_crossing_time_s(dnn_group,threshold,dt)
            per.append((nrmse,gap,float(dnn_group.next_soc.iloc[-1])>=threshold))
            diag_rows.append({"seed":seed,"trajectory_id":trajectory_id,"signed_step_difference":len(dnn_group)-len(mpc_group),
                              "discrete_arrival_time_difference_s":(len(dnn_group)-len(mpc_group))*dt,
                              "continuous_crossing_time_difference_s":dnn_cont-mpc_cont,
                              "cumulative_charge_error_ah":float((dnn_group.current_a.sum()-mpc_group.current_a.sum())*dt/3600.0)})
        voltage_violation=np.maximum(dnn.terminal_voltage_v.to_numpy()-model.inherited.mpc.terminal_voltage_max_v,0.0)
        current_violation=np.maximum(dnn.current_a.to_numpy()-model.inherited.mpc.current_bounds_a[1],0.0)
        metric_rows.append({"seed":seed,"mean_current_nrmse":float(np.mean([v[0] for v in per])),"maximum_current_nrmse":float(np.max([v[0] for v in per])),
                            "mean_charge_time_gap_fraction":float(np.mean([v[1] for v in per])),"target_reach_fraction":float(np.mean([v[2] for v in per])),
                            "maximum_voltage_violation_v":float(np.max(voltage_violation)),"maximum_current_violation_a":float(np.max(current_violation)),
                            "mpc_time_s":float(teacher.elapsed_s.sum()),"dnn_time_s":float(dnn.elapsed_s.sum()),
                            "speedup":float(teacher.elapsed_s.sum()/dnn.elapsed_s.sum())})
    metrics=pd.DataFrame(metric_rows); diagnostics=pd.DataFrame(diag_rows); metrics.to_csv(metrics_path,index=False); diagnostics.to_csv(diag_path,index=False)
    return trajectories,metrics,diagnostics


def _plots(output:Path,dataset:pd.DataFrame,audit:pd.DataFrame,offline:pd.DataFrame,closed:pd.DataFrame)->None:
    figure_dir=output/"figures"; figure_dir.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2)); scatter=axes[0].scatter(dataset.state_soc,dataset.state_polarization_1_v+dataset.state_polarization_2_v,c=dataset.teacher_current_a,s=6,cmap="viridis")
    axes[0].set(xlabel="SOC",ylabel="V1+V2 [V]",title="2RC teacher coverage"); fig.colorbar(scatter,ax=axes[0],label="Current [A]")
    axes[1].hist(audit.near_optimal_first_action_range_a,bins=25); axes[1].axvline(.05,color="red",linestyle="--"); axes[1].set(xlabel="Near-optimal first-action range [A]",title="100×15 audit")
    fig.tight_layout(); fig.savefig(figure_dir/"teacher_and_audit.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2)); axes[0].bar(offline.seed.astype(str),100*offline.test_nrmse,label="Global"); axes[0].bar(offline.seed.astype(str),100*offline.terminal_test_nrmse,alpha=.6,label="Terminal"); axes[0].axhline(1,color="red",linestyle="--"); axes[0].legend(); axes[0].set(ylabel="Frozen-test NRMSE [%]")
    axes[1].bar(closed.seed.astype(str),100*closed.mean_current_nrmse); axes[1].axhline(1,color="red",linestyle="--"); axes[1].set(ylabel="Closed-loop NRMSE [%]")
    fig.tight_layout(); fig.savefig(figure_dir/"five_seed_validation.png",dpi=180); plt.close(fig)


def _write_report(path:Path,p:dict[str,Any])->None:
    t,a,o,c,d=p["teacher"],p["teacher_audit"],p["offline"],p["closed_loop"],p["decision"]
    path.write_text(f"""# Phase 7A Level 2：2RC 三状态 pure DNN 验证报告

## 结论

Level 2 判定：**{d['conclusion']}**。允许进入 Level 3：**{'是' if d['proceed_to_level3'] else '否'}**。

## 单因素边界

- 状态仅从 `(SOC,Vp)` 扩展为 `(SOC,V1,V2)`；使用项目辨识的两条极化支路。
- 仅保留 0–10 A 电流边界和 4.20 V 端电压上限；无硬斜率、温度、DFN、扰动或 Phase 5A 压力场。
- 全域 240×8 与末端 160×24 从一开始共同生成；全域冻结测试 36 条，末端冻结测试 20 条。

## 教师与确定性

- 教师接受：{t['accepted_trajectories']}/{t['attempted_trajectories']}，样本 {t['sample_count']}，低电流标签 {t['low_current_label_count']}。
- 100×15 审计：{a['success']}；多值比例 {100*a['near_optimal_multivalued_fraction']:.2f}%；动作极差 P95 {a['near_optimal_first_action_range_p95_a']:.4e} A。

## 深层 LBFGS 五种子

- 全域冻结测试 NRMSE：{100*o['global_test_nrmse_min']:.4f}%–{100*o['global_test_nrmse_max']:.4f}%。
- 末端冻结测试 NRMSE：{100*o['terminal_test_nrmse_min']:.4f}%–{100*o['terminal_test_nrmse_max']:.4f}%。
- 同模型闭环电流 NRMSE：{100*c['current_nrmse_min']:.4f}%–{100*c['current_nrmse_max']:.4f}%。
- 最大平均充电时间偏差：{100*c['maximum_charge_time_gap_fraction']:.4f}%；最低到达率 {100*c['minimum_target_reach_fraction']:.1f}%；最低加速 {c['minimum_speedup']:.1f}×。

## 门槛

```json
{json.dumps(d['checks'],ensure_ascii=False,indent=2)}
```

本报告不包含斜率约束、热状态、DFN 或扰动，因此结论只回答“第二极化时间尺度是否使 pure DNN 首次失效”。
""",encoding="utf-8")


def _json_default(value: object) -> object:
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    raise TypeError(type(value).__name__)


def run_phase7a_level2(config:Phase7ALevel2Config,project_root:str|Path,resume:bool=False)->dict[str,Any]:
    root=Path(project_root).resolve(); data_dir=root/"data"/"phase7a_level2_2rc"; output=root/"outputs"/"phase7a_level2_2rc"; data_dir.mkdir(parents=True,exist_ok=True); output.mkdir(parents=True,exist_ok=True)
    _validate_parameters(config,root); inherited=load_phase7a_level1_config(root/config.source_level1_config); model=Level2Model(config,inherited,root)
    global_data,global_audit=_generate_domain(config,model,config.data.global_domain,"global","test",0,data_dir,resume)
    terminal_data,terminal_audit=_generate_domain(config,model,config.data.terminal_domain,"terminal","terminal_test",1000,data_dir,resume)
    dataset=pd.concat([global_data,terminal_data],ignore_index=True); dataset.to_csv(data_dir/"combined_teacher_dataset.csv",index=False)
    attempts=pd.concat([global_audit,terminal_audit],ignore_index=True); acceptance=float(attempts.teacher_accepted.astype(bool).mean()); low_count=int((dataset.teacher_current_a<=config.data.low_current_threshold_a).sum())
    expected_steps={**{v:config.data.global_domain.trajectory_steps for v in global_data.trajectory_id.unique()},**{v:config.data.terminal_domain.trajectory_steps for v in terminal_data.trajectory_id.unique()}}
    actual_steps=dataset.groupby("trajectory_id").size().to_dict()
    teacher_checks={"acceptance":bool(acceptance>=inherited.gates.minimum_teacher_acceptance_fraction),"complete_accepted_trajectories":bool(all(actual_steps[k]==v for k,v in expected_steps.items())),
                    "trajectory_split_isolation":bool(dataset.groupby("trajectory_id").split.nunique().max()==1),"low_current_coverage":bool(low_count>=config.data.minimum_low_current_label_count),
                    "zero_fallback":bool((dataset.teacher_used_fallback==False).all())}
    teacher={"attempted_trajectories":len(attempts),"accepted_trajectories":int(attempts.teacher_accepted.astype(bool).sum()),"acceptance_fraction":acceptance,"sample_count":len(dataset),"low_current_label_count":low_count,"checks":teacher_checks,"success":bool(all(teacher_checks.values()))}
    _,audit_table,audit_metrics=_teacher_audit(config,model,dataset,data_dir,resume); payload={"study_name":config.study_name,"configuration":asdict(config),"inherited_level1_configuration":asdict(inherited),"teacher":teacher,"teacher_audit":audit_metrics}
    teacher_gate=teacher["success"] and audit_metrics["success"]
    if teacher_gate:
        offline,networks=_train(config,model,dataset,output,resume); offline_checks={"five_global_test_seeds":bool((offline.test_nrmse<inherited.gates.offline_nrmse_max).all()),"five_terminal_test_seeds":bool((offline.terminal_test_nrmse<inherited.gates.offline_nrmse_max).all())}
        payload["offline"]={"seed_count":len(offline),"global_test_nrmse_min":float(offline.test_nrmse.min()),"global_test_nrmse_max":float(offline.test_nrmse.max()),"terminal_test_nrmse_min":float(offline.terminal_test_nrmse.min()),"terminal_test_nrmse_max":float(offline.terminal_test_nrmse.max()),"checks":offline_checks,"success":bool(all(offline_checks.values()))}
        _,closed,diagnostics=_closed_loop(config,model,networks,data_dir,resume); closed_checks={"all_seed_current_nrmse":bool((closed.mean_current_nrmse<inherited.gates.closed_loop_current_nrmse_max).all()),"all_seed_charge_time_gap":bool((closed.mean_charge_time_gap_fraction<inherited.gates.charge_time_gap_fraction_max).all()),"all_seed_target_reach":bool((closed.target_reach_fraction>=inherited.gates.minimum_target_reach_fraction).all()),"zero_voltage_violation":bool((closed.maximum_voltage_violation_v<=inherited.gates.maximum_constraint_violation).all()),"zero_current_violation":bool((closed.maximum_current_violation_a<=inherited.gates.maximum_constraint_violation).all()),"all_seed_speedup":bool((closed.speedup>inherited.gates.minimum_speedup).all())}
        payload["closed_loop"]={"current_nrmse_min":float(closed.mean_current_nrmse.min()),"current_nrmse_max":float(closed.mean_current_nrmse.max()),"maximum_charge_time_gap_fraction":float(closed.mean_charge_time_gap_fraction.max()),"minimum_target_reach_fraction":float(closed.target_reach_fraction.min()),"maximum_voltage_violation_v":float(closed.maximum_voltage_violation_v.max()),"minimum_speedup":float(closed.speedup.min()),"signed_step_difference_range":[int(diagnostics.signed_step_difference.min()),int(diagnostics.signed_step_difference.max())],"maximum_absolute_cumulative_charge_error_ah":float(diagnostics.cumulative_charge_error_ah.abs().max()),"checks":closed_checks,"success":bool(all(closed_checks.values()))}
        _plots(output,dataset,audit_table,offline,closed)
    checks={"teacher_determinism_passed":teacher_gate,"dual_offline_tests_passed":bool(payload.get("offline",{}).get("success",False)),"same_model_closed_loop_passed":bool(payload.get("closed_loop",{}).get("success",False))}; success=bool(all(checks.values()))
    payload["decision"]={"checks":checks,"level2_success":success,"proceed_to_level3":success,"conclusion":"Level 2 通过" if success else "Level 2 未通过，停止增加复杂度"}; payload["status"]="completed"; payload["success"]=success
    (output/"metrics.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=_json_default),encoding="utf-8"); _write_report(output/"PHASE7A_LEVEL2_中文实验报告.md",payload); print(json.dumps(payload["decision"],ensure_ascii=False,default=_json_default),flush=True); return payload
