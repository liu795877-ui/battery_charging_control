# Phase 6C: constraint-regime learning

Phase 6C is isolated from the frozen Phase 6B baseline. Phase 6C-1 reuses the exact
7024 supervised samples and the original trajectory-level train/validation/test split.
It does not generate teacher data or run a DFN closed loop.

Run Phase 6C-1 with:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
python -m battery_fast_charge.phase6c1_cli
```

The run matrix is checkpointed after every architecture/optimizer/seed combination in
`data/phase6c_constraint_regime_learning/c1_ablation/ablation_runs.csv`.

Phase 6C-2 and Phase 6C-3 entry points are:

```powershell
python -m battery_fast_charge.phase6c2_cli
python -m battery_fast_charge.phase6c3_cli
```

The final gate result is recorded in `outputs/phase6c_report.md`. Phase 6D must not run
when `phase6c_acceptance.proceed_to_phase6d` is false.
