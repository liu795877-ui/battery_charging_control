# Phase 6B: DNN failure diagnosis

Phase 6B is an independent follow-up to Phase 6A. Its goal is not to make the
controller look better, but to explain where the pure DNN fails to imitate the
MPC teacher.

The experiment has three parts:

1. Generate a larger paper-style MPC teacher dataset from 1000 initial states.
2. Train larger three-hidden-layer DNN candidates, including 5-32-32-16-1 and
   5-64-64-32-1. The first diagnostic run keeps one seed and one regularization
   value so the experiment can finish in a reasonable time; the code path still
   supports expanding the grid later.
3. Compare pure DNN and projected DNN separately on the 25 degC nominal DFN
   closed loop.

The projected DNN is only a diagnostic control group. It clips the raw network
output to the current bound and the one-step current-slew bound. It is not mixed
into the pure-DNN conclusion.

Main outputs:

- `data/phase6b_dnn_failure_diagnosis/error_partition_diagnostics.csv`
- `data/phase6b_dnn_failure_diagnosis/pure_dnn_dfn_25c.csv`
- `data/phase6b_dnn_failure_diagnosis/projected_dnn_dfn_25c.csv`
- `outputs/metrics/phase6b_metrics.json`
- `outputs/figures/phase6b_error_partitions.png`
- `outputs/figures/phase6b_pure_vs_projected_25c.png`
