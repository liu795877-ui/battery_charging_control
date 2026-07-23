# Phase 6C final report

Phase 6C completed all three planned stages while preserving the Phase 6B frozen test
set. The final decision is **do not proceed to Phase 6D**.

## Evidence summary

- Phase 6C-1: 45 optimizer/capacity/seed runs identified generalization and coverage as
  the primary limitation. The exact Phase 6B result was reproduced.
- Phase 6C-2: 800/800 targeted MPC trajectories were accepted, producing 6400 new
  training/new-validation samples without changing the original test set.
- Phase 6C-3: five-seed 25 °C DFN validation was completed for pure, projected, and
  structured-delta DNN controllers.

| Controller | Frozen-test NRMSE mean ± SD | 25 °C closed-loop NRMSE mean ± SD | Charge-time gap mean | Strict passes |
|---|---:|---:|---:|---:|
| pure DNN | 5.551% ± 0.086% | 4.969% ± 0.340% | 3.701% | 0/5 |
| projected DNN | 5.522% ± 0.071% | 5.179% ± 0.207% | 3.608% | 0/5 |
| structured delta DNN | 7.990% ± 0.072% | 3.511% ± 0.456% | 3.795% | 0/5 |

The structured route reached the declared 2%-3% stage-progress region for one seed and
was close for others, but did not meet the 1% frozen-test or closed-loop thresholds and
did not meet the 2% charge-time threshold. It is an explicit improvement method, not a
pure paper-style DNN result.

The projected controller was safe in all five nominal runs but did not improve current
tracking enough. The pure controller remains the valid test of paper-style direct MPC
imitation and did not pass.

## Gate outcome

- Frozen-test NRMSE below 1%: failed for all controllers.
- 25 °C DFN closed-loop NRMSE below 1%: failed for all controllers.
- Charge-time gap below 2%: failed on five-seed means and every strict combined run.
- Inference speedup above 100x: passed (all controller means are above 7700x).
- Majority of five seeds passing all requirements: failed (0/5 for every controller).
- Proceed to 15/30 °C, Phase 5A, or cross-battery validation: **no**.
