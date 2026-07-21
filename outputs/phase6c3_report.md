# Phase 6C-3: controller-output comparison and 25 °C nominal validation

## Data isolation

- Frozen Phase 6B samples: 7024
- Added Phase 6C training samples: 5440
- New validation samples: 960
- The original 704-sample test set remains unchanged and is used for every seed.

## Five-seed comparison

| Controller | Train NRMSE | Original validation NRMSE | New validation NRMSE | Frozen test NRMSE | 25 °C closed-loop NRMSE | Time gap | Passing seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| projected_dnn | 5.528% ± 0.026% | 5.329% ± 0.040% | 5.863% ± 0.028% | 5.522% ± 0.071% | 5.179% ± 0.207% | 3.608% | 0/5 |
| pure_dnn | 5.537% ± 0.027% | 5.348% ± 0.046% | 5.869% ± 0.030% | 5.551% ± 0.086% | 4.969% ± 0.340% | 3.701% | 0/5 |
| structured_delta_dnn | 7.804% ± 0.096% | 7.281% ± 0.106% | 8.775% ± 0.096% | 7.990% ± 0.072% | 3.511% ± 0.456% | 3.795% | 0/5 |

## Frozen-test maximum absolute error

| Controller | Mean ± SD | Best seed |
|---|---:|---:|
| projected_dnn | 1.874 ± 0.127 A | 1.704 A |
| pure_dnn | 1.948 ± 0.235 A | 1.704 A |
| structured_delta_dnn | 2.337 ± 0.168 A | 2.171 A |

## Gate decision

- 25 °C nominal majority gate passed: False
- Controllers with a passing majority: []
- Proceed to Phase 6D: False

The structured-delta route is an explicit improvement method and is not counted as a pure paper-style DNN result.
