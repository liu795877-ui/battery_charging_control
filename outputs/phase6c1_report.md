# Phase 6C-1: optimization versus generalization ablation

## Frozen baseline

- Phase 6B commit: `879ad0f`
- Dataset SHA-256: `640b45c017594c3a8f954b498a99058d2e5cc8ecf6cff88425d5df94815de5da`
- Samples: 7024
- Split samples: {'train': 5616, 'validation': 704, 'test': 704}
- Split trajectories: {'train': 702, 'validation': 88, 'test': 88}
- No teacher data were generated and no split was changed.

## Five-seed results

| Architecture | Optimizer | Train NRMSE mean ± SD | Validation NRMSE mean ± SD | Test NRMSE mean ± SD | Best test NRMSE | Best validation seed |
|---|---|---:|---:|---:|---:|---:|
| 16-16 | adam | 5.246% ± 0.019% | 5.307% ± 0.050% | 5.717% ± 0.108% | 5.634% | 22 |
| 16-16 | adam_lbfgs | 4.975% ± 0.046% | 5.491% ± 0.153% | 6.128% ± 0.234% | 5.734% | 101 |
| 16-16 | lbfgs | 4.940% ± 0.073% | 5.681% ± 0.295% | 6.103% ± 0.283% | 5.688% | 73 |
| 32-32-16 | adam | 4.017% ± 0.111% | 6.969% ± 0.225% | 7.863% ± 0.300% | 7.367% | 73 |
| 32-32-16 | adam_lbfgs | 3.089% ± 0.111% | 10.008% ± 0.186% | 10.684% ± 0.622% | 10.184% | 73 |
| 32-32-16 | lbfgs | 3.254% ± 0.103% | 9.078% ± 0.459% | 11.833% ± 0.879% | 10.823% | 101 |
| 64-64-32 | adam | 2.426% ± 0.137% | 9.177% ± 0.580% | 10.368% ± 0.834% | 9.626% | 101 |
| 64-64-32 | adam_lbfgs | 1.467% ± 0.128% | 13.369% ± 0.952% | 14.831% ± 0.636% | 14.028% | 22 |
| 64-64-32 | lbfgs | 2.090% ± 0.091% | 13.002% ± 1.420% | 15.362% ± 1.357% | 14.059% | 22 |

## Maximum absolute error across seeds

| Architecture | Optimizer | Train mean ± SD / best | Validation mean ± SD / best | Test mean ± SD / best |
|---|---|---:|---:|---:|
| 16-16 | adam | 1.855 ± 0.037 / 1.812 A | 1.801 ± 0.058 / 1.727 A | 1.854 ± 0.125 / 1.713 A |
| 16-16 | adam_lbfgs | 2.046 ± 0.070 / 1.945 A | 2.050 ± 0.137 / 1.927 A | 2.461 ± 0.379 / 2.002 A |
| 16-16 | lbfgs | 1.980 ± 0.044 / 1.932 A | 2.406 ± 0.343 / 1.952 A | 2.586 ± 0.701 / 2.143 A |
| 32-32-16 | adam | 1.973 ± 0.161 / 1.822 A | 3.038 ± 0.376 / 2.673 A | 3.625 ± 0.524 / 3.008 A |
| 32-32-16 | adam_lbfgs | 1.796 ± 0.077 / 1.725 A | 5.293 ± 0.601 / 4.423 A | 4.811 ± 0.825 / 3.770 A |
| 32-32-16 | lbfgs | 1.840 ± 0.074 / 1.711 A | 4.485 ± 0.336 / 4.164 A | 5.859 ± 0.748 / 4.786 A |
| 64-64-32 | adam | 1.700 ± 0.105 / 1.579 A | 4.289 ± 0.565 / 3.681 A | 4.024 ± 0.718 / 3.178 A |
| 64-64-32 | adam_lbfgs | 1.476 ± 0.099 / 1.357 A | 6.497 ± 0.731 / 5.524 A | 6.159 ± 0.291 / 5.673 A |
| 64-64-32 | lbfgs | 1.598 ± 0.081 / 1.469 A | 6.600 ± 0.745 / 5.982 A | 6.761 ± 0.762 / 5.913 A |

## Decision

- Selected group: 16-16 / adam
- Primary diagnosis: `generalization_or_coverage_limited`
- Selected-group generalization gap: 0.471 percentage points
- Stop pure network scaling: True
- Generalization-limited groups: 8
- Seed-unstable groups: 3

The diagnosis applies only to the frozen Phase 6B distribution. It does not yet establish 25 °C nominal closed-loop acceptance.
