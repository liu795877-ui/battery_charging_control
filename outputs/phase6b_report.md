# Phase 6B: why the pure DNN did not learn MPC well

## Main result

Phase 6B is diagnostic. It does not replace the Phase 6A pure-DNN conclusion.
- Accepted teacher trajectories: 878
- Unfolded samples: 7024
- Selected DNN architecture: [5, 32, 32, 16, 1]
- Selected optimizer iterations: 2500
- Selected optimizer reached limit: True
- Test current NRMSE: 11.494%
- Pure DNN closed-loop NRMSE: 5.228%
- Projected DNN closed-loop NRMSE: 7.023%
- Pure DNN slew violation: 2.5810 A
- Projected DNN slew violation: 0.0000 A

## Worst held-out error partitions

- previous_current / (2.0, 5.0]: RMSE 1.5138 A, NRMSE 15.138%, n=265
- previous_current / (-0.001, 2.0]: RMSE 1.4953 A, NRMSE 14.953%, n=54
- slew_near_boundary / near_slew_boundary: RMSE 1.4762 A, NRMSE 14.762%, n=86
- slew_active / slew_active: RMSE 1.4751 A, NRMSE 14.751%, n=85
- temperature / (24.999, 27.0]: RMSE 1.4238 A, NRMSE 14.238%, n=159
- soc / (0.25, 0.4]: RMSE 1.3170 A, NRMSE 13.170%, n=170
- soc / (0.099, 0.25]: RMSE 1.2838 A, NRMSE 12.838%, n=130
- any_constraint / near_any_constraint: RMSE 1.2826 A, NRMSE 12.826%, n=163

## Interpretation rule

If projection fixes only constraint violations but not current NRMSE, the network did not learn the teacher map accurately. If projection also reduces NRMSE materially, part of the Phase 6A error came from raw outputs that ignored basic controller bounds.
