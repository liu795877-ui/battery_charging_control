# Phase 6C: constraint-regime learning and nominal closed-loop revalidation

Phase 6C keeps the Phase 6B battery, 25 °C MPC teacher, 7024-sample baseline,
trajectory-level split, and frozen 704-sample test set unchanged. The immutable
baseline is Git commit `879ad0f`, tagged `phase6b-baseline`.

## Phase 6C-1: optimization versus generalization

The ablation used three architectures, three optimizers, and five initialization
seeds (45 runs). Every group retained the same Phase 6B samples and splits.

The 16-16 Adam group had the best mean validation NRMSE (5.307%) and mean frozen-test
NRMSE (5.717%). Increasing capacity reduced training error but worsened held-out error:

- 32-32-16 LBFGS: 3.254% train, 11.833% test;
- 64-64-32 LBFGS: 2.090% train, 15.362% test;
- 64-64-32 Adam-to-LBFGS: 1.467% train, 14.831% test.

The Phase 6B 32-32-16/LBFGS/seed-22 result was reproduced exactly (3.128% train,
9.192% validation, 11.494% test). The primary diagnosis is therefore
generalization/data-coverage limitation. Optimizer choice matters, but does not explain
the frozen-test error by itself. All group-mean test NRMSE values remained above 5%, so
pure network scaling stopped.

## Phase 6C-2: targeted teacher data

The MPC accepted all 800 requested trajectories and produced 6400 samples:

- 500 `targeted_boundary_sampling` trajectories;
- 300 `closed_loop_DAgger` trajectories selected by distance from the original
  training distribution.

The new data contain 999 current-change, 1261 temperature, and 919 voltage active
samples. They enter only the Phase 6C training set (5440 samples) and new validation
set (960 samples). The original Phase 6B validation and test sets are unchanged.
Current-upper-bound activation remains absent from the new data, so that sparse regime
still cannot support a reliable boundary-learning claim.

## Phase 6C-3: controller comparison at 25 °C

All controllers used the Phase 6C-1 selected 16-16 Adam setup and five seeds. The pure
and projected controllers predict absolute current. The structured controller predicts
a latent current increment and applies

\[
I_k=I_{k-1}+2\tanh(\hat z_k),
\]

which guarantees a per-step current change no greater than 2 A but does not itself
guarantee the absolute 0-10 A current range.

Five-seed mean results:

| Controller | Frozen-test NRMSE | 25 °C closed-loop NRMSE | Charge-time gap | Strict passing seeds |
|---|---:|---:|---:|---:|
| pure DNN | 5.551% | 4.969% | 3.701% | 0/5 |
| projected DNN | 5.522% | 5.179% | 3.608% | 0/5 |
| structured delta DNN | 7.990% | 3.511% | 3.795% | 0/5 |

The structured controller is a repeatable closed-loop improvement but not a validation
pass. One structured seed exceeded the absolute current limit by 0.783 A, although all
five seeds respected the 2 A slew limit. Projected DNN had no serious violation in all
five seeds, but its accuracy and charge-time gap failed the strict thresholds. Pure DNN
had serious violations in three seeds.

## Decision

No controller seed passed the complete offline, closed-loop, charge-time, safety, and
speed gate. The paper-style pure DNN method did not fully transfer under this
configuration. The structured and projected variants remain engineering improvement
directions, not pure-paper validation results.

Phase 6D is blocked by the declared 25 °C nominal gate. Do not run the 15/30 °C anchors,
Phase 5A perturbations, or cross-battery validation unless a later Phase 6C method first
passes the unchanged nominal criteria.
