# Phase 5B-0.5 MPC recovery and feasibility recheck

## Outcome

The representative study completed 30 closed-loop runs: 15 frozen scenarios with both
nominal-parameter recovery MPC and parameter-oracle recovery MPC. The representative gate
failed, so Phase 5B-1 remains disabled and the full 69-scenario recovery sweep is not
authorized.

## Acceptance checks

| Check | Result |
|---|---:|
| Non-conflict fallback slew violations equal zero | Pass |
| Matched nominal feasible gain at least +2 | Fail (-3) |
| Oracle recovery not weaker than nominal recovery | Fail (1 vs 2) |
| Failure categories fully auditable | Pass |

## Controller summary

| Controller | Phase 5B-0 baseline feasible | Recovery feasible | Gain | Prediction-domain infeasible decisions | Hard-safety/slew conflicts | Ordinary fallback slew violations |
|---|---:|---:|---:|---:|---:|---:|
| Nominal recovery MPC | 5/15 | 2/15 | -3 | 1220 | 487 | 0 |
| Oracle recovery MPC | 1/15 | 1/15 | 0 | 1427 | 544 | 0 |

The only fully feasible runs were nominal recovery on `lhs_029`, nominal recovery on the
nominal scenario, and oracle recovery on the nominal scenario. Both controller variants
completed the nominal 25 C case without a classified hard-safety/slew conflict.

## Interpretation

The new fallback correctly enforces the normal action interval

\[
\mathcal I_k =
\left[\max(0,I_{k-1}-2),\min(10,I_{k-1}+2)\right]\ \mathrm{A}.
\]

Its success is local: ordinary fallback no longer introduces slew violations. It does not
restore representative-domain feasibility. The dominant events are now explicitly labeled
as prediction-domain infeasibility and genuine one-step conflict between the hard safety
threshold and the 2 A/5 s slew limit. The zero count for numerical failures recovered by a
fully feasible retained candidate indicates that retained candidates did not solve the
dominant failure mode in this set.

The reduction from five matched baseline-feasible nominal cases to two recovery-feasible
cases must not be interpreted as a simple optimization regression without further audit:
Phase 5B-0 and Phase 5B-0.5 use different complete-feasibility contracts. The next step is
to compare those contracts per scenario and reconcile the prediction constraints with the
4.2 V and 35 C hard-safety thresholds.

## Decision

- Do not run the full 69-scenario Phase 5B-0.5 sweep.
- Do not start Phase 5B-1 hybrid control.
- Do not expand to cross-battery or Phase 5A robustness claims.
- First perform a per-scenario contract-consistency audit on the five Phase 5B-0 nominal
  teacher-feasible cases and the two Phase 5B-0.5 nominal-recovery feasible cases.

The executed review notebook is
`notebooks/11_phase5b05_mpc_recovery_executed.ipynb`.
