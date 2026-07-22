# Phase 5B-0.6 paired feasibility-contract audit

## Scope

The audit replayed `nominal`, `lhs_008`, `lhs_012`, `lhs_029`, and `lhs_056`. Original
MPC and recovery MPC shared the same noise innovations, initial state, plant/controller
parameters, frozen control-update schedule, and 3600 s cutoff. No ANN training or
69-scenario sweep was performed.

## Paired result

| Scenario | Original MPC | Recovery MPC |
|---|---:|---:|
| nominal | feasible | feasible |
| lhs_008 | feasible | infeasible |
| lhs_012 | feasible | infeasible |
| lhs_029 | feasible | feasible |
| lhs_056 | feasible | feasible |

The original MPC and recovery MPC both recovered all five frozen Phase 5B-0 feasible
scenarios under the corrected strict paired replay. Every pair used identical currents,
constraint slacks, and zero braking deficit to floating-point precision. The earlier
Phase 5B-0.5 loss therefore came from an inconsistent replay contract: the temporary audit
used the wrong random seed/scenario index and did not reuse the baseline cutoff/cap logic.

## Constraint relaxation evidence

For the two recovery-only failures, the maximum required relaxations of the audited
optimizer sequence were:

| Scenario | Voltage (V) | Temperature (C) | SOC | Slew (A/5 s) | Braking deficit (A) |
|---|---:|---:|---:|---:|---:|
| lhs_008 | ~0 | ~0 | ~0 | ~0 | 0 |
| lhs_012 | ~0 | ~0 | ~0 | ~0 | 0 |

These values are diagnostic slacks, not proposed safety-limit relaxations. The final
4.2 V, 35 C, and 2 A/5 s limits remain unchanged.

## Interpretation and decision

There is no failure in the corrected five-scenario paired contract. The earlier apparent
temperature/SOC/slew deficits were artifacts of inconsistent random-sequence indexing,
cutoff handling, and target-current cap handling. The audit nevertheless confirms that
the proposed slack and braking diagnostics are operational and should remain in future
full-domain studies.

Do not proceed to Phase 5B-1. The next implementation should:

1. freeze the corrected replay contract as the reference for any further recovery study;
2. preserve voltage, temperature, and slew as hard safety constraints;
3. add a prospective braking-feasibility constraint before hard-safety conflict occurs;
4. decide whether to soften only the terminal SOC objective after a separate contract audit;
5. rerun only these five paired scenarios after any contract change before expanding the domain.
