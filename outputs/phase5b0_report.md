# Phase 5B-0 result

The frozen 69-scenario Phase 5A reduced-model set was evaluated with nominal MPC and a
parameter-oracle MPC. Both controllers used the same noisy state-estimation sequence;
the oracle knew scenario parameters but did not receive perfect state measurements.

Only 5/69 nominal-MPC scenarios and 1/69 oracle-MPC scenarios satisfied the complete
teacher feasibility contract (true target completion, true physical safety, optimizer
success fraction >=95%, and zero fallback). Completion fractions were 86.96% and 85.51%,
while physical-safety fractions were 7.25% and 1.45%.

The scenario classification contained 35 teacher-and-ANN-infeasible cases, 5 cases where
both teachers and the Phase 5A safety-layer ANN were feasible, and 29 unresolved cases in
which the safety-layer ANN appeared feasible while both unprotected MPC teachers failed.
There were no nominal-failed/oracle-feasible cases.

Decision: the current Phase 5A stress domain is not a valid unfiltered ANN imitation
domain. Phase 5B-1 should focus on constrained ANN-MPC hybrid control and explicitly
separate teacher-infeasible scenarios from ANN imitation failures. No DFN anchors,
cross-battery tests, or battery changes were run in Phase 5B-0.
