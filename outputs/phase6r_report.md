# Phase 6R result

Phase 6R corrected the teacher/control timing mismatch by re-solving MPC at every 5 s
state and retaining only the first action. The independent teacher consistency audit
passed with a maximum discrepancy of 0 A.

The corrected dataset contains 1776 samples from 222 accepted trajectories, split by
trajectory into 1232/272/272 train/validation/frozen-test samples. Across three seeds,
no controller passed the frozen-test NRMSE <1% gate. Mean test NRMSE was 1.257% for the
five-state pure DNN, 1.990% for the full-state pure DNN, and 3.092% for the full-state
feasible-interval DNN.

At nominal 25 C, mean reduced-model closed-loop NRMSE was 2.624%, 2.625%, and 2.536%,
respectively. Representative DFN NRMSE was 3.361%, 3.032%, and 3.339%. Charge-time gaps
were 6%--8%. All methods exceeded the 100x inference-speed gate. The feasible-interval
controller removed serious current/slew violations but did not repair imitation error.

Decision: Phase 6R failed its frozen acceptance contract. Stop the unprotected pure-DNN
replacement route, do not run Phase 6D, and do not run 15/30 C, Phase 5A perturbations,
or cross-battery experiments as part of Phase 6R. The next planned stage is Phase 5B-0,
which first establishes the nominal/oracle MPC feasibility envelope.
