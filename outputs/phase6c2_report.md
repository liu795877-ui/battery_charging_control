# Phase 6C-2: targeted MPC teacher data

## Frozen-set protection

- Phase 6B dataset SHA-256: `640b45c017594c3a8f954b498a99058d2e5cc8ecf6cff88425d5df94815de5da`
- Frozen test trajectories: 88
- The original validation and test assignments were not modified.

## New teacher data

- Attempted trajectories: 800
- Accepted trajectories: 800
- Acceptance fraction: 100.00%
- Unfolded samples: 6400
- New split samples: {'phase6c_train': 5440, 'phase6c_validation': 960}
- Active constraints: {'voltage': 919, 'temperature': 1261, 'current_upper': 0, 'current_change': 999}

New samples are labeled by source as `targeted_boundary_sampling` or `closed_loop_DAgger` and enter only the Phase 6C training/new-validation sets.
