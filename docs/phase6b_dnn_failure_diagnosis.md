# Phase 6B DNN failure diagnosis

Phase 6B answers a narrower question than Phase 6A:

> Why did the pure paper-style DNN fail to learn the MPC teacher well enough?

It keeps the paper-method validation boundary: MPC generates labels, a DNN
imitates the explicit control law, and the pure DNN is evaluated without a
safety filter. The added projected DNN is only a control group.

## Error partitions

The offline test errors are grouped by:

- SOC interval
- average-temperature interval
- previous-current interval
- voltage, temperature, current-upper, and current-slew active flags
- near-current-slew-boundary flag

The current-slew boundary is the priority diagnosis because Phase 6A failed most
clearly on the one-step current-change constraint.

## Larger model and larger data

Phase 6B raises the default data scale to 1000 initial states and trains larger
paper-style three-hidden-layer networks:

- 5-32-32-16-1
- 5-64-64-32-1

This tests whether the Phase 6A failure is mainly a capacity/data issue before
adding any runtime safety filter.

The first Phase 6B implementation deliberately uses one seed and one
regularization value. A full grid over both large networks was too expensive for
an interactive diagnostic run, and the immediate question is whether larger
capacity and more data move the error enough to change the conclusion.

## Projected DNN control group

The projected DNN applies this post-processing after raw network inference:

1. clip current to `[0, I_max]`
2. clip current to `[I_previous - Delta I_max, I_previous + Delta I_max]`

Interpretation:

- If projection removes violations but NRMSE remains high, the DNN has not
  learned the MPC map accurately.
- If projection also lowers NRMSE materially, part of the Phase 6A error came
  from raw outputs that ignored basic controller bounds.
