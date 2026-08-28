# Supplementary — Bootstrap Coverage Simulation for the p99 Statistic

> **Anonymized supplementary material.** Referenced from Section 3.5
> ("Robust statistical protocol") of the paper, which states that the
> 99th-percentile statistic is reported for every cell but treated as
> *indicative only*. This file gives the coverage simulation and the
> exact figures behind that decision. It contains no author-identifying
> information. It sits alongside `preregistration_anonymized.md` in the
> same supplementary bundle.

## Why p99 is treated as indicative only

A cell's `p99` is determined by roughly its top 1% of backward-error
values (10 of 1000 trials, 50 of 5000). A percentile bootstrap resamples
with replacement from the observed sample, so it can reweight those
extreme values but can never produce a value more extreme than the
largest already drawn. The upper tail of a `p99` bootstrap CI is
therefore mechanically truncated, and its coverage falls below nominal.
This is a measured property of these data, not a generic caveat about
percentile bootstraps.

## Coverage simulation

- **Design:** 1000 simulated experiments per cell.
- **Operand model:** t-Student(nu = 2).
- **Trial budgets:** both budgets used in the main sweep (1000 and 5000
  trials per cell).
- **Nominal coverage:** 95%.

## Results

| Statistic | 1000 trials | 5000 trials |
|-----------|-------------|-------------|
| Median coverage | 95.5% | 94.8% |
| p99 coverage    | 92.7% | 94.0% |

Median coverage is close to nominal at both budgets. The `p99` coverage
is below the median's at both budgets and about 3.5 standard errors
less than nominal at 1000 trials. The gap is real and in the direction
the truncation mechanism forecasts, but modest, not a collapse. More
trials help `p99` measurably without closing the gap — hence the
indicative-only status at every trial count used in the paper.
