# Estimator policy (measurement channel)

We treat robustness as **measurement-channel dependent**.

## Primary channel (default)
- **InfoNCE** is the primary robustness channel in this repo.
- Robustness PASS criteria are evaluated under InfoNCE (e.g., std(C) < 0.03 across declared axes).

## Secondary diagnostics
- **KSG**: high-variance in some regimes (sample/geometry sensitive). Use as secondary diagnostic only.
- **MINE**: may sanity-fail (non-finite values) under some configurations. Treat as unstable unless stabilized with additional controls.

## Sanity / failure handling
- Non-finite ratios (NaN/±inf) are flagged as estimator failures.
- Failures are not silently ignored; they must be:
  1) counted,
  2) listed by run_id,
  3) excluded under a declared rule.

This policy is consistent with making the paper’s robustness claim explicit:
> “Robust under the InfoNCE measurement channel across declared axes.”
