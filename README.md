# SDS-source-code

Forecasting the US female employment level (FRED series `LNS12000002`) for
BMMS2094, SDG 5 / SDG 8.

## Data

`LNS12000002.xlsx` is the raw FRED download: monthly, thousands of persons,
**seasonally adjusted at source by BLS**, 1948-01 to 2026-07 (943 observations).

That the series is already seasonally adjusted is the single most important fact
about it. It is why `SARIMA.R` exists as a *test* for residual seasonality rather
than as an assumed-seasonal model, and why Holt-Winters fits `gamma` at
essentially zero.

One month, **2025-10**, is blank at source and is filled by linear interpolation
in `data_processing.R`. It falls inside the test window, so every model is scored
against one synthetic actual. `data_processing.R` writes an `imputed` column so
this is visible in every forecast table instead of being silently forgotten.

## Running

```
Rscript run_all.R
```

Runs the whole pipeline in order and writes `session_info.txt`. Takes about six
minutes end to end; `SARIMA.R` fits 225 candidate models and accounts for most
of it. That search is cached in `sarima_model_selection.csv` - delete it, or set
`SARIMA_REFRESH=TRUE`, to force a fresh one.

To run one model on its own, run `data_processing.R` first — it writes the CSV
that everything else reads.

Requires R with: `forecast`, `rugarch`, `tseries`, `zoo`, `dplyr`, `ggplot2`,
`readxl`, `FinTS`. No script installs packages for you.

## Files

| File | What it does |
|---|---|
| `data_processing.R` | Reads the workbook, interpolates gaps, flags them, writes `processed_female_employment.csv` |
| `common.R` | **Shared foundation** — the split, the metrics, the MASE denominator, the COVID regressors |
| `auto_arima.R` | `auto.arima` order selection |
| `SARIMA.R` | Seasonal ARIMA grid search — also the test for residual seasonality |
| `HoltWinter.R` | Holt-Winters additive, 2021+ window |
| `ARX-GARCH.R` | ARX with intervention dummies, fat tails and an optional GARCH(1,1) variance |
| `compare_models.R` | Builds `model_comparison.csv` — the deliverable table |
| `run_all.R` | Runs all of the above in order |

Generated CSVs and PNGs are gitignored; regenerate them with `run_all.R`.

## Rules that make the comparison valid

These are enforced in `common.R`. Breaking any of them silently invalidates the
comparison table, which is exactly what happened in an earlier version of this
repository.

**One split.** Train ends 2024-07, validation is 2024-08 to 2025-07, test is
2025-08 to 2026-07. A model may start its analysis window later than 1948 —
Holt-Winters uses 2021+, ARX-GARCH uses 2010+ — but it may not move those two
boundaries. Candidates are fitted on train, qualified on validation, refitted on
train+validation, and the test block is read exactly once.

**One MASE denominator.** `forecast::accuracy()` scales MASE by each model's own
training window, so models fitted on different windows produce MASE values that
cannot be compared. On the scripts' own numbers Holt-Winters appeared to beat
`auto.arima`; on a single denominator the ranking reverses. Use
`common.R::evaluate()`, never `accuracy()`, for anything that goes in the
comparison table.

**AIC and BIC are not comparable across models here.** Different training
windows, and ARX-GARCH is fitted on differences rather than levels (and
`rugarch` divides its information criteria by *n*). They are deliberately absent
from `model_comparison.csv`.

**Identifiability is a gate.** An over-parameterised ARIMA can converge with a
singular Hessian and return NaN standard errors; its coefficients, information
criteria and prediction intervals are then all unusable. An earlier version
selected such a model — and it had the best test RMSE in the group. Good holdout
error does not make a degenerate model sound.

**No test/train error ratio.** Dividing a 12-step-ahead test error by 1-step-ahead
in-sample residuals compares two different quantities, and on a window containing
an untreated structural break it can "pass" simply because the training residuals
are inflated. Compare validation against test instead — both are 12-step-ahead
forecasts of unseen blocks.

**COVID is a measured trade-off, not an oversight.** Fitted through 2020 without
intervention dummies, ARIMA on the full history leaves a 26-sigma residual at
2020-04 and kurtosis of 504, so its Gaussian prediction intervals are not
trustworthy. Adding additive-outlier dummies fixes that - max|z| drops to about
7, kurtosis to 10 - but makes the point forecasts worse in every one of the eight
configurations tested (four intervention windows, seasonal on and off), with the
best dummied variant still worse than a random walk with drift.

So `auto_arima.R` and `SARIMA.R` report the accurate undummied model and state
that their intervals are unreliable; each fits the dummied variant as a labelled
sensitivity so the trade-off is visible in the output. `ARX-GARCH.R` gets both
right, because in *difference* space the COVID event genuinely is four large
spikes that four pulses can represent - in level space the same four pulses cover
the crash but not the two-year recovery. `HoltWinter.R` sidesteps the break with
a 2021+ window, since it cannot carry a regressor at all.

Quote point forecasts from the primary rows; quote intervals only from
ARX-GARCH.

## Reading the results

`model_comparison.csv` ranks models by test RMSE on the identical 12-month test
window, with naive, seasonal-naive and random-walk-with-drift benchmarks in
`model_comparison_benchmarks.csv`. Quote the margin over random-walk-with-drift:
it is the honest measure of what the modelling bought.
