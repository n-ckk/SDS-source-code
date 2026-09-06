# SDS-source-code

Forecasting the US female employment level (FRED series `LNS12000002`) for
BMMS2094, SDG 5 / SDG 8.

## Data

`LNS12000002.xlsx` is the raw FRED download: monthly, thousands of persons,
**seasonally adjusted at source by BLS**, 1948-01 to 2026-07 (943 observations).

That the series is already seasonally adjusted is the single most important fact
about it. It is why Holt-Winters fits `gamma` at essentially zero, why MASE here
is scaled by the lag-1 naive forecast rather than the seasonal one, and why
`SARIMA.R` — which by design *requires* a seasonal term — ends up reporting a
seasonal coefficient of 0.026 with p = 0.44. The seasonal structure is fitted
because the model specification calls for it, not because the data carries it,
and the script says so on every run.

One month, **2025-10**, is blank at source and is filled by linear interpolation
in `data_processing.R`. It falls inside the test window, so `data_processing.R`
writes an `imputed` column and `common.R::evaluate()` **excludes it from every
score — models and benchmarks alike**. Every test figure in this project is
therefore on 11 months, not 12.

## Running

```
Rscript run_all.R
```

Runs the whole pipeline in order and writes `session_info.txt`. `SARIMA.R` fits
a 225-specification grid (about 203 converge) and accounts for most of the
runtime. That search is cached in `sarima_model_selection.csv`; the cache carries
a fingerprint of the data and the split and is rebuilt automatically if either
changes. Delete it, or set `SARIMA_REFRESH=TRUE`, to force a fresh one.

To run one model on its own, run `data_processing.R` first — it writes the CSV
that everything else reads. `rolling_cv.R` must run **after** the four model
scripts, because it reads each model's selected specification from its
`*_result.csv` rather than keeping its own copy.

Requires R with: `forecast`, `rugarch`, `tseries`, `zoo`, `dplyr`, `ggplot2`,
`readxl`, `FinTS`. No script installs packages for you.

## Files

| File | What it does |
|---|---|
| `data_processing.R` | Reads the workbook, completes the monthly grid, interpolates gaps, flags them, writes `processed_female_employment.csv` |
| `common.R` | **Shared foundation** — the split, the metrics, the MASE denominators, the benchmarks, the analysis windows, the COVID regressors |
| `auto_arima.R` | `auto.arima` order selection |
| `SARIMA.R` | Seasonal ARIMA grid search with a **compulsory** seasonal term, ranked on validation RMSE |
| `HoltWinter.R` | Holt-Winters additive, 2021+ window, plus a test of whether its seasonal terms earn their place |
| `ARX-GARCH.R` | ARX with intervention dummies, fat tails and an optional GARCH(1,1) variance |
| `compare_models.R` | Builds `model_comparison.csv` — the single-holdout table |
| `rolling_cv.R` | Re-scores the same specifications at 24 forecast origins, and builds the calibrated intervals |
| `run_all.R` | Runs all of the above in order |

Generated CSVs and PNGs are gitignored; regenerate them with `run_all.R`.
`processed_female_employment.csv` is the one exception — it is committed so a
model script can be run without the workbook.

## Rules that make the comparison valid

These are enforced in `common.R`. Breaking any of them silently invalidates the
comparison table, which is exactly what happened in earlier versions of this
repository.

**One split.** Train ends 2024-07, validation is 2024-08 to 2025-07, test is
2025-08 to 2026-07. A model may start its analysis window later than 1948 —
Holt-Winters uses 2021+, ARX-GARCH uses 2010+ — but it may not move those
boundaries. Those window constants live in `common.R`, not in the model scripts,
because `rolling_cv.R` needs the same ones. Months published after the test
window closes land in `parts$post` and are not scored.

**One MASE denominator, and it is the lag-1 one.** `forecast::accuracy()` scales
MASE by each model's own training window, so models fitted on different windows
produce MASE values that cannot be compared — use `common.R::evaluate()`, never
`accuracy()`. The scaling is `mean|y_t − y_{t−1}|` = 186.8, Hyndman's
non-seasonal convention, which is the right one for a series BLS has already
deseasonalised. `MASE_s`, on the seasonal-naive denominator of 1041.5, is
reported in the same tables but **must never be quoted alone**: on a strongly
trending series a 12-month difference is mostly drift, so that denominator is
5.6× larger and flatters every model by the same factor. The difference is not
cosmetic — it is the difference between "worse than a naive forecast" (MASE 1.22)
and "4.5× better than naive" (MASE_s 0.22).

**One benchmark.** `benchmark_forecasts()` in `common.R` defines naive, seasonal
naive and RW-with-drift once, with the drift estimated over the full history to
the test start. Every script uses it. Two different definitions used to coexist,
and the same claim came out as −8.2% in one file and −10.6% in another.

**AIC and BIC are not comparable across models here.** Different training
windows, and ARX-GARCH is fitted on differences rather than levels (and
`rugarch` divides its information criteria by *n*). They are deliberately absent
from `model_comparison.csv`.

**Ljung-Box p-values are not comparable across models either.** The rows test
different residuals at different lags on different sample sizes — the ARX-GARCH
row is on standardised residuals of the *differenced* series. The table prints
the lag and the df beside each p-value so this is visible. Read each row as a
diagnostic of its own model, never as a ranking.

**Identifiability is reported on every model, and SARIMA fails it.** An
over-parameterised ARIMA can converge with a singular Hessian and return NaN
standard errors; its coefficients, information criteria and prediction intervals
are then all unusable, while its point forecasts still compute normally.

`SARIMA.R` ranks on validation RMSE and does not gate on this, by design — that
is the selection rule the model owner chose. The consequence is that it selects
SARIMA(4,1,3)(1,0,0)[12], which refits on train+validation with **five of its
nine standard errors NaN**, and which has the best test RMSE in the group. Both
things are true at once, and that is the point: good holdout error does not make
a degenerate model sound. So for this model, **quote the point forecasts and the
accuracy metrics; do not quote its intervals, its AIC/BIC, or its coefficient
significance.** `SARIMA.R` section 6 prints this on every run, `sarima_result.csv`
carries `Identifiable = FALSE`, and `compare_models.R` raises it as a warning.

`ARX-GARCH.R` carries the equivalent check for its own failure mode, a variance
process on the IGARCH boundary, which finite standard errors do not detect.

**A residual diagnostic must have had power before its PASS counts.** The
ARX-GARCH search qualified specifications on a Ljung-Box test of squared
residuals. That test is destroyed by a single dominant outlier: the nine
specifications with a 8-to-40-sigma April 2020 residual "passed" at p ≈ 0.99999
while the nine well-behaved ones were correctly flagged and rejected — the rule
was admitting the worst fits and rejecting the best. `z2_share()` now gates on
whether one residual dominates the sum of squares.

**No test/train error ratio.** Dividing a 12-step-ahead test error by 1-step-ahead
in-sample residuals compares two different quantities, and on a window containing
an untreated structural break it can "pass" simply because the training residuals
are inflated. Compare validation against test instead — both are 12-step-ahead
forecasts of unseen blocks.

**COVID is a measured trade-off, not an oversight.** Fitted through 2020 without
intervention dummies, ARIMA on the full history leaves a ~26-sigma residual at
2020-04 and kurtosis in the hundreds, so its Gaussian prediction intervals are not
trustworthy. Adding additive-outlier dummies fixes that but makes the point
forecasts worse. `auto_arima.R` and `SARIMA.R` report the accurate undummied
model, state that its intervals are unreliable, and fit the dummied variant as a
**controlled** sensitivity — same orders, only the dummies change — on every run,
so the numbers cannot go stale. `HoltWinter.R` sidesteps the break with a 2021+
window, since it cannot carry a regressor at all.

**What the validation block actually does.** In `auto_arima.R` it is a genuine
selection input. In `SARIMA.R` and `ARX-GARCH.R` it is **not** — those scripts
qualify and rank on in-sample criteria (Ljung-Box, identifiability, AICc/BIC), so
in substance they use a two-way split with an in-sample selection rule. The
validation block's role there is the printed cross-check and the out-of-sample
`Val_RMSE` column. Earlier headers claimed it did selection work; it did not.

## Reading the results

Two evaluations. Read both, and read the caveats on each.

`model_comparison.csv` is the strict single-holdout result: one 12-month test
block read once, scored on the 11 observed months, with benchmarks in
`model_comparison_benchmarks.csv`. It is methodologically clean but **eleven
points cannot rank models that finish within a few percent of each other**.

`rolling_cv_summary.csv` re-scores the same fixed specifications at 24 forecast
origins. It has far more evidence behind it, but it is **not** clean:

- **Specification leakage.** Every specification was chosen using data through
  2025-07, while the origins start in 2023-08. That hindsight is not shared
  equally — ARIMA(2,1,2)+drift was picked as the best of 225 candidates, the ARX
  spec as the best of 36, and RW-with-drift was chosen by nobody. The model with
  the most tuning is the one that wins, so treat its margin as an upper bound.
- **The test block is re-read.** Twelve of the 24 origins forecast into 2025-08
  or later. Nothing is *selected* on it, so the model choice is uncontaminated,
  but this file and `model_comparison.csv` are not independent evidence about the
  same months.

Where the two disagree, the rolling result has more data behind it and the
single-block result has cleaner provenance. Report both and say which you are
quoting.

They do disagree, and it matters in two places.

**The benchmark.** On the single block, ARX-GARCH and Holt-Winters both beat the
random-walk-with-drift benchmark. Across 24 origins both are **worse** than it,
and only the two ARIMA-family models beat it consistently. The single block was
flattering them.

**SARIMA's lead.** On the single block SARIMA(4,1,3)(1,0,0)[12] has the best test
RMSE in the group, 3.7% ahead of ARIMA(2,1,2) with drift. Across 24 origins the
two are indistinguishable — 406.1 against 405.6 mean RMSE, a 0.1% gap, with
SARIMA the *more* variable of the pair (SD 95.7 against 90.5). The eleven-point
lead is not evidence of a better model. Quote it if you report the single block,
but do not build a conclusion on it.

`auto_arima.R` and `SARIMA.R` search overlapping spaces and differ in two ways at
once: SARIMA requires a seasonal term, and it ranks on validation RMSE where
`auto.arima` ranks on an information criterion. Either difference alone would be
enough to make them disagree, so their results are not independent confirmations
of each other. `SARIMA.R` section 7 prints the best non-seasonal candidate under
*each* ranking rule, so the two effects can be read apart: on validation RMSE it
is SARIMA(4,1,4)(0,0,0)[12], on AICc it is SARIMA(2,1,2)(0,0,0)[12] — the model
`auto_arima.R` reports.

## Which numbers to quote

**Point forecasts** — from the primary rows of `model_comparison.csv`. Note that
ARX-GARCH normally selects `ar = 0`, which makes its point forecast a straight
line: it *is* a random walk with drift, with the drift estimated by maximum
likelihood under Student-t errors instead of by the endpoint difference. Its
margin over the RW benchmark is an estimator difference, not a modelling one.

**Prediction intervals** — from `empirical_error_quantiles.csv`, not from any
model's own. Every model's intervals come out too wide, for different reasons:
the ARIMA family's sigma is inflated by the April 2020 outlier and its normal
quantile is wrong given the residual kurtosis; SARIMA's are worse still, because
its singular Hessian means they have no valid basis at all and not merely a
miscalibrated one; ARX-GARCH's variance process sits
on the IGARCH boundary with a Student-t shape near 3, so its simulated paths fan
out too fast (100% coverage at both the 80% and 95% level on the test block).
`rolling_cv.R` validates the empirical quantiles out of sample and reports both
in `rolling_cv_interval_calibration.csv`.

Interval at horizon *h* = point forecast + [q10, q90] for 80%, + [q025, q975] for
95%.
