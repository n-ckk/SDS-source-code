# ==============================================================================
# BMMS2094 - SDG 5 / SDG 8: US Female Employment Level
# MODEL: ARX(p)-GARCH(1,1) with fat-tailed errors and COVID intervention dummies.
#        The AR order p and the error distribution are NOT hardcoded - they are
#        chosen by the rule stated in section 9.
#
# Why this model. The series is I(1) with no seasonality (STL F_S = 0.014), so the
# mean equation is an AR model on the FIRST DIFFERENCES with a constant, which is a
# drift in levels. The irregular component is the problem: a plain ARIMA leaves a
# 12.3-sigma residual in April 2020 and residual kurtosis of 121. Three additions
# deal with that, in descending order of how much they actually contribute on this
# training block (see the specification table in section 8 for the evidence):
#   (1) COVID pulse dummies   - takes the four shock months out of the mean
#                               equation. All specifications without them fail the
#                               squared-residual diagnostic; all with them satisfy
#                               it. This appears to account for most of the gain.
#   (2) Fat-tailed errors     - lets the likelihood accept fat tails instead of
#                               inflating sigma to cover them. Worth about 0.06 of
#                               BIC over normal errors at each AR order.
#   (3) GARCH(1,1) variance   - lets sigma_t adapt, which downweights the high-
#                               variance months when estimating the drift
#
# Qualification on the conditional variance. The ARCH-LM test applied to plain
# ARIMA residuals does not reject homoskedasticity (section 7). The GARCH component
# is therefore adopted as a robust estimator that downweights outlying months, and
# not on the evidence of a detected ARCH effect.
#
# ------------------------------------------------------------------------------
# THREE-WAY SPLIT - read this before changing any date below.
#
# The specification search in section 8 tries 18 models. If it is scored on the
# test set, the test set has chosen the model, and the test RMSE reported at the
# end is the best of 18 tries rather than an out-of-sample number. So:
#
#   TRAIN      2010-01 .. 2024-07  (175 months)  fit the 18 candidates
#   VALIDATION 2024-08 .. 2025-07  ( 12 months)  score them, pick one
#   TEST       2025-08 .. 2026-07  ( 12 months)  read ONCE, in section 12
#
# The chosen specification is then refitted on TRAIN + VALIDATION (187 months)
# before it forecasts the test window. Both holdouts are 12 months because the
# reported forecast horizon is 12 months - grading and reporting at the same
# horizon. The test window spans the Jan 2026 peak and the decline after it, so
# it is a turning-point test, not a purely trending one. This is a deliberate
# fixed-horizon holdout rather than a proportional 80/20 split.
# ==============================================================================

# 1. Packages
rm(list = ls())
for (p in c("rugarch", "tidyverse", "lubridate", "forecast", "tseries", "zoo", "FinTS")) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
}
library(rugarch); library(tidyverse); library(lubridate)
library(forecast); library(tseries); library(zoo)
set.seed(123)

# 2. Import the cleaned data (written by data_processing.R)
# Accepts either the group's filename or the one used in my own repo. The two are
# the same series; they differ only in the interpolated value for 2025-10, which is
# blank at source (77072.85 here against 77074.00, a day-axis versus equal-month
# interpolation of the same two neighbours).
CANDIDATES <- c("processed_female_employment.csv",
                "female_emp_cleaned.csv",
                "data/processed/female_emp_cleaned.csv")
DATA_FILE <- CANDIDATES[file.exists(CANDIDATES)][1]
if (is.na(DATA_FILE)) {
  stop("Cleaned data not found. Looked for: ", paste(CANDIDATES, collapse = ", "),
       ". Run data_processing.R first.")
}
cat("Reading:", DATA_FILE, "\n")

raw_df <- read.csv(DATA_FILE, stringsAsFactors = FALSE)
names(raw_df)[1:2] <- c("date", "emp")
raw_df$date <- as.Date(raw_df$date)
raw_df$emp  <- as.numeric(raw_df$emp)

# 3. Analysis window
# Starts in 2010, not 1948. Female employment roughly quadrupled between 1948 and
# 2000 and has been flat-to-slowly-rising since; training on the full history
# estimates a drift from a regime that has ended.
ANALYSIS_START <- as.Date("2010-01-01")
VAL_START      <- as.Date("2024-08-01")
TEST_START     <- as.Date("2025-08-01")

emp_monthly <- raw_df %>%
  filter(date >= ANALYSIS_START) %>%
  select(date, emp) %>% arrange(date)
if (any(is.na(emp_monthly$emp))) emp_monthly$emp <- na.approx(emp_monthly$emp, na.rm = FALSE)
cat("Observations:", nrow(emp_monthly), "|",
    format(min(emp_monthly$date)), "to", format(max(emp_monthly$date)), "\n")

emp_ts <- ts(emp_monthly$emp,
             start = c(year(min(emp_monthly$date)), month(min(emp_monthly$date))),
             frequency = 12)

# 4. Confirm the order of integration before differencing
cat("\n--- Stationarity of the LEVEL ---\n")
print(adf.test(emp_ts)); print(kpss.test(emp_ts))
cat("\n--- Stationarity of the FIRST DIFFERENCE ---\n")
d_ts <- diff(emp_ts)
print(adf.test(d_ts)); print(kpss.test(d_ts))
cat("\nndiffs() suggests d =", ndiffs(emp_ts), "\n")

# 5. COVID intervention regressors (additive outliers on the four shock months).
# Built as a function of the date vector so the same rule generates the future
# regressor matrix required by ugarchforecast(). All four shock months fall inside
# the training block, so the validation and test regressor matrices are all zeros -
# that is correct, and rugarch still requires them to be supplied.
make_xreg <- function(dates) {
  shock <- as.Date(c("2020-03-01", "2020-04-01", "2020-05-01", "2020-06-01"))
  X <- sapply(shock, function(s) as.numeric(dates == s))
  colnames(X) <- paste0("AO", format(shock, "%y%m"))
  X
}

# 6. Three-way split
train_idx <- emp_monthly$date <  VAL_START
val_idx   <- emp_monthly$date >= VAL_START  & emp_monthly$date < TEST_START
test_idx  <- emp_monthly$date >= TEST_START
tv_idx    <- emp_monthly$date <  TEST_START     # train + validation

train_lvl <- emp_monthly$emp[train_idx];  train_dates <- emp_monthly$date[train_idx]
val_lvl   <- emp_monthly$emp[val_idx];    val_dates   <- emp_monthly$date[val_idx]
test_lvl  <- emp_monthly$emp[test_idx];   test_dates  <- emp_monthly$date[test_idx]
tv_lvl    <- emp_monthly$emp[tv_idx];     tv_dates    <- emp_monthly$date[tv_idx]

H_VAL  <- length(val_lvl)
H_TEST <- length(test_lvl)

stopifnot(length(train_lvl) + H_VAL + H_TEST == nrow(emp_monthly),
          length(tv_lvl) == length(train_lvl) + H_VAL)

# The model is fitted to differences, so the first date of each block drops out.
d_train       <- diff(train_lvl)
d_train_dates <- train_dates[-1]
last_lvl_tr   <- train_lvl[length(train_lvl)]

d_tv       <- diff(tv_lvl)
d_tv_dates <- tv_dates[-1]
last_lvl_tv <- tv_lvl[length(tv_lvl)]

X_train <- make_xreg(d_train_dates); X_val  <- make_xreg(val_dates)
X_tv    <- make_xreg(d_tv_dates);    X_test <- make_xreg(test_dates)

cat("\n=== SPLIT ===\n")
cat(sprintf("Train      : %s to %s  (%d months -> %d differences)\n",
            format(min(train_dates)), format(max(train_dates)),
            length(train_lvl), length(d_train)))
cat(sprintf("Validation : %s to %s  (%d months)  <- selection only\n",
            format(min(val_dates)), format(max(val_dates)), H_VAL))
cat(sprintf("Test       : %s to %s  (%d months)  <- scored once\n",
            format(min(test_dates)), format(max(test_dates)), H_TEST))
cat(sprintf("Train+Val  : %d months (refit set for the final model)\n", length(tv_lvl)))

# 7. Test for an ARCH effect before adopting a GARCH variance.
# This ARIMA is a diagnostic instrument rather than a competing model; it exists
# only to produce residuals for the ARCH-LM test.
cat("\n--- ARCH-LM test on ARIMA(1,1,1) residuals (H0: no ARCH) ---\n")
arima_ref <- Arima(ts(train_lvl, frequency = 12), order = c(1, 1, 1), include.drift = TRUE)
print(FinTS::ArchTest(residuals(arima_ref), lags = 12))
cat("A p-value above 0.05 provides no significant evidence of conditional\n",
    "heteroskedasticity, in which case the GARCH component is adopted as a robust\n",
    "estimator rather than on the evidence of a detected ARCH effect.\n")

# 8. Specification search over AR order, error distribution and the dummies.
# FITTED ON TRAIN, SCORED ON VALIDATION. BIC is in-sample parsimony; validation
# RMSE is out-of-sample. The test set is not touched anywhere in this section.
# NOTE: rugarch::infocriteria() returns AIC/BIC DIVIDED BY THE NUMBER OF OBSERVATIONS.
# They are comparable across the rows of this table but NOT against the AIC/BIC that
# Arima() or ets() report. Rescale by n before putting them in any shared table.
# Implementation note. rugarch constrains external-regressor coefficients to
# [-100, 100] by default. The COVID intervention coefficients are substantially
# larger in magnitude, the April 2020 effect being roughly -11,900 thousand persons,
# so the bounds are widened before estimation. Left at their defaults the estimates
# reach the boundary, the interventions absorb almost none of the shock, and the
# largest standardised residual rises to about 12.7 rather than 5. Section 11 checks
# that the fitted coefficients are interior to the widened bounds.
widen_bounds <- function(spec, k) {
  if (k > 0) {
    setbounds(spec) <- setNames(rep(list(c(-30000, 30000)), k), paste0("mxreg", seq_len(k)))
  }
  spec
}

build_spec <- function(ar_order, dist, X) {
  spec <- ugarchspec(
    variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
    mean.model     = list(armaOrder = c(ar_order, 0), include.mean = TRUE,
                          external.regressors = X),
    distribution.model = dist
  )
  widen_bounds(spec, if (is.null(X)) 0 else ncol(X))
}

fit_spec <- function(ar_order, dist, use_x) {
  spec <- build_spec(ar_order, dist, if (use_x) X_train else NULL)
  fit <- tryCatch(
    ugarchfit(spec, data = d_train, solver = "hybrid"),
    error = function(e) NULL
  )
  if (is.null(fit) || fit@fit$convergence != 0) return(NULL)

  fc <- ugarchforecast(fit, n.ahead = H_VAL,
                       external.forecasts = list(mregfor = if (use_x) X_val else NULL))
  lvl <- last_lvl_tr + cumsum(as.numeric(fitted(fc)))

  sr <- as.numeric(residuals(fit, standardize = TRUE))
  list(
    ar_order = ar_order, dist = dist, use_x = use_x,
    row = data.frame(
      Model    = sprintf("AR%s(%d)-GARCH(1,1) %s", ifelse(use_x, "X", ""), ar_order, dist),
      AIC      = infocriteria(fit)[1], BIC = infocriteria(fit)[2],
      Val_RMSE = sqrt(mean((val_lvl - lvl)^2)),
      Val_MAE  = mean(abs(val_lvl - lvl)),
      Val_MAPE = 100 * mean(abs((val_lvl - lvl) / val_lvl)),
      LB24_p   = Box.test(sr, lag = 24, type = "Ljung-Box", fitdf = ar_order)$p.value,
      LBsq_p   = Box.test(sr^2, lag = 12, type = "Ljung-Box")$p.value,
      MaxZ     = max(abs(sr)),
      stringsAsFactors = FALSE
    )
  )
}

grid <- expand.grid(ar = 0:2, dist = c("norm", "std", "sstd"), use_x = c(FALSE, TRUE),
                    stringsAsFactors = FALSE)
cand <- list(); tab <- data.frame()
for (i in seq_len(nrow(grid))) {
  r <- fit_spec(grid$ar[i], grid$dist[i], grid$use_x[i])
  if (!is.null(r)) { cand[[r$row$Model]] <- r; tab <- rbind(tab, r$row) }
}
cat("\n=== SPECIFICATION SEARCH (fitted on train, scored on validation) ===\n")
print(tab[order(tab$BIC), ], row.names = FALSE, digits = 5)

# 9. SELECTION RULE - stated up front and applied mechanically.
#
#   QUALIFY  a specification must leave no structure in the residuals:
#            Ljung-Box on standardised residuals   LB24_p  > 0.05
#            Ljung-Box on squared standardised res LBsq_p  > 0.05
#   RANK     among the qualifiers, lowest BIC wins.
#
# WHY BIC AND NOT VALIDATION RMSE FOR THE RANKING. The validation block is 12
# observations. The spread across the qualifying specs is 387 to 408 thousand, a
# range of about 5%, which on 12 points is noise - ranking on it would be picking
# the luckiest of 18, which is the failure mode this whole three-way split exists
# to prevent. BIC is computed on 174 observations and is far more stable. The
# validation set does real work here through the QUALIFY step and as the honest
# cross-check printed below; it just is not precise enough to break ties on.
# Report both columns and say which one you ranked on.
#
# The principal finding in the table above is that the intervention dummies, rather
# than the error distribution, appear to account for most of the improvement in the
# residuals:
#   with dummies   : LBsq_p 0.067 to 0.867, max|z| 3.5 to 5.8
#   without dummies: LBsq_p 0.0004 to 0.021, max|z| 6.2 to 10.8
# All specifications carrying the dummies satisfy the squared-residual diagnostic;
# none of those without them do. The normal-error specifications also satisfy it once
# the dummies are present, giving LBsq_p of 0.848, 0.867 and 0.364 at AR orders 0, 1
# and 2, so the Student-t distribution is not what rescues that test. It is preferred
# on BIC instead (14.107/14.126/14.155 against 14.163/14.193/14.241 for normal errors).
#
# max|z| is not comparable across distributions. Normal errors inflate sigma_t to
# cover the tails, which mechanically shrinks the standardised residuals, so the
# smaller max|z| of the normal specifications is not evidence of a better fit.
#
# "std" = standardised Student-t in rugarch's naming, "sstd" = skewed version.
LB_ALPHA <- 0.05

qualified <- tab[tab$LB24_p > LB_ALPHA & tab$LBsq_p > LB_ALPHA, ]
cat("\n=== QUALIFYING SPECIFICATIONS (LB24_p > 0.05 and LBsq_p > 0.05) ===\n")
if (nrow(qualified) == 0) {
  cat("None qualified. Falling back to lowest BIC over all specs - report this.\n")
  qualified <- tab
} else {
  print(qualified[order(qualified$BIC), ], row.names = FALSE, digits = 5)
}

BEST <- qualified$Model[which.min(qualified$BIC)]
best_cfg <- cand[[BEST]]

cat(sprintf("\nSelected by rule (min BIC among qualifiers): %s\n", BEST))
cat(sprintf("Cross-check - lowest validation RMSE among qualifiers would pick: %s\n",
            qualified$Model[which.min(qualified$Val_RMSE)]))
cat("If those two disagree, say so in the report rather than quoting only one.\n")

# 10. Refit the chosen specification on TRAIN + VALIDATION.
# The specification was picked using 175 months; there is no reason to discard the
# validation year once that choice is locked in. Everything below - diagnostics,
# forecast, intervals, accuracy - comes from this fit.
spec_final <- build_spec(best_cfg$ar_order, best_cfg$dist,
                         if (best_cfg$use_x) X_tv else NULL)
fit <- ugarchfit(spec_final, data = d_tv, solver = "hybrid")
cat("\n=== FINAL MODEL, REFITTED ON TRAIN + VALIDATION ===\n")
show(fit)

# 11. Residual diagnostics
# Guard against the rugarch bounds trap: if any mxreg coefficient sits on its
# boundary the dummies were never really estimated and every number below is wrong.
mx <- coef(fit)[grepl("^mxreg", names(coef(fit)))]
if (length(mx)) {
  cat("\nIntervention coefficients (should be close to the actual shock sizes",
      "-1768, -11861, 2015, 3128):\n")
  print(round(mx, 1))
  if (any(abs(abs(mx) - 30000) < 1) || any(abs(abs(mx) - 100) < 1e-6)) {
    warning("mxreg coefficient on its boundary - widen the bounds and refit.")
  }
}

sr <- as.numeric(residuals(fit, standardize = TRUE))
cat("\n=== STANDARDISED-RESIDUAL DIAGNOSTICS ===\n")
print(Box.test(sr,   lag = 12, type = "Ljung-Box", fitdf = best_cfg$ar_order))
print(Box.test(sr,   lag = 24, type = "Ljung-Box", fitdf = best_cfg$ar_order))
cat("Ljung-Box on SQUARED standardised residuals (leftover ARCH):\n")
print(Box.test(sr^2, lag = 12, type = "Ljung-Box"))
# Jarque-Bera is expected to reject here. The model specifies Student-t errors, so
# the standardised residuals should follow a t distribution with the estimated shape
# parameter rather than a normal, and the model carries no normality assumption for
# the rejection to contradict. The tails are assessed instead by max |z| and by
# whether the interval coverage reported below is close to nominal.
print(jarque.bera.test(sr))
cat(sprintf("max |standardised residual| = %.2f  (plain ARIMA on this window: 12.3)\n",
            max(abs(sr))))
cat(sprintf("skewness = %.2f   kurtosis = %.1f\n",
            mean((sr - mean(sr))^3) / sd(sr)^3,
            mean((sr - mean(sr))^4) / sd(sr)^4 - 3))

par(mfrow = c(2, 2))
plot(d_tv_dates, sr, type = "h", main = "Standardised residuals",
     xlab = "Year", ylab = "z"); abline(h = c(-3, 0, 3), lty = c(3, 1, 3))
Acf(sr,   main = "ACF - standardised residuals")
Acf(sr^2, main = "ACF - squared standardised residuals")
qqnorm(sr, main = "Normal Q-Q"); qqline(sr)
par(mfrow = c(1, 1))

plot(d_tv_dates, sigma(fit), type = "l", lwd = 2,
     main = "Conditional standard deviation sigma_t",
     xlab = "Year", ylab = "thousands of persons")

# 12. FINAL TEST EVALUATION.
# The test observations took no part in estimation or in model selection. This is the
# first point at which test outcomes are used, and they are used only to evaluate the
# specification already locked in above.
fc <- ugarchforecast(fit, n.ahead = H_TEST,
                     external.forecasts = list(mregfor = if (best_cfg$use_x) X_test else NULL))
point_lvl <- last_lvl_tv + cumsum(as.numeric(fitted(fc)))

# 13. Prediction intervals by forward simulation.
# The level forecast is a CUMULATIVE SUM of differences, so its variance is not the
# per-step GARCH variance. Simulating whole paths and cumulating each one is the
# correct way to propagate both the AR dynamics and the evolving sigma_t.
NSIM <- 20000
sim <- ugarchsim(
  fit,
  n.sim       = H_TEST,
  m.sim       = NSIM,
  startMethod = "sample",                       # continue from the end of the fitted data
  mexsimdata  = if (best_cfg$use_x) replicate(NSIM, X_test, simplify = FALSE) else NULL
)
sim_paths <- fitted(sim)                        # H_TEST x NSIM simulated differences
lvl_paths <- last_lvl_tv + apply(sim_paths, 2, cumsum)
qs <- apply(lvl_paths, 1, quantile, probs = c(0.025, 0.10, 0.90, 0.975))

forecast_tbl <- data.frame(
  Period   = format(as.yearmon(test_dates)),
  Forecast = point_lvl,
  Actual   = test_lvl,
  Error    = test_lvl - point_lvl,
  APE_pct  = 100 * abs(test_lvl - point_lvl) / test_lvl,
  Lo80 = qs[2, ], Hi80 = qs[3, ], Lo95 = qs[1, ], Hi95 = qs[4, ]
)
cat("\n=== FORECAST vs ACTUAL, TEST SET (thousands of persons) ===\n")
print(forecast_tbl, row.names = FALSE, digits = 6)

cov80 <- mean(test_lvl >= forecast_tbl$Lo80 & test_lvl <= forecast_tbl$Hi80)
cov95 <- mean(test_lvl >= forecast_tbl$Lo95 & test_lvl <= forecast_tbl$Hi95)
cat(sprintf("\n80 pct PI coverage = %.0f%%   95 pct PI coverage = %.0f%%\n",
            100 * cov80, 100 * cov95))

# 14. Accuracy, and the train/test gap checks
fit_diff  <- as.numeric(fitted(fit))
fit_lvl   <- head(tv_lvl, -1) + fit_diff
train_act <- tail(tv_lvl, -1)

# MASE SCALING - read this before copying a MASE into the group table.
# MASE divides the forecast MAE by the in-sample MAE of a naive forecast computed on
# the TRAINING data. The denominator therefore depends entirely on which training
# window a model used, and two models fitted on different windows produce MASE values
# that CANNOT be compared. A model trained from 1948 gets a much smaller denominator
# than one trained from 2010, because absolute month-to-month changes in the 1950s-70s
# were far smaller, and it will look better on MASE for that reason alone.
# Whoever assembles the group table must apply ONE denominator to every row. All three
# candidates are computed here so the choice is explicit rather than accidental.
#
#   snaive_tv   seasonal naive on this model's own window (2010+). This is what
#               forecast::accuracy() uses for a ts with frequency = 12, so it is the
#               correct default for this model considered on its own.
#   naive_tv    non-seasonal naive on the same window. Arguably the more meaningful
#               benchmark here, because the series is seasonally adjusted and has no
#               seasonality to exploit (STL F_S = 0.014), which makes a seasonal-naive
#               denominator an artificially weak comparator.
#   snaive_full seasonal naive over the FULL 1948+ history up to the test start.
#               Included only so this row can be matched to group members whose models
#               were fitted on the full series.
mase_scale <- c(
  snaive_tv   = mean(abs(diff(tv_lvl, lag = 12))),
  naive_tv    = mean(abs(diff(tv_lvl, lag =  1))),
  snaive_full = mean(abs(diff(raw_df$emp[raw_df$date < TEST_START], lag = 12)))
)
MASE_DEFAULT <- "snaive_tv"

acc <- function(a, f, scale = mase_scale[[MASE_DEFAULT]]) {
  c(RMSE = sqrt(mean((a - f)^2)),
    MAE  = mean(abs(a - f)),
    MAPE = 100 * mean(abs((a - f) / a)),
    MASE = mean(abs(a - f)) / scale)
}
tr_acc <- acc(train_act, fit_lvl)
te_acc <- acc(test_lvl,  point_lvl)

# Benchmark: random walk with drift, the minimum standard the model must improve on.
drift  <- (last_lvl_tv - tv_lvl[1]) / (length(tv_lvl) - 1)
rw_fc  <- last_lvl_tv + drift * seq_len(H_TEST)
rw_acc <- acc(test_lvl, rw_fc)

cat("\n=== ACCURACY ===\n")
print(rbind(`Train+Val (in-sample)` = tr_acc, Test = te_acc,
            `Benchmark RW-drift` = rw_acc), digits = 5)
cat(sprintf("\nMAPE gap  = %+.4f pp   (group rule: |gap| <= 1.3)  -> %s\n",
            te_acc["MAPE"] - tr_acc["MAPE"],
            ifelse(abs(te_acc["MAPE"] - tr_acc["MAPE"]) <= 1.3, "PASS", "FAIL")))
cat(sprintf("RMSE ratio = %.3f          (group rule: <= 1.3)      -> %s\n",
            te_acc["RMSE"] / tr_acc["RMSE"],
            ifelse(te_acc["RMSE"] / tr_acc["RMSE"] <= 1.3, "PASS", "FAIL")))
cat(sprintf("Test RMSE vs RW-drift benchmark: %+.1f%%  (negative = model is better)\n",
            100 * (te_acc["RMSE"] / rw_acc["RMSE"] - 1)))

cat("\n=== MASE UNDER EACH SCALING CONVENTION ===\n")
cat(sprintf("MASE reported above uses '%s' (denominator = %.2f).\n",
            MASE_DEFAULT, mase_scale[[MASE_DEFAULT]]))
mase_tbl <- data.frame(
  Scaling     = names(mase_scale),
  Denominator = as.numeric(mase_scale),
  Test_MASE   = as.numeric(te_acc["MAE"]) / as.numeric(mase_scale),
  Benchmark_MASE = as.numeric(rw_acc["MAE"]) / as.numeric(mase_scale),
  row.names = NULL)
print(mase_tbl, row.names = FALSE, digits = 5)
cat("Quote ONE of these rows in the group table and use the same one for every model.\n")
cat("A MASE below 1 means the model beats the naive forecast that defines the scale.\n")

# 15. Forecast plot
plot_df <- rbind(
  data.frame(date = emp_monthly$date, value = emp_monthly$emp, series = "Actual",
             lo = NA, hi = NA),
  data.frame(date = test_dates, value = point_lvl, series = "ARX-GARCH forecast",
             lo = forecast_tbl$Lo95, hi = forecast_tbl$Hi95)
)
print(
  ggplot(plot_df %>% filter(date >= as.Date("2021-01-01")),
         aes(date, value, colour = series)) +
    geom_ribbon(aes(ymin = lo, ymax = hi), fill = "steelblue", alpha = 0.18,
                colour = NA, na.rm = TRUE) +
    geom_line(linewidth = 0.8, na.rm = TRUE) +
    geom_vline(xintercept = VAL_START,  linetype = 3) +
    geom_vline(xintercept = TEST_START, linetype = 2) +
    labs(title = paste(BEST, "forecast vs actual, with 95% interval"),
         subtitle = "dotted = validation start, dashed = test start",
         x = "Year", y = "Employment (thousands of persons)", colour = NULL) +
    theme_minimal()
)

# 16. Refit on the full series and forecast 12 months beyond the data
d_all       <- diff(emp_monthly$emp)
d_all_dates <- emp_monthly$date[-1]
future_dates <- seq(max(emp_monthly$date) %m+% months(1), by = "month", length.out = 12)

spec_all  <- build_spec(best_cfg$ar_order, best_cfg$dist,
                        if (best_cfg$use_x) make_xreg(d_all_dates) else NULL)
fit_final <- ugarchfit(spec_all, data = d_all, solver = "hybrid")
fc_final  <- ugarchforecast(fit_final, n.ahead = 12,
                            external.forecasts = list(
                              mregfor = if (best_cfg$use_x) make_xreg(future_dates) else NULL))
final_lvl <- tail(emp_monthly$emp, 1) + cumsum(as.numeric(fitted(fc_final)))

cat("\n=== 12-MONTH AHEAD FORECAST (beyond the observed sample) ===\n")
print(data.frame(Period = format(as.yearmon(future_dates)), Forecast = final_lvl),
      row.names = FALSE, digits = 6)
cat("\nLimitation. The series peaked at 77,749 in Jan 2026 and has fallen since, by\n",
    "a mean of 91.7 thousand per month over the last six months. The drift is\n",
    "estimated over the whole window and remains positive, so this forecast continues\n",
    "to rise through a series that has recently turned.\n")

# 17. Save results
results_arx_garch <- data.frame(
  Model = BEST,
  AIC = infocriteria(fit)[1], BIC = infocriteria(fit)[2],
  Train_RMSE = tr_acc["RMSE"], Test_RMSE = te_acc["RMSE"],
  Train_MAE  = tr_acc["MAE"],  Test_MAE  = te_acc["MAE"],
  Train_MAPE = tr_acc["MAPE"], Test_MAPE = te_acc["MAPE"],
  Train_MASE = tr_acc["MASE"], Test_MASE = te_acc["MASE"],
  MASE_scaling = MASE_DEFAULT, MASE_denominator = mase_scale[[MASE_DEFAULT]],
  Test_MASE_snaive_tv   = as.numeric(te_acc["MAE"]) / mase_scale[["snaive_tv"]],
  Test_MASE_naive_tv    = as.numeric(te_acc["MAE"]) / mase_scale[["naive_tv"]],
  Test_MASE_snaive_full = as.numeric(te_acc["MAE"]) / mase_scale[["snaive_full"]],
  LjungBox24_p = Box.test(sr, lag = 24, type = "Ljung-Box",
                          fitdf = best_cfg$ar_order)$p.value,
  LjungBox_sq_p = Box.test(sr^2, lag = 12, type = "Ljung-Box")$p.value,
  Max_abs_z = max(abs(sr)),
  PI80_coverage = cov80, PI95_coverage = cov95,
  Benchmark_RMSE = rw_acc["RMSE"],
  row.names = NULL
)
# Full coefficient vector, so the report can quote the estimated parameters
# (mu, omega, alpha1, beta1, shape) rather than describing them in words.
coef_tbl <- data.frame(Parameter = names(coef(fit)),
                       Estimate  = as.numeric(coef(fit)),
                       StdError  = as.numeric(fit@fit$se.coef),
                       tvalue    = as.numeric(fit@fit$tval),
                       pvalue    = as.numeric(fit@fit$matcoef[, 4]),
                       row.names = NULL)
cat("\n=== FITTED COEFFICIENTS (final model, train + validation) ===\n")
print(coef_tbl, row.names = FALSE, digits = 5)
write.csv(coef_tbl, "ARX_GARCH_coefficients.csv", row.names = FALSE)

write.csv(results_arx_garch, "ARX_GARCH_results.csv", row.names = FALSE)
write.csv(forecast_tbl,      "ARX_GARCH_forecast.csv", row.names = FALSE)
write.csv(tab,               "ARX_GARCH_search.csv",  row.names = FALSE)
cat("\nSaved ARX_GARCH_results.csv, ARX_GARCH_forecast.csv, ARX_GARCH_search.csv\n",
    "and ARX_GARCH_coefficients.csv\n")
