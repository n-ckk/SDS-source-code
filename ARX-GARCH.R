# ==============================================================================
# BMMS2094 - SDG 5 / SDG 8: US Female Employment Level
# MODEL: ARX(p) with fat-tailed errors, COVID intervention dummies and an
#        optional GARCH(1,1) variance. The AR order, the error distribution, the
#        dummies AND the variance model are all chosen by the rule in section 7;
#        none of them is hardcoded.
#
# Split, metrics, MASE denominators, benchmarks, the analysis window and the
# COVID regressors come from common.R. Do not redefine them here.
#
# WHY THIS MODEL. The series is I(1) with no seasonality, so the mean equation is
# an AR model on the FIRST DIFFERENCES with a constant, which is a drift in
# levels. The irregular component is the problem: a plain ARIMA on this window
# leaves a double-digit-sigma residual in April 2020. Two additions deal with it:
#   (1) COVID pulse dummies - take the four shock months out of the mean
#       equation. This accounts for most of the gain.
#   (2) Fat-tailed errors   - let the likelihood accept fat tails instead of
#       inflating sigma to cover them.
#
# WHAT THE SELECTED MEAN EQUATION USUALLY IS - state this plainly in the report.
# The rule below normally selects ar_order = 0, which means the mean equation has
# no AR terms at all: the forecast of each difference is the constant mu, so the
# LEVEL forecast is a perfectly straight line with slope mu. That is a random
# walk with drift. This model's point forecasts are therefore not a different
# model from the RW-with-drift benchmark - they are the same model with the drift
# estimated by maximum likelihood under Student-t errors with the COVID months
# removed, instead of by the endpoint difference. Any margin over that benchmark
# is an ESTIMATOR difference, not a modelling one. Saying "ARX-GARCH beats the
# random walk" without that sentence overstates what was shown.
#
# THE GARCH COMPONENT IS NOT ASSUMED - IT IS TESTED FOR, and the test is reported
# honestly in section 7. Read the note there before claiming GARCH earns its
# place: the earlier version of this header claimed a 0.243 BIC margin in its
# favour that turned out to be a dummies effect rather than a variance effect.
#
# CALIBRATION CAVEAT. The fitted variance sits on the IGARCH boundary
# (alpha1 + beta1 ~ 0.999), omega is not significant, and the Student-t shape
# parameter comes out near 3, so the innovation has no finite kurtosis. The
# simulated prediction intervals are correspondingly too WIDE - they cover 100%
# of the test points at both the 80% and the 95% level, and rolling_cv.R measures
# 90% coverage where 80% is nominal across 24 origins. Do NOT present these
# intervals as well calibrated. The calibrated intervals in this project are the
# empirical error quantiles in empirical_error_quantiles.csv.
#
# SEASONALITY. STL seasonal strength is F_s = 0.072 on this 2010+ analysis window
# (0.012 on the full 1948+ history - quote the figure for the window you actually
# model), and nsdiffs() = 0. No seasonal terms are carried.
#
# THREE-WAY SPLIT. Defined once in common.R and shared with every other model:
#   TRAIN      2010-01 .. 2024-07   fit the candidates
#   VALIDATION 2024-08 .. 2025-07   cross-check only - see section 7
#   TEST       2025-08 .. 2026-07   read ONCE, in section 10
# ==============================================================================

# 1. Packages
# No install.packages() here: a script that silently installs into the user's
# library is not reproducible. No rm(list = ls()) either - it wipes the caller's
# environment when scripts are sourced in sequence by run_all.R.
required <- c("rugarch", "dplyr", "ggplot2", "forecast", "tseries", "zoo", "FinTS")
missing  <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
  stop("Missing packages: ", paste(missing, collapse = ", "),
       "\nInstall them with install.packages(c(\"",
       paste(missing, collapse = "\", \""), "\"))")
}

suppressMessages({
  library(rugarch)
  library(dplyr)
  library(ggplot2)
  library(forecast)
  library(tseries)
  library(zoo)
})

source("common.R")

set.seed(123)

MODEL_ID       <- "arx_garch"
ANALYSIS_START <- ARX_WINDOW_START   # from common.R; rolling_cv.R reads it too

# 2. Data and split
# Analysis starts in 2010, not 1948. Female employment roughly quadrupled between
# 1948 and 2000 and has been flat-to-slowly-rising since; training on the full
# history estimates a drift from a regime that has ended.
data  <- load_series()
parts <- split_series(data, ANALYSIS_START)
describe_split(parts, "ARX-GARCH (2010+ window)")

emp_ts <- as_monthly_ts(parts$all)

# 3. Confirm the order of integration before differencing
cat("\n--- Stationarity of the LEVEL ---\n")
print(adf.test(emp_ts)); print(kpss.test(emp_ts))
cat("\n--- Stationarity of the FIRST DIFFERENCE ---\n")
print(adf.test(diff(emp_ts))); print(kpss.test(diff(emp_ts)))
cat("\nndiffs() suggests d =", ndiffs(emp_ts),
    "| nsdiffs() suggests D =", nsdiffs(emp_ts), "\n")

# 4. Blocks. The model is fitted to differences, so the first date of each block
#    drops out of the differenced series.
train_lvl <- parts$train$value
val_lvl   <- parts$val$value
test_lvl  <- parts$test$value
tv_lvl    <- parts$train_val$value

d_train       <- diff(train_lvl)
d_train_dates <- parts$train$date[-1]
last_lvl_tr   <- train_lvl[length(train_lvl)]

d_tv        <- diff(tv_lvl)
d_tv_dates  <- parts$train_val$date[-1]
last_lvl_tv <- tv_lvl[length(tv_lvl)]

# Difference-space dummies: the shock months are pulses in the DIFFERENCED series
# here, which is what this mean equation needs. covid_dummies() applies the same
# rule to whatever date vector it is handed - see the note in common.R.
X_train <- covid_dummies(d_train_dates)
X_val   <- covid_dummies(parts$val$date)
X_tv    <- covid_dummies(d_tv_dates)
X_test  <- covid_dummies(parts$test$date)

# 5. Test for an ARCH effect BEFORE adopting a GARCH variance.
# This ARIMA is a diagnostic instrument, not a competing model; it exists only to
# produce residuals for the ARCH-LM test.
cat("\n--- ARCH-LM test on ARIMA(1,1,1) residuals (H0: no ARCH) ---\n")
arima_ref <- Arima(ts(train_lvl, frequency = FREQ), order = c(1, 1, 1),
                   include.drift = TRUE)
arch_test <- FinTS::ArchTest(residuals(arima_ref), lags = 12)
print(arch_test)
cat("p =", signif(arch_test$p.value, 4),
    "- above 0.05 means no significant evidence of conditional\n",
    "heteroskedasticity, so the constant-variance specifications in the search\n",
    "below are genuine contenders rather than straw men.\n")

# 6. Specification search: AR order x error distribution x dummies x variance model
# FITTED ON TRAIN, SCORED ON VALIDATION. The test set is not touched here.
#
# NOTE: rugarch::infocriteria() returns AIC/BIC DIVIDED BY THE NUMBER OF
# OBSERVATIONS. They are comparable across the rows of this table but NOT against
# the AIC/BIC that Arima() or ets() report. Rescale by n before putting them in
# any shared table.
#
# Implementation note. rugarch constrains external-regressor coefficients to
# [-100, 100] by default. The COVID coefficients are far larger - the April 2020
# effect is about -11,900 thousand persons - so the bounds are widened before
# estimation. Left at their defaults the estimates hit the boundary, the
# interventions absorb almost none of the shock, and the largest standardised
# residual rises to about 12.7 rather than 5. Section 9 checks that the fitted
# coefficients are interior to the widened bounds.
widen_bounds <- function(spec, k) {
  if (k > 0) {
    setbounds(spec) <- setNames(rep(list(c(-30000, 30000)), k),
                                paste0("mxreg", seq_len(k)))
  }
  spec
}

build_spec <- function(ar_order, dist, X, garch) {
  spec <- ugarchspec(
    variance.model = list(model = "sGARCH",
                          garchOrder = if (garch) c(1, 1) else c(0, 0)),
    mean.model     = list(armaOrder = c(ar_order, 0), include.mean = TRUE,
                          external.regressors = X),
    distribution.model = dist
  )
  widen_bounds(spec, if (is.null(X)) 0 else ncol(X))
}

fit_spec <- function(ar_order, dist, use_x, garch) {
  spec <- build_spec(ar_order, dist, if (use_x) X_train else NULL, garch)
  fit  <- tryCatch(ugarchfit(spec, data = d_train, solver = "hybrid"),
                   error = function(e) NULL)
  if (is.null(fit) || fit@fit$convergence != 0) return(NULL)

  fc <- ugarchforecast(fit, n.ahead = HORIZON,
                       external.forecasts = list(
                         mregfor = if (use_x) X_val else NULL))
  lvl <- last_lvl_tr + cumsum(as.numeric(fitted(fc)))
  val_acc <- evaluate(val_lvl, lvl)

  sr <- as.numeric(residuals(fit, standardize = TRUE))

  # The squared-residual Ljung-Box must deduct the GARCH parameters it is testing
  # around; with fitdf = 0 it was systematically lenient towards the GARCH specs
  # it was meant to be judging.
  garch_df <- if (garch) 2L else 0L

  list(
    ar_order = ar_order, dist = dist, use_x = use_x, garch = garch,
    val_lvl_fc = lvl,          # keep it: section 12 scores this without refitting
    row = data.frame(
      Model    = sprintf("AR%s(%d)-%s %s", ifelse(use_x, "X", ""), ar_order,
                         ifelse(garch, "GARCH(1,1)", "constvar"), dist),
      Variance = ifelse(garch, "GARCH(1,1)", "constant"),
      Dummies  = use_x,
      AIC      = infocriteria(fit)[1],
      BIC      = infocriteria(fit)[2],
      Val_RMSE = unname(val_acc["RMSE"]),
      Val_MAE  = unname(val_acc["MAE"]),
      Val_MAPE = unname(val_acc["MAPE"]),
      LB24_p   = Box.test(sr, lag = 24, type = "Ljung-Box",
                          fitdf = ar_order)$p.value,
      LBsq_p   = Box.test(sr^2, lag = 12, type = "Ljung-Box",
                          fitdf = garch_df)$p.value,
      MaxZ     = max(abs(sr)),
      Z2_share = z2_share(sr),
      stringsAsFactors = FALSE
    )
  )
}

grid <- expand.grid(ar = 0:2, dist = c("norm", "std", "sstd"),
                    use_x = c(FALSE, TRUE), garch = c(FALSE, TRUE),
                    stringsAsFactors = FALSE)

cand <- list(); tab <- data.frame()
for (i in seq_len(nrow(grid))) {
  r <- fit_spec(grid$ar[i], grid$dist[i], grid$use_x[i], grid$garch[i])
  if (!is.null(r)) { cand[[r$row$Model]] <- r; tab <- rbind(tab, r$row) }
}

cat("\n=== SPECIFICATION SEARCH (fitted on train, scored on validation) ===\n")
cat("Candidates fitted:", nrow(tab), "of", nrow(grid), "attempted\n\n")
print(tab[order(tab$BIC), ], row.names = FALSE, digits = 5)

write.csv(tab, "ARX_GARCH_search.csv", row.names = FALSE)

# 7. SELECTION RULE - stated up front and applied mechanically.
#
#   QUALIFY  a specification must leave no structure in the residuals AND the
#            diagnostic that checks this must actually have had power:
#              Ljung-Box on standardised residuals    LB24_p   > 0.05
#              Ljung-Box on squared standardised res  LBsq_p   > 0.05
#              no single residual dominates the fit   Z2_share < 0.25
#   RANK     among the qualifiers, lowest BIC wins.
#
# WHY THE THIRD CONDITION EXISTS - this is a correction to the earlier rule, and
# it matters. With only the two Ljung-Box conditions, the rule was INVERTED:
#
#   group                        BIC      LBsq_p     max|z|   admitted?
#   constant variance + dummies  14.075   ~0.003     ~5.5     REJECTED
#   constant variance, no dummy  14.350   ~0.99999   8 to 40  ADMITTED
#
# The nine specifications with well-behaved residuals were thrown out and the
# nine with a 40-sigma April 2020 residual were let through. The reason is that a
# Ljung-Box test on squared residuals is destroyed by a single dominant outlier:
# when one point carries most of the sum of squares, the autocorrelations of the
# squared series are driven to zero and the test returns p ~ 1 regardless of what
# the rest of the series does. Those specifications were not passing the ARCH
# diagnostic, they were disabling it. Z2_share measures exactly that - see the
# note on z2_share() in common.R.
#
# WHY BIC AND NOT VALIDATION RMSE. The validation block is 12 observations and
# the spread across qualifying specs is only a few percent, which on 12 points is
# noise. BIC is computed on 174 observations and is far more stable.
#
# WHAT THE VALIDATION BLOCK ACTUALLY DOES HERE. Nothing in the selection. An
# earlier version of this comment claimed it "does real work through the QUALIFY
# step"; it does not - all three qualification conditions are in-sample residual
# diagnostics, and the ranking is on in-sample BIC. The validation block's roles
# are the cross-check printed below and the out-of-sample Val_RMSE that reaches
# the comparison table.
#
# "std" = standardised Student-t in rugarch's naming, "sstd" = skewed version.
#
# max|z| is NOT comparable across distributions: normal errors inflate sigma_t to
# cover the tails, which mechanically shrinks the standardised residuals.
LB_ALPHA     <- 0.05
Z2_SHARE_MAX <- 0.25

# is.finite() guards against an NA p-value producing phantom all-NA rows in the
# subset, which would inflate nrow(qualified) and corrupt the group comparisons
# below. SARIMA.R has always had this guard; this file did not.
ok <- is.finite(tab$LB24_p) & is.finite(tab$LBsq_p) & is.finite(tab$Z2_share) &
  tab$LB24_p > LB_ALPHA & tab$LBsq_p > LB_ALPHA & tab$Z2_share < Z2_SHARE_MAX

qualified <- tab[ok, ]

cat("\n=== QUALIFYING SPECIFICATIONS ===\n")
cat(sprintf("LB24_p > %.2f  AND  LBsq_p > %.2f  AND  Z2_share < %.2f\n",
            LB_ALPHA, LB_ALPHA, Z2_SHARE_MAX))
cat(sprintf("Rejected for residual autocorrelation : %d\n",
            sum(tab$LB24_p <= LB_ALPHA, na.rm = TRUE)))
cat(sprintf("Rejected for leftover ARCH            : %d\n",
            sum(tab$LBsq_p <= LB_ALPHA, na.rm = TRUE)))
cat(sprintf("Rejected because one residual dominates: %d  <- these were previously ADMITTED\n",
            sum(tab$Z2_share >= Z2_SHARE_MAX, na.rm = TRUE)))

if (nrow(qualified) == 0) {
  cat("None qualified. Falling back to lowest BIC over all specs - report this.\n")
  qualified <- tab
} else {
  print(qualified[order(qualified$BIC), ], row.names = FALSE, digits = 5)
}

BEST     <- qualified$Model[which.min(qualified$BIC)]
best_cfg <- cand[[BEST]]

cat(sprintf("\nSelected by rule (min BIC among qualifiers): %s\n", BEST))
cat(sprintf("Cross-check - lowest validation RMSE among qualifiers would pick: %s\n",
            qualified$Model[which.min(qualified$Val_RMSE)]))
cat("If those two disagree, say so in the report rather than quoting only one.\n")

# Does the GARCH variance actually earn its place?
#
# THIS COMPARISON MUST HOLD THE DUMMIES FIXED. The earlier version compared the
# best qualifying GARCH spec against the best qualifying constant-variance spec
# and reported the gap as evidence for GARCH - but under the old rule those two
# rows differed in their DUMMIES as well as their variance model, so the margin
# was mostly a dummies effect. Compare like with like.
cat("\n=== DOES THE GARCH VARIANCE EARN ITS PLACE? ===\n")
for (dm in c(TRUE, FALSE)) {
  g <- tab$BIC[tab$Variance == "GARCH(1,1)" & tab$Dummies == dm]
  k <- tab$BIC[tab$Variance == "constant"   & tab$Dummies == dm]
  if (length(g) && length(k)) {
    cat(sprintf("  dummies = %-5s : best GARCH BIC %.5f vs best constant %.5f  -> %+.5f in favour of %s\n",
                dm, min(g), min(k), min(k) - min(g),
                ifelse(min(g) < min(k), "GARCH(1,1)", "constant variance")))
  }
}
cat("\nRead the row for the dummy setting the SELECTED model actually uses. On this\n")
cat("data the like-for-like margin is small and can favour constant variance even\n")
cat("though the selected model carries GARCH - the GARCH term is then a robust\n")
cat("variance estimator, not a detected effect. The ARCH-LM test in section 5\n")
cat("found no ARCH effect either. Say that plainly rather than claiming GARCH was\n")
cat("shown to be necessary.\n")

# 8. Refit the chosen specification on TRAIN + VALIDATION.
spec_final <- build_spec(best_cfg$ar_order, best_cfg$dist,
                         if (best_cfg$use_x) X_tv else NULL, best_cfg$garch)
fit <- ugarchfit(spec_final, data = d_tv, solver = "hybrid")
cat("\n=== FINAL MODEL, REFITTED ON TRAIN + VALIDATION ===\n")
show(fit)

# 9. Residual diagnostics
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

# GARCH HEALTH CHECK.
# common.R::is_identifiable() gates the ARIMA models on finite standard errors.
# This model passes that test and is still on a variance boundary, so it needs
# its own check: finite standard errors say nothing about a persistence of 0.999.
cf  <- coef(fit)
pv  <- setNames(as.numeric(fit@fit$matcoef[, 4]), rownames(fit@fit$matcoef))
persistence <- sum(cf[names(cf) %in% c("alpha1", "beta1")])
shape_par   <- if ("shape" %in% names(cf)) unname(cf["shape"]) else NA_real_
se_finite   <- all(is.finite(as.numeric(fit@fit$se.coef)))

cat("\n=== VARIANCE-MODEL HEALTH ===\n")
cat(sprintf("Finite standard errors      : %s\n", se_finite))
if (best_cfg$garch) {
  cat(sprintf("Persistence alpha1 + beta1  : %.5f%s\n", persistence,
              if (persistence > 0.99) "   <- IGARCH boundary" else ""))
  cat(sprintf("omega p-value               : %.3f%s\n",
              pv[["omega"]],
              if (pv[["omega"]] > 0.05) "   <- not significant" else ""))
}
if (is.finite(shape_par)) {
  cat(sprintf("Student-t shape             : %.2f%s\n", shape_par,
              if (shape_par <= 4) "   <- kurtosis does not exist" else ""))
}
garch_healthy <- se_finite && (!best_cfg$garch || persistence <= 0.99) &&
  (!is.finite(shape_par) || shape_par > 4)
if (!garch_healthy) {
  cat("\nVERDICT: the variance model is on or near a boundary. The point forecasts\n")
  cat("are unaffected - they come from the mean equation - but every interval this\n")
  cat("model produces inherits a variance process that barely mean-reverts, which\n")
  cat("is why they come out too wide. Report this next to any interval you quote,\n")
  cat("or quote the empirical quantiles from rolling_cv.R instead.\n")
}

sr <- as.numeric(residuals(fit, standardize = TRUE))
cat("\n=== STANDARDISED-RESIDUAL DIAGNOSTICS ===\n")
print(Box.test(sr, lag = 12, type = "Ljung-Box", fitdf = best_cfg$ar_order))
print(Box.test(sr, lag = 24, type = "Ljung-Box", fitdf = best_cfg$ar_order))
cat("Ljung-Box on SQUARED standardised residuals (leftover ARCH):\n")
print(Box.test(sr^2, lag = 12, type = "Ljung-Box",
               fitdf = if (best_cfg$garch) 2L else 0L))
# Jarque-Bera is expected to reject when Student-t errors are selected: the
# standardised residuals should then follow a t distribution with the estimated
# shape parameter, not a normal, and the model carries no normality assumption
# for the rejection to contradict.
print(jarque.bera.test(sr))
cat(sprintf("max |standardised residual| = %.2f   share of total z^2 = %.3f\n",
            max(abs(sr)), z2_share(sr)))
cat(sprintf("skewness = %.2f   kurtosis = %.1f\n",
            mean((sr - mean(sr))^3) / sd(sr)^3,
            mean((sr - mean(sr))^4) / sd(sr)^4 - 3))

png("arx_garch_residual_diagnostics.png", width = 1400, height = 1000, res = 150)
par(mfrow = c(2, 2))
plot(d_tv_dates, sr, type = "h", main = "Standardised residuals",
     xlab = "Year", ylab = "z"); abline(h = c(-3, 0, 3), lty = c(3, 1, 3))
Acf(sr,   main = "ACF - standardised residuals")
Acf(sr^2, main = "ACF - squared standardised residuals")
qqnorm(sr, main = "Normal Q-Q"); qqline(sr)
par(mfrow = c(1, 1))
dev.off()

# 10. FINAL TEST EVALUATION - the test block is read only here.
fc <- ugarchforecast(fit, n.ahead = HORIZON,
                     external.forecasts = list(
                       mregfor = if (best_cfg$use_x) X_test else NULL))
point_lvl <- last_lvl_tv + cumsum(as.numeric(fitted(fc)))

# 11. Prediction intervals by forward simulation.
# The level forecast is a CUMULATIVE SUM of differences, so its variance is not
# the per-step variance. Simulating whole paths and cumulating each one is the
# correct way to propagate both the AR dynamics and any evolving sigma_t.
NSIM <- 20000
simulate_levels <- function(fitted_model, anchor, Xf, n = NSIM) {
  sim <- ugarchsim(
    fitted_model, n.sim = HORIZON, m.sim = n,
    startMethod = "sample",
    mexsimdata  = if (!is.null(Xf)) replicate(n, Xf, simplify = FALSE) else NULL
  )
  anchor + apply(fitted(sim), 2, cumsum)
}

lvl_paths <- simulate_levels(fit, last_lvl_tv,
                             if (best_cfg$use_x) X_test else NULL)
qs <- apply(lvl_paths, 1, quantile, probs = c(0.025, 0.10, 0.90, 0.975))

forecast_tbl <- data.frame(
  Date     = parts$test$date,
  Forecast = point_lvl,
  Actual   = test_lvl,
  Error    = test_lvl - point_lvl,
  APE_pct  = 100 * abs(test_lvl - point_lvl) / test_lvl,
  Lo80 = qs[2, ], Hi80 = qs[3, ], Lo95 = qs[1, ], Hi95 = qs[4, ],
  Imputed  = parts$test$imputed
)
cat("\n=== FORECAST vs ACTUAL, TEST SET (thousands of persons) ===\n")
cat("Imputed = the actual is interpolated, not observed. It is excluded from\n")
cat("every score below, INCLUDING the coverage figures.\n")
print(forecast_tbl, row.names = FALSE, digits = 6)

write.csv(forecast_tbl, "ARX_GARCH_forecast.csv", row.names = FALSE)

# Coverage on the SAME points the accuracy metrics use. This used to be computed
# over all 12 test months while the accuracy excluded the interpolated one, and
# the resulting figure was written into the Notes column of the shared table.
scored <- !parts$test$imputed
cov80 <- mean((test_lvl >= forecast_tbl$Lo80 & test_lvl <= forecast_tbl$Hi80)[scored])
cov95 <- mean((test_lvl >= forecast_tbl$Lo95 & test_lvl <= forecast_tbl$Hi95)[scored])
cat(sprintf("\n80%% PI coverage = %.0f%%   95%% PI coverage = %.0f%%   (on %d scored months)\n",
            100 * cov80, 100 * cov95, sum(scored)))
cat("Nominal coverage on 11 points is about 9/11 and 10/11. Coverage of 100% at\n")
cat("BOTH levels means the intervals are too WIDE, not well calibrated - report\n")
cat("this as a limitation rather than as a success. rolling_cv.R measures the\n")
cat("same thing across 24 origins, which is far better evidence than 11 points.\n")

# 12. Accuracy
# MASE uses the shared denominators from common.R. The validation forecast was
# produced during the search by the train-only fit; reuse it rather than
# refitting.
val_metrics  <- evaluate(val_lvl, best_cfg$val_lvl_fc)
test_metrics <- evaluate(test_lvl, point_lvl, exclude = parts$test$imputed)

# Benchmarks come from common.R so that "the RW-with-drift benchmark" means the
# same thing here as it does in compare_models.R and rolling_cv.R. This script
# used to build its own, estimating the drift over its 2010+ window instead of
# the full history AND scoring it on 12 points against a model scored on 11 - so
# the same claim came out as -10.6% here and -8.2% in compare_models.R.
# benchmark_table() takes no arguments: it always reads the full-history split
# from common.R. Handing it this script's 2010+ `parts` would benchmark the model
# against a drift estimated over its own window - the very inconsistency that
# made one claim come out as -10.6% here and -8.2% in compare_models.R.
bench <- benchmark_table()

cat("\n=== ACCURACY (shared MASE denominators:", round(MASE_DENOM, 2),
    "lag-1,", round(MASE_DENOM_S, 2), "seasonal) ===\n")
print(rbind(Validation = val_metrics, Test = test_metrics), digits = 5)
cat("\nShared benchmarks, scored on the identical months:\n")
print(round(bench[, c("RMSE", "MAE", "MASE")], 3))
cat(sprintf("\nTest RMSE vs RW-drift benchmark: %+.1f%%  (negative = model is better)\n",
            100 * (test_metrics["RMSE"] / bench["RW with drift", "RMSE"] - 1)))
if (best_cfg$ar_order == 0) {
  cat("REMINDER: the selected mean equation has ar_order = 0, so these point\n")
  cat("forecasts ARE a random walk with drift. The margin above is the difference\n")
  cat("between two ways of estimating one drift, not between two models.\n")
}

# For reference only - the window-local drift this script used to call "the
# benchmark". Reported so the difference is visible instead of silent.
drift_local <- (last_lvl_tv - tv_lvl[1]) / (length(tv_lvl) - 1)
cat(sprintf("\n(For reference: drift estimated over this 2010+ window is %.2f/month;\n",
            drift_local))
cat(sprintf(" the shared benchmark uses %.2f/month from the full history, and the\n",
            (tail(SHARED_PARTS$train_val$value, 1) - SHARED_PARTS$train_val$value[1]) /
              (nrow(SHARED_PARTS$train_val) - 1)))
cat(sprintf(" ML estimate in this model's mean equation is %.2f/month.)\n",
            unname(cf["mu"])))

# NOTE: no test/train ratio is reported. Dividing a 12-step-ahead test error by
# 1-step-ahead in-sample residuals compares two different quantities.

save_model_result(
  model_id     = MODEL_ID,
  model_name   = BEST,
  window_start = ANALYSIS_START,
  n_train      = nrow(parts$train),
  n_train_val  = nrow(parts$train_val),
  val_metrics  = val_metrics,
  test_metrics = test_metrics,
  lb           = list(p  = Box.test(sr, lag = 24, type = "Ljung-Box",
                                    fitdf = best_cfg$ar_order)$p.value,
                      lag = 24, fitdf = best_cfg$ar_order,
                      df = 24 - best_cfg$ar_order,
                      on = "std. resid. of differences"),
  identifiable = se_finite,
  aic          = infocriteria(fit)[1] * length(d_tv),
  bic          = infocriteria(fit)[2] * length(d_tv),
  spec         = sprintf("kind=arx;ar=%d;dist=%s;garch=%s;dummies=%s",
                         best_cfg$ar_order, best_cfg$dist,
                         best_cfg$garch, best_cfg$use_x),
  notes        = sprintf("PI coverage 80/95 = %.0f%%/%.0f%% on %d months; ARCH-LM p = %.3f;%s%s",
                         100 * cov80, 100 * cov95, sum(scored), arch_test$p.value,
                         if (best_cfg$ar_order == 0)
                           " ar=0 so point forecasts = RW with drift;" else "",
                         if (!garch_healthy)
                           " variance on boundary - intervals too wide" else "")
)

# 13. Fitted coefficients
coef_tbl <- data.frame(Parameter = names(coef(fit)),
                       Estimate  = as.numeric(coef(fit)),
                       StdError  = as.numeric(fit@fit$se.coef),
                       tvalue    = as.numeric(fit@fit$tval),
                       pvalue    = as.numeric(fit@fit$matcoef[, 4]),
                       row.names = NULL)
cat("\n=== FITTED COEFFICIENTS (final model, train + validation) ===\n")
print(coef_tbl, row.names = FALSE, digits = 5)
write.csv(coef_tbl, "ARX_GARCH_coefficients.csv", row.names = FALSE)

# 14. Forecast plot
plot_df <- rbind(
  data.frame(date = parts$all$date, value = parts$all$value, series = "Actual",
             lo = NA, hi = NA),
  data.frame(date = parts$test$date, value = point_lvl,
             series = "ARX forecast", lo = forecast_tbl$Lo95, hi = forecast_tbl$Hi95)
)

arx_plot <- ggplot(plot_df %>% filter(date >= as.Date("2021-01-01")),
                   aes(date, value, colour = series)) +
  geom_ribbon(aes(ymin = lo, ymax = hi), fill = "steelblue", alpha = 0.18,
              colour = NA, na.rm = TRUE) +
  geom_line(linewidth = 0.8, na.rm = TRUE) +
  geom_vline(xintercept = VAL_START,  linetype = 3) +
  geom_vline(xintercept = TEST_START, linetype = 2) +
  labs(title = paste(BEST, "forecast vs actual, with 95% interval"),
       subtitle = "dotted = validation start, dashed = test start; interval is too wide (see section 11)",
       x = "Year", y = "Employment (thousands of persons)", colour = NULL) +
  theme_minimal()

ggsave("arx_garch_test_forecast_plot.png", arx_plot,
       width = 9, height = 5, dpi = 300)

# 15. Refit on the full window and forecast 12 months beyond the data
# Intervals are simulated here too. The earlier version wrote a Date/Forecast
# table with no intervals at all, while the README told the reader to quote this
# model's intervals in preference to the others'.
d_all       <- diff(parts$all$value)
d_all_dates <- parts$all$date[-1]
future_dates <- seq(
  from = month_add(max(data$date), 1),
  by = "month", length.out = HORIZON
)

X_future <- if (best_cfg$use_x) covid_dummies(future_dates) else NULL

spec_all  <- build_spec(best_cfg$ar_order, best_cfg$dist,
                        if (best_cfg$use_x) covid_dummies(d_all_dates) else NULL,
                        best_cfg$garch)
fit_final <- ugarchfit(spec_all, data = d_all, solver = "hybrid")
fc_final  <- ugarchforecast(fit_final, n.ahead = HORIZON,
                            external.forecasts = list(mregfor = X_future))
anchor    <- tail(parts$all$value, 1)
final_lvl <- anchor + cumsum(as.numeric(fitted(fc_final)))

future_paths <- simulate_levels(fit_final, anchor, X_future)
fq <- apply(future_paths, 1, quantile, probs = c(0.025, 0.10, 0.90, 0.975))

future_table <- data.frame(Date = future_dates, Forecast = final_lvl,
                           Lo80 = fq[2, ], Hi80 = fq[3, ],
                           Lo95 = fq[1, ], Hi95 = fq[4, ])
cat("\n=== 12-MONTH AHEAD FORECAST (beyond the observed sample) ===\n")
print(future_table, row.names = FALSE, digits = 6)
cat("These intervals are simulated the same way as the test-window ones and\n")
cat("inherit the same calibration problem - they are too wide. Use the empirical\n")
cat("quantiles in empirical_error_quantiles.csv for intervals worth quoting.\n")
write.csv(future_table, "ARX_GARCH_future_forecast.csv", row.names = FALSE)

recent <- tail(parts$all$value, 7)
cat(sprintf("\nLimitation. The series peaked at %s in %s and has fallen since, by a\n",
            format(max(parts$all$value), big.mark = ","),
            format(parts$all$date[which.max(parts$all$value)], "%b %Y")))
cat(sprintf("mean of %.1f thousand per month over the last six months. The drift is\n",
            (recent[7] - recent[1]) / 6))
cat("estimated over the whole window and remains positive, so this forecast\n")
cat("continues to rise through a series that has recently turned.\n")

cat("\nSaved arx_garch_result.csv, ARX_GARCH_forecast.csv,\n")
cat("ARX_GARCH_search.csv, ARX_GARCH_coefficients.csv and\n")
cat("ARX_GARCH_future_forecast.csv\n")
