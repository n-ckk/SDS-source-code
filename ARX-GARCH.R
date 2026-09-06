# ==============================================================================
# BMMS2094 - SDG 5 / SDG 8: US Female Employment Level
# MODEL: ARX(p) with fat-tailed errors, COVID intervention dummies and an
#        optional GARCH(1,1) variance. The AR order, the error distribution, the
#        dummies AND the variance model are all chosen by the rule in section 9;
#        none of them is hardcoded.
#
# Split, metrics, the MASE denominator and the COVID regressors come from
# common.R. Do not redefine them here.
#
# WHY THIS MODEL. The series is I(1) with no seasonality, so the mean equation is
# an AR model on the FIRST DIFFERENCES with a constant, which is a drift in
# levels. The irregular component is the problem: a plain ARIMA on this window
# leaves a 12.2-sigma residual in April 2020 and residual kurtosis of 117. Two
# additions deal with that:
#   (1) COVID pulse dummies - take the four shock months out of the mean
#       equation. Every specification without them fails the squared-residual
#       diagnostic; every specification with them satisfies it. This accounts for
#       most of the gain.
#   (2) Fat-tailed errors   - let the likelihood accept fat tails instead of
#       inflating sigma to cover them.
#
# THE GARCH COMPONENT IS NOT ASSUMED - IT IS TESTED FOR. The ARCH-LM test on
# plain ARIMA residuals does not reject homoskedasticity (section 5, p = 0.9996),
# so constant-variance specifications are included in the search on equal terms
# and the rule in section 7 decides between them. What the search actually finds:
#   - constant-variance specs reach a slightly LOWER raw BIC (14.075 against
#     14.107 for the best GARCH spec), but every one of them FAILS the
#     squared-residual Ljung-Box diagnostic (LBsq_p around 0.003) and is
#     therefore disqualified;
#   - among the QUALIFYING specifications, GARCH(1,1) wins by 0.243 of BIC.
# So the GARCH component does earn its place on this evidence, even though the
# ARCH-LM test on a plain ARIMA detected no effect. The two tests examine
# different residuals; report both rather than only the one that agrees with the
# model you wanted.
#
# CALIBRATION CAVEAT that survives all of the above: alpha1 + beta1 = 0.999 (a
# boundary IGARCH), omega is not significant (p = 0.41), and the simulated
# prediction intervals cover 100% of the test points at BOTH the 80% and 95%
# levels, so they are too WIDE rather than well calibrated. Report as a
# limitation.
#
# SEASONALITY. STL seasonal strength is F_s = 0.072 on this 2010+ analysis window
# (it is 0.012 on the full 1948+ history - quote the figure for the window you
# actually model), and nsdiffs() = 0. No seasonal terms are carried.
#
# THREE-WAY SPLIT. Defined once in common.R and shared with every other model:
#   TRAIN      2010-01 .. 2024-07   fit the candidates
#   VALIDATION 2024-08 .. 2025-07   qualify them
#   TEST       2025-08 .. 2026-07   read ONCE, in section 12
# The chosen specification is refitted on TRAIN + VALIDATION before it forecasts
# the test window. The test window spans the Jan 2026 peak and the decline after
# it, so it is a turning-point test rather than a purely trending one.
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
ANALYSIS_START <- as.Date("2010-01-01")

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
# FITTED ON TRAIN, QUALIFIED ON VALIDATION. The test set is not touched here.
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
# residual rises to about 12.7 rather than 5. Section 11 checks that the fitted
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
      LBsq_p   = Box.test(sr^2, lag = 12, type = "Ljung-Box")$p.value,
      MaxZ     = max(abs(sr)),
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
#   QUALIFY  a specification must leave no structure in the residuals:
#            Ljung-Box on standardised residuals    LB24_p > 0.05
#            Ljung-Box on squared standardised res  LBsq_p > 0.05
#   RANK     among the qualifiers, lowest BIC wins.
#
# WHY BIC AND NOT VALIDATION RMSE. The validation block is 12 observations and
# the spread across qualifying specs is only a few percent, which on 12 points is
# noise - ranking on it would be picking the luckiest of the grid, the failure
# mode this three-way split exists to prevent. BIC is computed on 174
# observations and is far more stable. The validation block still does real work
# through the QUALIFY step and as the cross-check printed below.
#
# "std" = standardised Student-t in rugarch's naming, "sstd" = skewed version.
#
# max|z| is NOT comparable across distributions: normal errors inflate sigma_t to
# cover the tails, which mechanically shrinks the standardised residuals, so a
# smaller max|z| under normal errors is not evidence of a better fit.
LB_ALPHA <- 0.05

qualified <- tab[tab$LB24_p > LB_ALPHA & tab$LBsq_p > LB_ALPHA, ]
cat("\n=== QUALIFYING SPECIFICATIONS (LB24_p > 0.05 and LBsq_p > 0.05) ===\n")
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

# Does the GARCH variance actually earn its place? Report the margin explicitly.
cat("\n=== DOES THE GARCH VARIANCE EARN ITS PLACE? ===\n")
best_garch <- qualified[qualified$Variance == "GARCH(1,1)", ]
best_const <- qualified[qualified$Variance == "constant", ]
if (nrow(best_garch) && nrow(best_const)) {
  bg <- min(best_garch$BIC); bc <- min(best_const$BIC)
  cat(sprintf("Best qualifying GARCH(1,1) BIC : %.5f\n", bg))
  cat(sprintf("Best qualifying constant-var BIC: %.5f\n", bc))
  cat(sprintf("Margin: %+.5f in favour of %s\n", bc - bg,
              ifelse(bg < bc, "GARCH(1,1)", "constant variance")))
  cat("The ARCH-LM test found no ARCH effect, so a small margin here means the\n",
      "GARCH component is a robust estimator, not a detected effect. Say so.\n")
} else {
  cat("Only one variance family qualified; report that directly.\n")
}

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

sr <- as.numeric(residuals(fit, standardize = TRUE))
cat("\n=== STANDARDISED-RESIDUAL DIAGNOSTICS ===\n")
print(Box.test(sr,   lag = 12, type = "Ljung-Box", fitdf = best_cfg$ar_order))
print(Box.test(sr,   lag = 24, type = "Ljung-Box", fitdf = best_cfg$ar_order))
cat("Ljung-Box on SQUARED standardised residuals (leftover ARCH):\n")
print(Box.test(sr^2, lag = 12, type = "Ljung-Box"))
# Jarque-Bera is expected to reject when Student-t errors are selected: the
# standardised residuals should then follow a t distribution with the estimated
# shape parameter, not a normal, and the model carries no normality assumption
# for the rejection to contradict. The tails are assessed instead by max|z| and
# by whether the interval coverage reported below is close to nominal.
print(jarque.bera.test(sr))
cat(sprintf("max |standardised residual| = %.2f  (plain ARIMA on this window: 12.2)\n",
            max(abs(sr))))
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
sim <- ugarchsim(
  fit, n.sim = HORIZON, m.sim = NSIM,
  startMethod = "sample",
  mexsimdata  = if (best_cfg$use_x) replicate(NSIM, X_test, simplify = FALSE) else NULL
)
lvl_paths <- last_lvl_tv + apply(fitted(sim), 2, cumsum)
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
cat("Imputed = the actual is interpolated, not observed.\n")
print(forecast_tbl, row.names = FALSE, digits = 6)

write.csv(forecast_tbl, "ARX_GARCH_forecast.csv", row.names = FALSE)

cov80 <- mean(test_lvl >= forecast_tbl$Lo80 & test_lvl <= forecast_tbl$Hi80)
cov95 <- mean(test_lvl >= forecast_tbl$Lo95 & test_lvl <= forecast_tbl$Hi95)
cat(sprintf("\n80%% PI coverage = %.0f%%   95%% PI coverage = %.0f%%\n",
            100 * cov80, 100 * cov95))
cat("Nominal coverage on 12 points is about 10/12 and 11/12. Coverage of 100% at\n")
cat("BOTH levels means the intervals are too WIDE, not well calibrated - report\n")
cat("this as a limitation rather than as a success.\n")

# 12. Accuracy
# MASE uses the single shared denominator from common.R. The earlier version of
# this script computed three candidate denominators of its own; that choice now
# lives in one place so every model in the group table is scaled identically.
# The validation forecast was produced during the search by the train-only fit;
# reuse it rather than refitting.
val_metrics  <- evaluate(val_lvl, best_cfg$val_lvl_fc)
test_metrics <- evaluate(test_lvl, point_lvl,
                         exclude = parts$test$imputed)

# Benchmark: random walk with drift, the minimum standard the model must beat.
drift  <- (last_lvl_tv - tv_lvl[1]) / (length(tv_lvl) - 1)
rw_fc  <- last_lvl_tv + drift * seq_len(HORIZON)
rw_metrics <- evaluate(test_lvl, rw_fc)

cat("\n=== ACCURACY (all on the shared MASE denominator", round(MASE_DENOM, 2), ") ===\n")
print(rbind(Validation = val_metrics, Test = test_metrics,
            `Benchmark RW-drift` = rw_metrics), digits = 5)
cat(sprintf("\nTest RMSE vs RW-drift benchmark: %+.1f%%  (negative = model is better)\n",
            100 * (test_metrics["RMSE"] / rw_metrics["RMSE"] - 1)))

# NOTE: no test/train ratio is reported. Dividing a 12-step-ahead test error by
# 1-step-ahead in-sample residuals compares two different quantities, and on a
# window containing an untreated structural break it can "PASS" simply because
# the training residuals are inflated. Validation vs test above is the honest
# comparison: both are 12-step-ahead forecasts of unseen blocks.

save_model_result(
  model_id     = MODEL_ID,
  model_name   = BEST,
  window_start = ANALYSIS_START,
  n_train      = nrow(parts$train),
  n_train_val  = nrow(parts$train_val),
  val_metrics  = val_metrics,
  test_metrics = test_metrics,
  ljung_p      = Box.test(sr, lag = 24, type = "Ljung-Box",
                          fitdf = best_cfg$ar_order)$p.value,
  identifiable = all(is.finite(as.numeric(fit@fit$se.coef))),
  aic          = infocriteria(fit)[1] * length(d_tv),
  bic          = infocriteria(fit)[2] * length(d_tv),
  notes        = sprintf("PI coverage 80/95 = %.0f%%/%.0f%%; ARCH-LM p = %.3f",
                         100 * cov80, 100 * cov95, arch_test$p.value)
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
       subtitle = "dotted = validation start, dashed = test start",
       x = "Year", y = "Employment (thousands of persons)", colour = NULL) +
  theme_minimal()

ggsave("arx_garch_test_forecast_plot.png", arx_plot,
       width = 9, height = 5, dpi = 300)

# 15. Refit on the full window and forecast 12 months beyond the data
d_all       <- diff(parts$all$value)
d_all_dates <- parts$all$date[-1]
future_dates <- seq(
  from = seq(max(data$date), by = "month", length.out = 2)[2],
  by = "month", length.out = HORIZON
)

spec_all  <- build_spec(best_cfg$ar_order, best_cfg$dist,
                        if (best_cfg$use_x) covid_dummies(d_all_dates) else NULL,
                        best_cfg$garch)
fit_final <- ugarchfit(spec_all, data = d_all, solver = "hybrid")
fc_final  <- ugarchforecast(fit_final, n.ahead = HORIZON,
                            external.forecasts = list(
                              mregfor = if (best_cfg$use_x)
                                covid_dummies(future_dates) else NULL))
final_lvl <- tail(parts$all$value, 1) + cumsum(as.numeric(fitted(fc_final)))

future_table <- data.frame(Date = future_dates, Forecast = final_lvl)
cat("\n=== 12-MONTH AHEAD FORECAST (beyond the observed sample) ===\n")
print(future_table, row.names = FALSE, digits = 6)
write.csv(future_table, "ARX_GARCH_future_forecast.csv", row.names = FALSE)

recent <- tail(parts$all$value, 7)
cat(sprintf("\nLimitation. The series peaked at %s in %s and has fallen since, by a\n",
            format(max(parts$all$value), big.mark = ","),
            format(parts$all$date[which.max(parts$all$value)], "%b %Y")))
cat(sprintf("mean of %.1f thousand per month over the last six months. The drift is\n",
            (recent[7] - recent[1]) / 6))
cat("estimated over the whole window and remains positive, so this forecast\n")
cat("continues to rise through a series that has recently turned.\n")

cat("\nSaved ARX_GARCH_results (via save_model_result), ARX_GARCH_forecast.csv,\n")
cat("ARX_GARCH_search.csv and ARX_GARCH_coefficients.csv\n")
