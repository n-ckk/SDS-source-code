# ==============================================================================
# SARIMA - Dataset: LNS12000002 - Employment Level: Women
#
# WHAT THIS MODEL IS FOR
# LNS12000002 is published SEASONALLY ADJUSTED by BLS, so this script is not an
# attempt to exploit seasonality. It is the TEST for whether any exploitable
# seasonality survives that adjustment. The seasonal orders are free to come out
# zero, and reporting that they do is the finding.
#
# THREE CHANGES OF SUBSTANCE OVER THE EARLIER VERSION
#
# (1) The search no longer forces a seasonal term. It previously skipped every
#     candidate with P + Q == 0, which guaranteed a seasonal model on a
#     deseasonalised series. The term it was forced to pick came out at
#     sar1 = 0.0264 with s.e. 0.0341 (t = 0.77, p = 0.44) - indistinguishable
#     from zero. Corroborating evidence: STL seasonal strength F_s = 0.012 on the
#     full history, nsdiffs() = 0 on every window tested, and auto.arima given a
#     free choice selects a non-seasonal model.
#
# (2) Candidates must be IDENTIFIABLE. The forced search selected
#     SARIMA(4,1,3)(1,0,0)[12], whose Hessian is singular: five of its nine
#     standard errors came back NaN, so its coefficients, information criteria
#     and prediction intervals were all unusable. It nevertheless posted the best
#     test RMSE in the group, which is precisely why test error cannot be the
#     only gate. is_identifiable() in common.R now rejects such fits.
#
# (3) Ranking is on AICc, not validation RMSE. The validation block is twelve
#     observations, and the earlier run had 113 qualifying models spanning a 1.8%
#     validation-RMSE range - noise on twelve points. AICc is computed on 919
#     observations and is far more stable.
#
# WHAT THE VALIDATION BLOCK ACTUALLY DOES HERE - honestly stated.
# NOTHING in the selection. An earlier version of this header claimed the
# validation block "does real work through the qualification step"; it does not.
# Qualification is Ljung-Box plus identifiability, both computed IN SAMPLE on the
# training fit, and ranking is on in-sample AICc. This is in substance a two-way
# split with an in-sample selection criterion. The validation block's only roles
# are the cross-check printed in section 5 and the genuinely out-of-sample
# Val_RMSE that reaches the comparison table. That is still worth having - it is
# just not model selection, and calling it selection overstated the design.
#
# COVID. The primary search carries NO intervention dummies. Treating the 2020
# break cuts the ~26-sigma April 2020 residual to about 7 and the excess kurtosis
# by roughly an order of magnitude, and makes the point forecasts worse. Residual
# quality and accuracy move in opposite directions and no configuration achieves
# both. The primary model is therefore the accurate one and its Gaussian
# prediction intervals are NOT trustworthy; quote its point forecasts, not its
# intervals. Section 12 MEASURES the trade-off on every run rather than quoting
# it from prose - see sarima_covid_sensitivity.csv. For calibrated intervals use
# the empirical error quantiles from rolling_cv.R.
# ==============================================================================

library(forecast)

source("common.R")

MODEL_ID <- "sarima"

# 1. LOAD DATA AND SPLIT
data  <- load_series()
parts <- split_series(data)
describe_split(parts, "SARIMA (full history)")

train_ts     <- as_monthly_ts(parts$train)
train_val_ts <- as_monthly_ts(parts$train_val)
full_ts      <- as_monthly_ts(parts$all)

# Built here but used ONLY by the sensitivity in section 12.
X_train_val <- covid_dummies(parts$train_val$date)
X_test      <- covid_dummies(parts$test$date)

# 2. DETERMINE DIFFERENCING ORDERS
d <- ndiffs(train_ts, test = "kpss", max.d = 2)
D <- nsdiffs(train_ts, test = "seas", max.D = 1)

cat("\n=== DIFFERENCING ===\n")
cat("d =", d, "\n")
cat("D =", D, " (0 means no seasonal differencing is required)\n")

# 3. HELPER: Ljung-Box p-value with the right degrees of freedom
get_ljung_p <- function(model, p, q, P, Q) {
  ljung_box(residuals(model), fitdf = p + q + P + Q,
            on = "level residuals")$p
}

# 4. SPECIFICATION SEARCH - FIT ON TRAIN, SCORE ON VALIDATION
# P and Q are allowed to be zero, so a non-seasonal model can win.
# The grid is 225 maximum-likelihood fits and takes 15-30 minutes. It is cached
# so that re-running the selection, the refit or the reporting below does not pay
# for it again.
#
# THE CACHE IS FINGERPRINTED. It previously had no record of what it was built
# from, so changing VAL_START, TEST_START, the data or the MASE denominator left
# a stale table on disk that the script would happily reuse - every cached
# Validation_RMSE and AICc silently describing a different question from the one
# being asked. The fingerprint below covers the split boundaries and the training
# data itself; a mismatch forces a fresh search instead of a wrong answer.
SEARCH_CACHE <- "sarima_model_selection.csv"   # primary search, no dummies

fingerprint <- sprintf("val=%s;test=%s;freq=%d;n=%d;s1=%.4f;s2=%.4f;d=%d;D=%d",
                       format(VAL_START), format(TEST_START), FREQ,
                       length(train_ts), sum(train_ts), sum(train_ts^2), d, D)

cache_ok <- FALSE
if (file.exists(SEARCH_CACHE)) {
  cached <- read.csv(SEARCH_CACHE, stringsAsFactors = FALSE)
  cache_ok <- !is.null(cached$Fingerprint) &&
    nrow(cached) > 0 &&
    identical(as.character(cached$Fingerprint[1]), fingerprint)
  if (!cache_ok) {
    cat("\nCached search in", SEARCH_CACHE, "does not match the current data or\n")
    cat("split (missing or stale fingerprint). Re-running the grid.\n")
  }
}

refresh <- !cache_ok || isTRUE(as.logical(Sys.getenv("SARIMA_REFRESH", "FALSE")))

candidate_table <- data.frame()
candidate_id    <- 0

if (!refresh) {

  cat("\nReusing cached search from", SEARCH_CACHE, "\n")
  cat("Fingerprint matches. Set SARIMA_REFRESH=TRUE or delete it to re-run.\n")
  candidate_table <- cached

} else {

cat("\nSearching p 0:4, q 0:4, P 0:2, Q 0:2 (seasonal terms may be zero)...\n")

for (p in 0:4) {
  for (q in 0:4) {
    for (P in 0:2) {
      for (Q in 0:2) {

        fit <- tryCatch(
          Arima(train_ts,
                order    = c(p, d, q),
                seasonal = list(order = c(P, D, Q), period = FREQ),
                include.drift = (d == 1 && D == 0),
                method   = "ML"),
          error = function(e) NULL
        )
        # NOTE: warnings are deliberately NOT trapped here. Trapping them would
        # discard fits that merely warn about a near-non-invertible root, which
        # are often fine. The precise guard is is_identifiable() below, which
        # tests the thing that actually matters - a singular Hessian.

        if (is.null(fit)) next

        # An unidentifiable fit is discarded here, before it can be ranked.
        identifiable <- is_identifiable(fit)

        val_fc <- tryCatch(forecast(fit, h = HORIZON),
                           error = function(e) NULL)
        if (is.null(val_fc)) next

        val_acc <- evaluate(parts$val$value, val_fc$mean)
        lb_p <- tryCatch(get_ljung_p(fit, p, q, P, Q),
                         error = function(e) NA_real_)

        candidate_id <- candidate_id + 1

        candidate_table <- rbind(candidate_table, data.frame(
          ID    = candidate_id,
          Model = sprintf("SARIMA(%d,%d,%d)(%d,%d,%d)[%d]",
                          p, d, q, P, D, Q, FREQ),
          p = p, d = d, q = q, P = P, D = D, Q = Q,
          Seasonal        = (P + Q) > 0,
          Identifiable    = identifiable,
          Validation_RMSE = unname(val_acc["RMSE"]),
          Validation_MAE  = unname(val_acc["MAE"]),
          Validation_MAPE = unname(val_acc["MAPE"]),
          Ljung_Box_p     = lb_p,
          AIC = fit$aic, AICc = fit$aicc, BIC = fit$bic,
          Fingerprint     = fingerprint,
          stringsAsFactors = FALSE
        ))
      }
    }
  }
}

write.csv(candidate_table, SEARCH_CACHE, row.names = FALSE)

}  # end of search / cache branch

if (nrow(candidate_table) == 0) stop("No SARIMA candidate could be fitted.")

# MASE columns are DERIVED, never cached. MASE is MAE divided by a denominator
# that lives in common.R, so recomputing it here means a change to that
# denominator propagates into the cached table instead of being silently ignored.
candidate_table$Validation_MASE   <- candidate_table$Validation_MAE / MASE_DENOM
candidate_table$Validation_MASE_s <- candidate_table$Validation_MAE / MASE_DENOM_S

cat("Candidates in table  :", nrow(candidate_table), "of 225 attempted",
    sprintf("(%d specifications failed to converge)\n", 225 - nrow(candidate_table)))

# 5. MODEL SELECTION RULE - stated up front, applied mechanically
#
#   QUALIFY  Ljung-Box p > 0.05  AND  finite standard errors (identifiable)
#   RANK     lowest AICc among the qualifiers
#
# Both are IN-SAMPLE criteria computed on the training fit. Neither the
# validation block nor the test block plays any part.
cat("\n=== MODEL SELECTION PROCESS ===\n")
cat("Candidates fitted    :", nrow(candidate_table), "\n")
cat("Unidentifiable       :", sum(!candidate_table$Identifiable),
    "(rejected: NaN standard errors)\n")
cat("Qualification        : Ljung-Box p > 0.05 AND identifiable (both in-sample)\n")
cat("Ranking              : lowest AICc (in-sample)\n")
cat("Validation block     : NOT used for selection - cross-check only\n")

qualified <- candidate_table[
  is.finite(candidate_table$Ljung_Box_p) &
    candidate_table$Ljung_Box_p > 0.05 &
    candidate_table$Identifiable,
]

if (nrow(qualified) == 0) {
  cat("\nNo model passed qualification. Falling back to identifiable models",
      "ranked by AICc - report this.\n")
  qualified <- candidate_table[candidate_table$Identifiable, ]
}
if (nrow(qualified) == 0) stop("No identifiable SARIMA candidate was found.")

cat("Qualified models     :", nrow(qualified), "\n")

ranked <- qualified[order(qualified$AICc), ]
selected <- ranked[1, ]   # AICc-best; section 7 reassigns if it cannot refit

cat("\nTop 10 candidates by AICc:\n")
print(head(ranked[, c("Model", "Seasonal", "AICc", "Validation_RMSE",
                      "Ljung_Box_p")], 10),
      row.names = FALSE, digits = 6)

cat("\nSelected model:", selected$Model, "\n")
cat("AICc           :", selected$AICc, "\n")
cat("Validation RMSE:", selected$Validation_RMSE, "\n")
cat("Ljung-Box p    :", selected$Ljung_Box_p, "\n")

# Honest cross-check: say so when the two criteria disagree.
by_val <- qualified$Model[which.min(qualified$Validation_RMSE)]
cat("\nCross-check - lowest validation RMSE among qualifiers would pick:",
    by_val, "\n")
if (!identical(by_val, selected$Model)) {
  cat("These disagree. AICc is ranked on because the validation block is only",
      HORIZON, "points;\nreport both rather than quoting only one.\n")
}

# 6. THE SEASONALITY FINDING - the reason this script exists
cat("\n=== SEASONALITY TEST ===\n")
cat("Seasonal differencing required (D)  :", D, "\n")
cat("Selected seasonal orders (P, Q)     :", selected$P, ",", selected$Q, "\n")

if (selected$P + selected$Q == 0) {
  cat("RESULT: the selected model is NON-SEASONAL. No exploitable seasonality\n",
      "       survives the BLS seasonal adjustment, so SARIMA reduces to ARIMA.\n")
} else {
  cat("RESULT: seasonal terms were selected. Check their significance below\n",
      "       before claiming the series carries genuine seasonality.\n")
}

# 7. REFIT ON TRAIN + VALIDATION, WALKING DOWN THE RANKING IF NECESSARY
# A specification that converged on the training block can still fail to refit on
# the longer train+validation block - optim() reports "non-finite
# finite-difference value" - or can come back unidentifiable. Walk down the
# ranking until one refits cleanly, and report which rank was actually used.
refit_on <- function(row, series, xreg = NULL) {
  tryCatch(
    Arima(series,
          order    = c(row$p, row$d, row$q),
          seasonal = list(order = c(row$P, row$D, row$Q), period = FREQ),
          xreg     = xreg,
          include.drift = (row$d == 1 && row$D == 0),
          method   = "ML"),
    error = function(e) NULL
  )
}

final_model <- NULL
rank_used   <- NA_integer_

for (i in seq_len(nrow(ranked))) {
  cand_fit <- refit_on(ranked[i, ], train_val_ts)
  if (!is.null(cand_fit) && is_identifiable(cand_fit)) {
    final_model <- cand_fit
    rank_used   <- i
    break
  }
  cat(sprintf("  rank %d (%s) unusable on train+validation - %s\n",
              i, ranked$Model[i],
              if (is.null(cand_fit)) "estimation failed" else "unidentifiable"))
}

if (is.null(final_model)) {
  stop("No qualifying specification could be refitted on train + validation.")
}

selected <- ranked[rank_used, ]

cat("\n=== FINAL MODEL: REFITTED ON TRAIN + VALIDATION ===\n")
if (rank_used > 1) {
  cat(sprintf("NOTE: ranks 1..%d could not be refitted; using rank %d (%s).\n",
              rank_used - 1, rank_used, selected$Model))
  cat("Report this - the AICc-best specification was not usable.\n")
  cat(sprintf("Final seasonal orders (P, Q) = %d, %d%s\n",
              selected$P, selected$Q,
              if (selected$P + selected$Q == 0) "  -> NON-SEASONAL" else ""))
}
print(final_model)

identifiable <- is_identifiable(final_model)
cat("\nIdentifiable (no NaN standard errors):", identifiable, "\n")

# Coefficient significance, so the report can quote t-values rather than adjectives
se <- suppressWarnings(sqrt(diag(final_model$var.coef)))
parameter_table <- data.frame(
  Parameter = names(coef(final_model)),
  Estimate  = as.numeric(coef(final_model)),
  StdError  = as.numeric(se),
  tvalue    = as.numeric(coef(final_model)) / as.numeric(se)
)
parameter_table$pvalue <- 2 * pnorm(-abs(parameter_table$tvalue))

cat("\n=== FINAL COEFFICIENTS ===\n")
print(parameter_table, row.names = FALSE, digits = 5)
write.csv(parameter_table, "sarima_final_parameters.csv", row.names = FALSE)

# 8. FINAL RESIDUAL DIAGNOSTICS
lb <- ljung_box(residuals(final_model),
                fitdf = selected$p + selected$q + selected$P + selected$Q,
                on = "level residuals")

cat("\n=== FINAL RESIDUAL DIAGNOSTIC ===\n")
cat(sprintf("Ljung-Box: lag = %d, df = %d, n = %d, p = %.4f\n",
            lb$lag, lb$df, lb$n, lb$p))
cat("Interpretation:",
    ifelse(is.finite(lb$p) && lb$p > 0.05,
           "no significant residual autocorrelation detected.",
           "residual autocorrelation may remain."), "\n")

png("sarima_final_residual_diagnostics.png",
    width = 1400, height = 900, res = 150)
checkresiduals(final_model)
dev.off()

# 9. FINAL TEST EVALUATION - the test block is read only here
test_fc <- forecast(final_model, h = HORIZON, level = c(80, 95))

# Validation metrics were recorded during the search (train-only fit), so they
# are read from the table rather than recomputed. The SELECTED row's orders were
# refitted unchanged on train+validation above, so - unlike auto_arima.R, which
# re-runs its whole selection - the Val_RMSE below does belong to the reported
# specification.
val_metrics  <- c(RMSE   = selected$Validation_RMSE,
                  MAE    = selected$Validation_MAE,
                  MAPE   = selected$Validation_MAPE,
                  MASE   = selected$Validation_MASE,
                  MASE_s = selected$Validation_MASE_s,
                  N      = HORIZON)
test_metrics <- evaluate(parts$test$value, test_fc$mean,
                         exclude = parts$test$imputed)

cat("\n=== FINAL TEST ACCURACY (2025-08 .. 2026-07) ===\n")
print(round(test_metrics, 4))
cat(sprintf("\nMASE uses the shared lag-1 denominator %.2f from common.R;\n",
            MASE_DENOM))
cat(sprintf("MASE_s uses the seasonal-naive denominator %.2f and is never quoted alone.\n",
            MASE_DENOM_S))
cat(sprintf("Scored on %d observed months; 2025-10 is excluded because it is\n",
            unname(test_metrics["N"])))
cat(sprintf("interpolated, not observed. Including it would give RMSE %.2f.\n",
            unname(evaluate(parts$test$value, test_fc$mean)["RMSE"])))

bench <- benchmark_table()
cat("\nAgainst the shared benchmarks (same scored months):\n")
print(round(bench[, c("RMSE", "MAE", "MASE")], 3))
cat(sprintf("Test RMSE vs RW-with-drift: %+.1f%%  (negative = model is better)\n",
            100 * (test_metrics["RMSE"] / bench["RW with drift", "RMSE"] - 1)))

save_model_result(
  model_id     = MODEL_ID,
  model_name   = selected$Model,
  window_start = min(parts$all$date),
  n_train      = nrow(parts$train),
  n_train_val  = nrow(parts$train_val),
  val_metrics  = val_metrics,
  test_metrics = test_metrics,
  lb           = lb,
  identifiable = identifiable,
  aic          = final_model$aic,
  bic          = final_model$bic,
  spec         = sprintf("kind=arima;p=%d;d=%d;q=%d;P=%d;D=%d;Q=%d;drift=%s",
                         selected$p, selected$d, selected$q,
                         selected$P, selected$D, selected$Q,
                         selected$d == 1 && selected$D == 0),
  notes        = sprintf("ranked on in-sample AICc among %d qualifiers; seasonal=%s",
                         nrow(qualified), selected$P + selected$Q > 0)
)

# 10. TEST FORECAST TABLE + PLOT
test_forecast_table <- data.frame(
  Date     = parts$test$date,
  Actual   = parts$test$value,
  Forecast = as.numeric(test_fc$mean),
  Error    = parts$test$value - as.numeric(test_fc$mean),
  Lower_80 = as.numeric(test_fc$lower[, "80%"]),
  Upper_80 = as.numeric(test_fc$upper[, "80%"]),
  Lower_95 = as.numeric(test_fc$lower[, "95%"]),
  Upper_95 = as.numeric(test_fc$upper[, "95%"]),
  Imputed  = parts$test$imputed
)

cat("\nTest forecast (Imputed = actual is interpolated, not observed):\n")
print(test_forecast_table, row.names = FALSE, digits = 6)
cat("The intervals are the model's own Gaussian ones and are NOT calibrated on\n")
cat("this fit - use empirical_error_quantiles.csv from rolling_cv.R instead.\n")

write.csv(test_forecast_table, "sarima_final_test_forecast.csv",
          row.names = FALSE)

png("sarima_final_test_forecast_plot.png",
    width = 1400, height = 900, res = 150)
plot(test_fc,
     main = paste0(selected$Model, " - Final Test Forecast vs Actual"),
     xlab = "Year", ylab = "Female Employment Level (Thousands)",
     include = 60)
lines(as_monthly_ts(parts$test), lwd = 2)
dev.off()

# 11. REFIT ON FULL DATA + FUTURE 12-MONTH FORECAST
# The SELECTED orders are refitted, not re-searched, so the forward forecast
# comes from the same specification whose accuracy was measured above.
full_model <- refit_on(selected, full_ts)

if (is.null(full_model)) {
  warning("Selected specification could not be refitted on the full series; ",
          "no forward forecast produced.")
  cat("\nWARNING: the full-sample refit failed, so no 12-month forward forecast",
      "\nis produced. The test-set evaluation above is unaffected.\n")
}

future_dates <- seq(
  from = month_add(max(data$date), 1),
  by = "month", length.out = HORIZON
)

if (!is.null(full_model)) {

  future_fc <- forecast(full_model, h = HORIZON, level = c(80, 95))

  future_table <- data.frame(
    Date     = future_dates,
    Forecast = as.numeric(future_fc$mean),
    Lower_80 = as.numeric(future_fc$lower[, "80%"]),
    Upper_80 = as.numeric(future_fc$upper[, "80%"]),
    Lower_95 = as.numeric(future_fc$lower[, "95%"]),
    Upper_95 = as.numeric(future_fc$upper[, "95%"])
  )

  cat("\n=== 12-MONTH AHEAD FORECAST (beyond the observed sample) ===\n")
  print(future_table, row.names = FALSE, digits = 6)
  cat("Intervals are the model's own Gaussian ones - NOT calibrated. Use the\n")
  cat("empirical quantiles from rolling_cv.R for intervals worth quoting.\n")

  write.csv(future_table, "sarima_future_12_month_forecast.csv", row.names = FALSE)

  png("sarima_future_12_month_forecast.png",
      width = 1400, height = 900, res = 150)
  plot(future_fc,
       main = paste0(selected$Model, " - 12-Month Future Forecast"),
       xlab = "Year", ylab = "Female Employment Level (Thousands)",
       include = 60)
  dev.off()

}

# ==============================================================================
# 12. SENSITIVITY: WHAT COVID INTERVENTION DUMMIES COST AND BUY
#
# Not the reported model. The SELECTED orders are refitted with the intervention
# dummies rather than re-running the whole 225-model search, so this is a
# CONTROLLED comparison - one thing changes - and costs one extra fit. It exists
# so the trade-off stated in the header is measured on every run instead of being
# quoted from prose that can go stale.
# ==============================================================================
cat("\n\n=== SENSITIVITY: COVID DUMMIES (not the reported model) ===\n")

sens_model <- refit_on(selected, train_val_ts, X_train_val)

if (is.null(sens_model)) {

  cat("The selected orders could not be refitted with intervention dummies;",
      "\nno sensitivity reported.\n")

} else {

  sens_fc      <- forecast(sens_model, xreg = X_test, level = c(80, 95))
  sens_metrics <- evaluate(parts$test$value, sens_fc$mean,
                           exclude = parts$test$imputed)

  prim_resid <- residual_summary(final_model)
  sens_resid <- residual_summary(sens_model)

  comparison <- data.frame(
    Specification = c("Primary (no dummies)", "Sensitivity (COVID dummies)"),
    Model         = c(selected$Model, paste(selected$Model, "+ AO")),
    Test_RMSE     = c(test_metrics["RMSE"], sens_metrics["RMSE"]),
    Test_MAE      = c(test_metrics["MAE"],  sens_metrics["MAE"]),
    Test_MASE     = c(test_metrics["MASE"], sens_metrics["MASE"]),
    Max_abs_z     = c(prim_resid["max_abs_z"], sens_resid["max_abs_z"]),
    Kurtosis      = c(prim_resid["kurtosis"],  sens_resid["kurtosis"]),
    row.names     = NULL
  )

  print(comparison, row.names = FALSE, digits = 5)

  cat(sprintf("\nThe dummies cost %.1f RMSE (%+.1f%%) and cut max|z| from %.1f to %.1f\n",
              sens_metrics["RMSE"] - test_metrics["RMSE"],
              100 * (sens_metrics["RMSE"] / test_metrics["RMSE"] - 1),
              prim_resid["max_abs_z"], sens_resid["max_abs_z"]))
  cat(sprintf("and excess kurtosis from %.0f to %.1f - which is what makes Gaussian\n",
              prim_resid["kurtosis"], sens_resid["kurtosis"]))
  cat("prediction intervals meaningful. This is a controlled comparison: the\n")
  cat("orders are identical and only the dummies change.\n")

  write.csv(comparison, "sarima_covid_sensitivity.csv", row.names = FALSE)

  save_model_result(
    model_id     = "sarima_covid_sensitivity",
    model_name   = paste(selected$Model, "+ COVID dummies [sensitivity]"),
    window_start = min(parts$all$date),
    n_train      = nrow(parts$train),
    n_train_val  = nrow(parts$train_val),
    val_metrics  = setNames(rep(NA_real_, 3), c("RMSE", "MAE", "MAPE")),
    test_metrics = sens_metrics,
    lb           = ljung_box(residuals(sens_model),
                             fitdf = selected$p + selected$q +
                                     selected$P + selected$Q,
                             on = "level residuals"),
    identifiable = is_identifiable(sens_model),
    aic          = sens_model$aic,
    bic          = sens_model$bic,
    spec         = NA_character_,
    notes        = "SENSITIVITY ONLY - controlled (same orders as primary)"
  )
}

cat("\n=== COMPLETED ===\n")
cat("Selected model:", selected$Model, "\n")
