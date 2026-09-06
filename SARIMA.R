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
#     observations. The earlier run had 113 qualifying models spanning 390.4 to
#     397.3 validation RMSE - a 1.8% spread, which on twelve points is noise.
#     Ranking on it means picking the luckiest of 113. AICc is computed on 919
#     observations and is far more stable. The validation block still does real
#     work through the qualification step and as the honest cross-check printed
#     below.
#
# COVID. The primary search carries NO intervention dummies. Treating the 2020
# break was tested across four intervention windows with and without seasonal
# terms: it cuts the 26-sigma April 2020 residual to about 7 and the excess
# kurtosis from 494 to 10, and it makes the point forecasts worse in every
# configuration tried - the best dummied variant is still worse than a random
# walk with drift. Residual quality and accuracy move in opposite directions and
# no configuration achieves both. The primary model is therefore the accurate one
# and its Gaussian prediction intervals are NOT trustworthy; quote its point
# forecasts, not its intervals. Section 12 fits the dummied alternative as a
# labelled sensitivity. ARX-GARCH.R gets both right, because in difference space
# the COVID event genuinely is four large spikes.
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
  res <- residuals(model)
  res <- res[is.finite(res)]

  model_df  <- p + q + P + Q
  lag_value <- min(24, floor(length(res) / 5))
  lag_value <- max(lag_value, model_df + 3)
  lag_value <- min(lag_value, length(res) - 1)

  if (lag_value <= model_df) return(NA_real_)

  Box.test(res, lag = lag_value, type = "Ljung-Box", fitdf = model_df)$p.value
}

# 4. SPECIFICATION SEARCH - FIT ON TRAIN, QUALIFY ON VALIDATION
# P and Q are allowed to be zero, so a non-seasonal model can win.
# The grid is 225 maximum-likelihood fits and takes 15-30 minutes. It is cached
# so that re-running the selection, the refit or the reporting below does not pay
# for it again. Delete sarima_model_selection.csv, or set SARIMA_REFRESH=TRUE, to
# force a fresh search.
SEARCH_CACHE <- "sarima_model_selection.csv"   # primary search, no dummies
refresh <- !file.exists(SEARCH_CACHE) ||
  isTRUE(as.logical(Sys.getenv("SARIMA_REFRESH", "FALSE")))

candidate_table <- data.frame()
candidate_id    <- 0

if (!refresh) {

  cat("\nReusing cached search from", SEARCH_CACHE, "\n")
  cat("Set SARIMA_REFRESH=TRUE or delete it to re-run the grid.\n")
  candidate_table <- read.csv(SEARCH_CACHE, stringsAsFactors = FALSE)

  # A cache written before Validation_MASE existed can be upgraded in place:
  # MASE is exactly MAE divided by the shared denominator.
  if (is.null(candidate_table$Validation_MASE)) {
    candidate_table$Validation_MASE <- candidate_table$Validation_MAE / MASE_DENOM
    cat("Backfilled Validation_MASE from Validation_MAE / ", round(MASE_DENOM, 2),
        ".\n", sep = "")
  }

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
          Validation_MASE = unname(val_acc["MASE"]),
          Ljung_Box_p     = lb_p,
          AIC = fit$aic, AICc = fit$aicc, BIC = fit$bic,
          stringsAsFactors = FALSE
        ))
      }
    }
  }
}

write.csv(candidate_table, SEARCH_CACHE, row.names = FALSE)

}  # end of search / cache branch

if (nrow(candidate_table) == 0) stop("No SARIMA candidate could be fitted.")

# 5. MODEL SELECTION RULE - stated up front, applied mechanically
#
#   QUALIFY  Ljung-Box p > 0.05  AND  finite standard errors (identifiable)
#   RANK     lowest AICc among the qualifiers
#
# The test block plays no part in any of this.
cat("\n=== MODEL SELECTION PROCESS ===\n")
cat("Candidates fitted    :", nrow(candidate_table), "\n")
cat("Unidentifiable       :", sum(!candidate_table$Identifiable),
    "(rejected: NaN standard errors)\n")
cat("Qualification        : Ljung-Box p > 0.05 AND identifiable\n")
cat("Ranking              : lowest AICc\n")

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
# A specification that converged on the 919-month training block can still fail
# to refit on the 931-month train+validation block - optim() reports
# "non-finite finite-difference value" - or can come back unidentifiable. That is
# a property of the high-order specifications an AICc ranking favours, and it
# happened to the top-ranked model on this data. Walk down the ranking until one
# refits cleanly, and report which rank was actually used.
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
final_ljung_p <- get_ljung_p(final_model, selected$p, selected$q,
                             selected$P, selected$Q)

cat("\n=== FINAL RESIDUAL DIAGNOSTIC ===\n")
cat("Ljung-Box p-value:", final_ljung_p, "\n")
cat("Interpretation:",
    ifelse(is.finite(final_ljung_p) && final_ljung_p > 0.05,
           "no significant residual autocorrelation detected.",
           "residual autocorrelation may remain."), "\n")

png("sarima_final_residual_diagnostics.png",
    width = 1400, height = 900, res = 150)
checkresiduals(final_model)
dev.off()

# 9. FINAL TEST EVALUATION - the test block is read only here
test_fc <- forecast(final_model, h = HORIZON, level = c(80, 95))

# Validation metrics were recorded during the search (train-only fit), so they
# are read from the table rather than recomputed.
val_metrics  <- c(RMSE = selected$Validation_RMSE,
                  MAE  = selected$Validation_MAE,
                  MAPE = selected$Validation_MAPE,
                  MASE = selected$Validation_MASE,
                  N    = HORIZON)
test_metrics <- evaluate(parts$test$value, test_fc$mean,
                         exclude = parts$test$imputed)

cat("\n=== FINAL TEST ACCURACY (2025-08 .. 2026-07) ===\n")
print(round(test_metrics, 4))
cat(sprintf("\nMASE uses the shared denominator %.2f from common.R.\n", MASE_DENOM))
cat(sprintf("Scored on %d observed months; 2025-10 is excluded because it is\n",
            unname(test_metrics["N"])))
cat(sprintf("interpolated, not observed. Including it would give RMSE %.2f.\n",
            unname(evaluate(parts$test$value, test_fc$mean)["RMSE"])))


save_model_result(
  model_id     = MODEL_ID,
  model_name   = selected$Model,
  window_start = min(parts$all$date),
  n_train      = nrow(parts$train),
  n_train_val  = nrow(parts$train_val),
  val_metrics  = val_metrics,
  test_metrics = test_metrics,
  ljung_p      = final_ljung_p,
  identifiable = identifiable,
  aic          = final_model$aic,
  bic          = final_model$bic,
  notes        = sprintf("ranked on AICc among %d qualifiers; seasonal=%s",
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
# Same failure mode as the train+validation refit, so the same guard.
full_model <- refit_on(selected, full_ts)

if (is.null(full_model)) {
  warning("Selected specification could not be refitted on the full series; ",
          "no forward forecast produced.")
  cat("\nWARNING: the full-sample refit failed, so no 12-month forward forecast",
      "\nis produced. The test-set evaluation above is unaffected.\n")
}

future_dates <- seq(
  from = seq(max(data$date), by = "month", length.out = 2)[2],
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
# dummies rather than re-running the whole 225-model search, so this costs one
# extra fit. It exists so the trade-off stated in the header is visible in the
# output instead of merely asserted.
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
    Test_MAPE     = c(test_metrics["MAPE"], sens_metrics["MAPE"]),
    Max_abs_z     = c(prim_resid["max_abs_z"], sens_resid["max_abs_z"]),
    Kurtosis      = c(prim_resid["kurtosis"],  sens_resid["kurtosis"]),
    row.names     = NULL
  )

  print(comparison, row.names = FALSE, digits = 5)

  cat("\nThe dummies cut the largest standardised residual and the excess",
      "\nkurtosis by an order of magnitude - which is what makes Gaussian",
      "\nprediction intervals meaningful - and cost point accuracy to do it.\n")

  write.csv(comparison, "sarima_covid_sensitivity.csv", row.names = FALSE)

  save_model_result(
    model_id     = "sarima_covid_sensitivity",
    model_name   = paste(selected$Model, "+ COVID dummies [sensitivity]"),
    window_start = min(parts$all$date),
    n_train      = nrow(parts$train),
    n_train_val  = nrow(parts$train_val),
    val_metrics  = setNames(rep(NA_real_, 4), c("RMSE", "MAE", "MAPE", "MASE")),
    test_metrics = sens_metrics,
    ljung_p      = NA_real_,
    identifiable = is_identifiable(sens_model),
    aic          = sens_model$aic,
    bic          = sens_model$bic,
    notes        = "SENSITIVITY ONLY - not the reported model"
  )
}

cat("\n=== COMPLETED ===\n")
cat("Selected model:", selected$Model, "\n")
