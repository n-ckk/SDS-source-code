# ============================================================
# AUTO ARIMA MODEL
# SDG 5: Gender Equality
# Dataset: LNS12000002 - Employment Level: Women
#
# Split, metrics, MASE denominators, benchmarks and COVID regressors all come
# from common.R. Do not redefine any of them here.
#
# COVID AND THE TRADE-OFF (read before changing the specification).
# The 2020 break is NOT treated in the primary specification, and that is a
# deliberate, measured choice rather than an oversight. Treating it with additive
# outlier dummies does exactly what it is supposed to do - it cuts the 26-sigma
# April 2020 residual to about 7 and the excess kurtosis from ~500 to ~10 - and
# it costs point accuracy to do it. Residual quality and point accuracy move in
# opposite directions here and no configuration achieves both.
#
# NO NUMBERS ARE QUOTED IN THIS HEADER ON PURPOSE. An earlier version carried a
# hardcoded eight-row table of test RMSEs. Those numbers were measured before the
# interpolated month was excluded from scoring, nothing recomputed them, and they
# drifted about 4% away from what the script actually produces - the header
# claimed a primary test RMSE of 272.73 while the script was printing 284.65.
# Section 14 now MEASURES the trade-off on every run and writes it to
# auto_arima_covid_sensitivity.csv. Quote that file, not prose.
#
# So: the primary model below carries NO dummies and is accurate, and its
# Gaussian prediction intervals are correspondingly NOT trustworthy - a 26-sigma
# residual and kurtosis of ~500 violate the normality those intervals assume.
# Quote its point forecasts; do not quote its intervals. For calibrated intervals
# use the empirical error quantiles from rolling_cv.R.
#
# ARX-GARCH.R handles the 2020 break better in the mean equation, because in
# difference space the COVID event genuinely is four large spikes which four
# pulses can represent. Its INTERVALS are a separate matter - see that script.
# ============================================================

# 1. Load Packages
library(dplyr)
library(ggplot2)
library(forecast)

source("common.R")

MODEL_ID <- "auto_arima"

# 2. Read Processed Data and Split
data  <- load_series()
parts <- split_series(data)
describe_split(parts, "auto.arima (full history)")

train_ts     <- as_monthly_ts(parts$train)
train_val_ts <- as_monthly_ts(parts$train_val)
full_ts      <- as_monthly_ts(parts$all)

# 3. COVID regressors are built here but used ONLY by the sensitivity in
#    section 14. The primary specification does not carry them - see the header.
X_train_val <- covid_dummies(parts$train_val$date)
X_test      <- covid_dummies(parts$test$date)

# 4. Fit Auto ARIMA on the TRAINING block
# seasonal = TRUE leaves auto.arima free to choose seasonal orders. It selects a
# non-seasonal model, which is the expected result for a series that BLS has
# already seasonally adjusted.
auto_arima_model <- auto.arima(
  train_ts,
  seasonal      = TRUE,
  stepwise      = FALSE,
  approximation = FALSE
)

cat("\nSelected Auto ARIMA Model (fitted on train):\n")
print(auto_arima_model)

cat("\nIdentifiable (no NaN standard errors):",
    is_identifiable(auto_arima_model), "\n")

# 5. Score on the VALIDATION block
val_fc <- forecast(auto_arima_model, h = HORIZON, level = c(80, 95))
val_metrics <- evaluate(parts$val$value, val_fc$mean)

cat("\nValidation accuracy (2024-08 .. 2025-07):\n")
print(round(val_metrics, 4))

# 6. Refit on TRAIN + VALIDATION
# auto.arima re-runs its own order selection on the longer block. There is no
# reason to discard the validation year once it has done its job.
final_model <- auto.arima(
  train_val_ts,
  seasonal      = TRUE,
  stepwise      = FALSE,
  approximation = FALSE
)

cat("\n=== FINAL MODEL, REFITTED ON TRAIN + VALIDATION ===\n")
print(final_model)

# Does the reported Val_RMSE actually belong to the reported model?
# auto.arima runs its selection TWICE - once on train, once on train+validation -
# and nothing forces the two to agree. When they disagree, the Val_RMSE that goes
# into the comparison table was earned by a DIFFERENT specification from the one
# whose Test_RMSE sits beside it, and compare_models.R's "Val_RMSE shows a model
# that only looks good on test" reading no longer holds. Check it and say so.
order_train     <- arima_spec(auto_arima_model)
order_train_val <- arima_spec(final_model)
same_order      <- identical(order_train, order_train_val)

cat("\nOrder selected on train        :", as.character(auto_arima_model), "\n")
cat("Order selected on train+val    :", as.character(final_model), "\n")
if (same_order) {
  cat("The two selections AGREE, so Val_RMSE below belongs to the reported model.\n")
} else {
  cat("WARNING: the two selections DISAGREE. Val_RMSE was earned by the",
      "train-only\nspecification, not by the model whose test error is reported",
      "beside it.\nDo not read the two columns as describing one model.\n")
}

cat("\nModel Coefficients:\n")
print(coef(final_model))

identifiable <- is_identifiable(final_model)
cat("\nAIC :", AIC(final_model), "\n")
cat("BIC :", BIC(final_model), "\n")
cat("Identifiable (no NaN standard errors):", identifiable, "\n")

if (!identifiable) {
  warning("Final model has non-finite standard errors: its coefficients, ",
          "information criteria and prediction intervals are not usable.")
}

# 7. Residual Diagnostics
# Recorded through common.R::ljung_box() so the lag, the degrees of freedom and
# the residual space travel into the comparison table alongside the p-value. The
# table used to print one LjungBox_p column over three incomparable tests.
lb <- ljung_box(residuals(final_model), fitdf = arima_fitdf(final_model),
                on = "level residuals")

cat("\nLjung-Box test on residuals:\n")
cat(sprintf("  lag = %d, df = %d, n = %d, p = %.4f\n", lb$lag, lb$df, lb$n, lb$p))
cat("Verdict:",
    ifelse(lb$p > 0.05,
           "PASS - no significant residual autocorrelation",
           "FAIL - structure remains in the residuals"), "\n")

png("auto_arima_residual_diagnostics.png", width = 1400, height = 900, res = 150)
checkresiduals(final_model)
dev.off()

# 8. FINAL TEST EVALUATION - the test block is read only here
test_fc <- forecast(final_model, h = HORIZON, level = c(80, 95))
test_metrics <- evaluate(parts$test$value, test_fc$mean,
                         exclude = parts$test$imputed)

cat("\n=== FINAL TEST ACCURACY (2025-08 .. 2026-07) ===\n")
print(round(test_metrics, 4))

cat(sprintf("\nMASE uses the shared lag-1 denominator %.2f from common.R.\n",
            MASE_DENOM))
cat(sprintf("MASE_s uses the seasonal-naive denominator %.2f and is reported only\n",
            MASE_DENOM_S))
cat("alongside it - never on its own. See the note in common.R for why.\n")
cat(sprintf("Scored on %d observed months; 2025-10 is excluded because it is\n",
            unname(test_metrics["N"])))
cat(sprintf("interpolated, not observed. Including it would give RMSE %.2f.\n",
            unname(evaluate(parts$test$value, test_fc$mean)["RMSE"])))

# Against the shared benchmarks, on identical points.
bench <- benchmark_table()
cat("\nAgainst the shared benchmarks (same 11 scored months):\n")
print(round(bench[, c("RMSE", "MAE", "MASE")], 3))
cat(sprintf("Test RMSE vs RW-with-drift: %+.1f%%  (negative = model is better)\n",
            100 * (test_metrics["RMSE"] / bench["RW with drift", "RMSE"] - 1)))

# 9. Forecast Results
forecast_results <- data.frame(
  date     = parts$test$date,
  actual   = parts$test$value,
  forecast = as.numeric(test_fc$mean),
  lower_80 = as.numeric(test_fc$lower[, "80%"]),
  upper_80 = as.numeric(test_fc$upper[, "80%"]),
  lower_95 = as.numeric(test_fc$lower[, "95%"]),
  upper_95 = as.numeric(test_fc$upper[, "95%"]),
  imputed  = parts$test$imputed
) %>%
  mutate(
    error         = actual - forecast,
    abs_pct_error = abs(error / actual) * 100
  )

cat("\nForecast Results (imputed = actual is interpolated, not observed):\n")
print(forecast_results, row.names = FALSE, digits = 6)

write.csv(forecast_results, "auto_arima_test_forecast.csv", row.names = FALSE)

# 10. Save the shared result row
save_model_result(
  model_id     = MODEL_ID,
  model_name   = as.character(final_model),
  window_start = min(parts$all$date),
  n_train      = nrow(parts$train),
  n_train_val  = nrow(parts$train_val),
  val_metrics  = val_metrics,
  test_metrics = test_metrics,
  lb           = lb,
  identifiable = identifiable,
  aic          = AIC(final_model),
  bic          = BIC(final_model),
  spec         = order_train_val,
  notes        = paste0("primary: no COVID dummies; Gaussian intervals ",
                        "unreliable (see header)",
                        if (!same_order)
                          "; WARNING Val_RMSE is from a different order" else "")
)

# 11. Actual vs Forecast Plot
test_forecast_plot <- ggplot(forecast_results, aes(x = date)) +
  geom_ribbon(aes(ymin = lower_95, ymax = upper_95), alpha = 0.15) +
  geom_ribbon(aes(ymin = lower_80, ymax = upper_80), alpha = 0.25) +
  geom_line(aes(y = actual, linetype = "Actual"), linewidth = 0.9) +
  geom_line(aes(y = forecast, linetype = "Forecast"), linewidth = 0.9) +
  labs(
    title    = "Auto ARIMA Forecast of Female Employment",
    subtitle = "Actual vs Forecast - 12-Month Test Set (shaded intervals are NOT calibrated)",
    x = "Date",
    y = "Employment Level (Thousands)",
    linetype = ""
  ) +
  theme_minimal()

ggsave("auto_arima_test_forecast_plot.png", test_forecast_plot,
       width = 8, height = 5, dpi = 300)

# 12. Final Model Using Full Dataset, then a genuine 12-month forward forecast
# auto.arima selects a THIRD time here. If that selection differs from the one
# evaluated on the test block, the forward forecast does not come from the model
# whose accuracy was reported, and the report must not present it as though it
# did. SARIMA.R refits its selected orders instead of re-searching; the check
# below makes the difference visible rather than silent.
full_model <- auto.arima(
  full_ts,
  seasonal      = TRUE,
  stepwise      = FALSE,
  approximation = FALSE
)

cat("\nFull-sample Auto ARIMA Model:\n")
print(full_model)

if (!identical(arima_spec(full_model), order_train_val)) {
  cat("\nWARNING: the full-sample selection (", as.character(full_model),
      ") differs from\nthe evaluated model (", as.character(final_model),
      "). The 12-month forward forecast\nbelow therefore comes from a",
      "specification whose out-of-sample accuracy was\nnever measured. Report",
      "that, or refit the evaluated orders instead.\n", sep = "")
} else {
  cat("Full-sample selection matches the evaluated model.\n")
}

future_dates <- seq(
  from = month_add(max(data$date), 1),
  by = "month", length.out = HORIZON
)

final_forecast <- forecast(full_model, h = HORIZON, level = c(80, 95))

future_forecast_results <- data.frame(
  date     = future_dates,
  forecast = as.numeric(final_forecast$mean),
  lower_80 = as.numeric(final_forecast$lower[, "80%"]),
  upper_80 = as.numeric(final_forecast$upper[, "80%"]),
  lower_95 = as.numeric(final_forecast$lower[, "95%"]),
  upper_95 = as.numeric(final_forecast$upper[, "95%"])
)

cat("\nFuture 12-Month Forecast:\n")
print(future_forecast_results, row.names = FALSE, digits = 6)
cat("The intervals in this table are the model's own Gaussian ones and are NOT\n")
cat("trustworthy on this fit - use empirical_error_quantiles.csv instead.\n")

write.csv(future_forecast_results, "auto_arima_future_forecast.csv",
          row.names = FALSE)

# 13. Final Forecast Plot
history_data <- data.frame(date = data$date, actual = data$value) %>%
  filter(date >= as.Date("2020-01-01"))

forecast_line_data <- rbind(
  data.frame(date = max(history_data$date),
             forecast = tail(history_data$actual, 1)),
  future_forecast_results %>% select(date, forecast)
)

final_forecast_plot <- ggplot() +
  geom_line(data = history_data,
            aes(x = date, y = actual, colour = "Historical Data"),
            linewidth = 0.9) +
  geom_ribbon(data = future_forecast_results,
              aes(x = date, ymin = lower_95, ymax = upper_95),
              fill = "#90CAF9", alpha = 0.30) +
  geom_ribbon(data = future_forecast_results,
              aes(x = date, ymin = lower_80, ymax = upper_80),
              fill = "#42A5F5", alpha = 0.35) +
  geom_line(data = forecast_line_data,
            aes(x = date, y = forecast, colour = "Forecast"),
            linewidth = 1.2) +
  geom_point(data = future_forecast_results,
             aes(x = date, y = forecast, colour = "Forecast"), size = 1.8) +
  scale_colour_manual(values = c("Historical Data" = "#222222",
                                 "Forecast" = "#1565C0")) +
  labs(
    title    = "Auto ARIMA 12-Month Forecast of Female Employment",
    subtitle = "Shaded bands are the model's own Gaussian intervals - NOT calibrated",
    x = "Year",
    y = "Employment Level (Thousands)",
    colour = ""
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title       = element_text(size = 16, face = "bold"),
    plot.subtitle    = element_text(size = 11),
    legend.position  = "bottom",
    panel.grid.minor = element_blank()
  )

ggsave("auto_arima_future_forecast_plot.png", final_forecast_plot,
       width = 9, height = 5, dpi = 300)

# ==============================================================================
# 14. SENSITIVITY: WHAT COVID INTERVENTION DUMMIES COST AND BUY
#
# Not the reported model. This exists so the trade-off described in the header is
# MEASURED on every run rather than asserted from stale prose.
#
# TWO variants are fitted, because the earlier single-variant version could not
# support the conclusion it drew. It compared the primary ARIMA(2,1,2)+drift
# against a freely re-selected ARIMA(0,1,2) WITHOUT drift, and attributed the
# entire accuracy loss to the dummies - when that comparison also changed the AR
# and MA orders and dropped the drift term, which on a strongly trending series
# is by itself enough to wreck a 12-month forecast.
#
#   (a) CONTROLLED  same orders as the primary, dummies added. This is the one
#                   that isolates the effect of the dummies, and it is the number
#                   to quote for "what the dummies cost".
#   (b) RE-SELECTED auto.arima free to choose new orders given the dummies. This
#                   is the fairest version of the dummied ALTERNATIVE as a model,
#                   but it is not a controlled comparison.
#
# seasonal = FALSE in (b) on purpose: once the dummies absorb the COVID variance,
# auto.arima starts selecting seasonal terms on a series BLS has already
# seasonally adjusted.
# ==============================================================================
cat("\n\n=== SENSITIVITY: COVID DUMMIES (not the reported model) ===\n")

o <- arima_orders(final_model)
sens_ctrl <- Arima(train_val_ts,
                   order    = unname(o[c("p", "d", "q")]),
                   seasonal = list(order = unname(o[c("P", "D", "Q")]),
                                   period = FREQ),
                   xreg          = X_train_val,
                   include.drift = "drift" %in% names(coef(final_model)),
                   method        = "ML")

sens_free <- auto.arima(train_val_ts, xreg = X_train_val, seasonal = FALSE,
                        stepwise = FALSE, approximation = FALSE)

sens_metrics <- function(m) evaluate(parts$test$value,
                                     forecast(m, xreg = X_test)$mean,
                                     exclude = parts$test$imputed)
m_ctrl <- sens_metrics(sens_ctrl)
m_free <- sens_metrics(sens_free)

r_prim <- residual_summary(final_model)
r_ctrl <- residual_summary(sens_ctrl)
r_free <- residual_summary(sens_free)

comparison <- data.frame(
  Specification = c("Primary (no dummies)",
                    "Sensitivity A: same orders + dummies (CONTROLLED)",
                    "Sensitivity B: re-selected with dummies"),
  Model      = c(as.character(final_model), as.character(sens_ctrl),
                 as.character(sens_free)),
  Test_RMSE  = c(test_metrics["RMSE"], m_ctrl["RMSE"], m_free["RMSE"]),
  Test_MAE   = c(test_metrics["MAE"],  m_ctrl["MAE"],  m_free["MAE"]),
  Test_MASE  = c(test_metrics["MASE"], m_ctrl["MASE"], m_free["MASE"]),
  Max_abs_z  = c(r_prim["max_abs_z"], r_ctrl["max_abs_z"], r_free["max_abs_z"]),
  Kurtosis   = c(r_prim["kurtosis"],  r_ctrl["kurtosis"],  r_free["kurtosis"]),
  row.names  = NULL
)

print(comparison, row.names = FALSE, digits = 5)

cat("\nReading this table: row 2 is the honest cost of the dummies, because it is\n")
cat("the only row that changes ONE thing. Row 3 also changes the orders, so any\n")
cat("gap between rows 2 and 3 is an order effect, not a dummy effect.\n")
cat(sprintf("\nThe dummies cost %.1f RMSE (%+.1f%%) and buy a fall in max|z| from %.1f\n",
            m_ctrl["RMSE"] - test_metrics["RMSE"],
            100 * (m_ctrl["RMSE"] / test_metrics["RMSE"] - 1),
            r_prim["max_abs_z"]))
cat(sprintf("to %.1f and in excess kurtosis from %.0f to %.1f - which is what makes\n",
            r_ctrl["max_abs_z"], r_prim["kurtosis"], r_ctrl["kurtosis"]))
cat("Gaussian prediction intervals mean anything. Neither specification is both\n")
cat("accurate and well behaved; rolling_cv.R's empirical quantiles get both, by\n")
cat("keeping the accurate point forecast and dropping the Gaussian assumption.\n")

write.csv(comparison, "auto_arima_covid_sensitivity.csv", row.names = FALSE)

save_model_result(
  model_id     = paste0(MODEL_ID, "_covid_sensitivity"),
  model_name   = paste(as.character(sens_ctrl), "+ COVID dummies [controlled]"),
  window_start = min(parts$all$date),
  n_train      = nrow(parts$train),
  n_train_val  = nrow(parts$train_val),
  val_metrics  = setNames(rep(NA_real_, 3), c("RMSE", "MAE", "MAPE")),
  test_metrics = m_ctrl,
  lb           = ljung_box(residuals(sens_ctrl), fitdf = arima_fitdf(sens_ctrl),
                           on = "level residuals"),
  identifiable = is_identifiable(sens_ctrl),
  aic          = AIC(sens_ctrl),
  bic          = BIC(sens_ctrl),
  spec         = arima_spec(sens_ctrl),
  notes        = "SENSITIVITY ONLY - controlled (same orders as primary)"
)

cat("\nAuto ARIMA analysis completed.\n")
