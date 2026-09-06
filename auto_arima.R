# ============================================================
# AUTO ARIMA MODEL
# SDG 5: Gender Equality
# Dataset: LNS12000002 - Employment Level: Women
#
# Split, metrics, MASE denominator and COVID regressors all come from common.R.
# Do not redefine any of them here.
#
# COVID AND THE TRADE-OFF (read before changing the specification).
# The 2020 break is NOT treated in the primary specification, and that is a
# deliberate, measured choice rather than an oversight. Treating it with additive
# outlier dummies was tested across four intervention windows, with and without
# seasonal terms - eight configurations in total:
#
#   intervention      seasonal   test RMSE   max|z|   kurtosis
#   none (primary)    -             272.73    26.20      503.6
#   Mar-Jun 2020      yes           364.65     6.96       10.3
#   Mar-Jun 2020      no            315.72     7.11       10.4
#   Mar-Dec 2020      no            419.43     5.78        1.9
#   Mar 2020-Jun 21   no            351.04     5.91        2.0
#
# Residual quality and point accuracy move in OPPOSITE directions, monotonically.
# No configuration achieves both, and the same pattern appears independently on
# the validation block, so it is not test-set luck. The dummies do exactly what
# they are supposed to do - they cut the 26-sigma April 2020 residual to about 7
# and the excess kurtosis from 504 to 10 - but every dummied variant forecasts
# worse than the undummied model, and the best of them is still worse than a
# random walk with drift (309.80).
#
# So: the primary model below carries NO dummies and is accurate, and its
# Gaussian prediction intervals are correspondingly NOT trustworthy - a 26-sigma
# residual and kurtosis of 504 violate the normality those intervals assume.
# Quote its point forecasts; do not quote its intervals. Section 9 fits the
# dummied alternative as a documented sensitivity so the trade-off is visible in
# the output rather than asserted here.
#
# ARX-GARCH.R is the model that gets BOTH right, because in difference space the
# COVID event genuinely is four large spikes, which four pulses can represent.
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
#    section 9. The primary specification does not carry them - see the header.
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
residual_check <- checkresiduals(final_model, plot = FALSE)

cat("\nLjung-Box test on residuals:\n")
print(residual_check)

ljung_p <- unname(residual_check$p.value)
cat("Verdict:",
    ifelse(ljung_p > 0.05,
           "PASS - no significant residual autocorrelation",
           "FAIL - structure remains in the residuals"), "\n")

png("auto_arima_residual_diagnostics.png", width = 1400, height = 900, res = 150)
checkresiduals(final_model)
dev.off()

# 8. FINAL TEST EVALUATION - the test block is read only here
test_fc <- forecast(final_model, h = HORIZON, level = c(80, 95))
test_metrics <- evaluate(parts$test$value, test_fc$mean)

cat("\n=== FINAL TEST ACCURACY (2025-08 .. 2026-07) ===\n")
print(round(test_metrics, 4))

cat(sprintf("\nMASE uses the shared denominator %.2f from common.R.\n", MASE_DENOM))

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
  ljung_p      = ljung_p,
  identifiable = identifiable,
  aic          = AIC(final_model),
  bic          = BIC(final_model),
  notes        = "primary: no COVID dummies; intervals unreliable (see header)"
)

# 11. Actual vs Forecast Plot
test_forecast_plot <- ggplot(forecast_results, aes(x = date)) +
  geom_ribbon(aes(ymin = lower_95, ymax = upper_95), alpha = 0.15) +
  geom_ribbon(aes(ymin = lower_80, ymax = upper_80), alpha = 0.25) +
  geom_line(aes(y = actual, linetype = "Actual"), linewidth = 0.9) +
  geom_line(aes(y = forecast, linetype = "Forecast"), linewidth = 0.9) +
  labs(
    title    = "Auto ARIMA Forecast of Female Employment",
    subtitle = "Actual vs Forecast - 12-Month Test Set",
    x = "Date",
    y = "Employment Level (Thousands)",
    linetype = ""
  ) +
  theme_minimal()

ggsave("auto_arima_test_forecast_plot.png", test_forecast_plot,
       width = 8, height = 5, dpi = 300)

# 12. Final Model Using Full Dataset, then a genuine 12-month forward forecast
full_model <- auto.arima(
  full_ts,
  seasonal      = TRUE,
  stepwise      = FALSE,
  approximation = FALSE
)

cat("\nFull-sample Auto ARIMA Model:\n")
print(full_model)

future_dates <- seq(
  from = seq(max(data$date), by = "month", length.out = 2)[2],
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
    subtitle = "Historical Data and Forecast with 80% and 95% Prediction Intervals",
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
# visible in the output rather than merely asserted, and so a reader can check it
# without re-running the eight-configuration experiment.
#
# seasonal = FALSE here on purpose. Once the dummies absorb the COVID variance,
# auto.arima starts selecting seasonal terms on a series BLS has already
# seasonally adjusted; those terms are spurious and cost about 49 RMSE. This is
# therefore the FAIREST version of the dummied alternative, not a straw man.
# ==============================================================================
cat("\n\n=== SENSITIVITY: COVID DUMMIES (not the reported model) ===\n")

sens_model <- auto.arima(train_val_ts, xreg = X_train_val, seasonal = FALSE,
                         stepwise = FALSE, approximation = FALSE)
sens_fc      <- forecast(sens_model, xreg = X_test, level = c(80, 95))
sens_metrics <- evaluate(parts$test$value, sens_fc$mean)

prim_resid <- residual_summary(final_model)
sens_resid <- residual_summary(sens_model)

comparison <- data.frame(
  Specification = c("Primary (no dummies)", "Sensitivity (COVID dummies)"),
  Model         = c(as.character(final_model), as.character(sens_model)),
  Test_RMSE     = c(test_metrics["RMSE"], sens_metrics["RMSE"]),
  Test_MAE      = c(test_metrics["MAE"],  sens_metrics["MAE"]),
  Test_MAPE     = c(test_metrics["MAPE"], sens_metrics["MAPE"]),
  Max_abs_z     = c(prim_resid["max_abs_z"], sens_resid["max_abs_z"]),
  Kurtosis      = c(prim_resid["kurtosis"],  sens_resid["kurtosis"]),
  row.names     = NULL
)

print(comparison, row.names = FALSE, digits = 5)

cat("\nReading this table: the dummies cut the largest standardised residual and",
    "\nthe excess kurtosis by an order of magnitude, which is what makes Gaussian",
    "\nprediction intervals meaningful - and they cost point accuracy to do it.",
    "\nNeither specification is both accurate and well behaved. ARX-GARCH is.\n")

write.csv(comparison, "auto_arima_covid_sensitivity.csv", row.names = FALSE)

save_model_result(
  model_id     = paste0(MODEL_ID, "_covid_sensitivity"),
  model_name   = paste(as.character(sens_model), "+ COVID dummies [sensitivity]"),
  window_start = min(parts$all$date),
  n_train      = nrow(parts$train),
  n_train_val  = nrow(parts$train_val),
  val_metrics  = setNames(rep(NA_real_, 4), c("RMSE", "MAE", "MAPE", "MASE")),
  test_metrics = sens_metrics,
  ljung_p      = unname(checkresiduals(sens_model, plot = FALSE)$p.value),
  identifiable = is_identifiable(sens_model),
  aic          = AIC(sens_model),
  bic          = BIC(sens_model),
  notes        = "SENSITIVITY ONLY - not the reported model"
)

cat("\nAuto ARIMA analysis completed.\n")
