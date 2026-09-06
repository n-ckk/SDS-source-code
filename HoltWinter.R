# ============================================================
# HOLT-WINTERS ADDITIVE MODEL
# SDG 5: Gender Equality
# Dataset: LNS12000002 - Employment Level: Women
#
# Split, metrics and the MASE denominator come from common.R. Do not redefine
# them here - divergent copies of exactly those three things are what made the
# earlier comparison table misleading.
#
# WINDOW. The 2020 collapse is a structural break, not a trend. Holt-Winters has
# no mechanism for a level shift of that size and no way to carry an intervention
# regressor, so this model starts in January 2021 and simply avoids it. That is a
# legitimate choice, but it means this model is fitted on 55 months against the
# 943 available to the ARIMA models - which is why AIC, BIC and any per-model
# MASE are not comparable across scripts, and why common.R fixes one MASE
# denominator for everyone.
#
# SEASONALITY. LNS12000002 is seasonally adjusted at source, so the fitted gamma
# comes out at essentially zero and the seasonal indices are frozen rather than
# evolving. Reported below as a limitation, not hidden.
# ============================================================

# 1. Load Packages
library(dplyr)
library(ggplot2)
library(forecast)

source("common.R")

MODEL_ID     <- "holt_winters"
WINDOW_START <- as.Date("2021-01-01")

# The Metrics package exports its own accuracy(), rmse(), mae() and mape(). If it
# is attached in the same session it masks the forecast versions and this script
# fails with "'list' object cannot be coerced to type 'double'". Detach it if
# present. Restarting R also clears the problem.
if ("package:Metrics" %in% search()) {
  detach("package:Metrics", unload = TRUE)
  cat("Detached the Metrics package to avoid masking forecast::accuracy().\n")
}

# 2. Read Processed Data and Split
# NOTE: no calendar repair is done here any more. data_processing.R keeps every
# month on the grid and flags the ones it interpolated in the 'imputed' column,
# so common.R can report them directly. The previous version of this script
# rebuilt the calendar on the belief that data_processing.R dropped whole rows
# for missing values - it does not, it only drops rows with a missing DATE - and
# because the repair was a no-op it reported "imputed months in test set: 0" even
# though 2025-10 is interpolated.
data  <- load_series()
parts <- split_series(data, WINDOW_START)
describe_split(parts, "Holt-Winters (2021+ window)")

train_ts     <- as_monthly_ts(parts$train)
train_val_ts <- as_monthly_ts(parts$train_val)

# 3. Fit on TRAIN, score on VALIDATION
hw_train <- forecast::hw(train_ts, seasonal = "additive",
                         h = HORIZON, level = c(80, 95))

val_metrics <- evaluate(parts$val$value, hw_train$mean)

cat("\nValidation accuracy (2024-08 .. 2025-07):\n")
print(round(val_metrics, 4))

# 4. Refit on TRAIN + VALIDATION
# Holt-Winters has no order to select, so the validation block is a diagnostic
# here rather than a selection device. The model that forecasts the test block is
# refitted on everything up to the test start, exactly as the other scripts do.
hw_model <- forecast::hw(train_val_ts, seasonal = "additive",
                         h = HORIZON, level = c(80, 95))

cat("\n=== FINAL MODEL, REFITTED ON TRAIN + VALIDATION ===\n")
print(hw_model$model)

cat("\nSmoothing Parameters:\n")
print(round(hw_model$model$par[c("alpha", "beta", "gamma")], 6))

cat("\nAIC :", hw_model$model$aic, "\n")
cat("AICc:", hw_model$model$aicc, "\n")
cat("BIC :", hw_model$model$bic, "\n")
cat("NOTE: these are computed on", nrow(parts$train_val), "observations and are",
    "NOT comparable\n      with the ARIMA scripts' AIC/BIC.\n")

# Final seasonal indices, one per month
seasonal_columns <- grep("^s", colnames(hw_model$model$states))
seasonal_indices <- data.frame(
  month = month.abb,
  index = round(as.numeric(tail(hw_model$model$states[, seasonal_columns], 1)), 2)
)

cat("\nFinal Seasonal Indices (Thousands):\n")
print(seasonal_indices, row.names = FALSE)

# A gamma at or near zero means the seasonal indices are held constant rather
# than updated over time. LNS12000002 is seasonally adjusted at source, so this
# is the expected result and is reported as a limitation.
if (hw_model$model$par["gamma"] < 0.01) {
  cat("\nNote: gamma is effectively zero. The seasonal component is fixed rather",
      "\nthan evolving, consistent with a pre-adjusted series. Holt-Winters is",
      "\ntherefore operating as Holt's linear trend method with a frozen",
      "\nseasonal offset.\n")
}

# 5. Residual Diagnostics
# H0: residuals are independently distributed. The criterion is p > 0.05, so a
# PASS means we fail to reject H0.
#
# CAVEAT worth reporting: checkresiduals() applies df = 0 for an ETS object even
# though this model estimates 16 parameters, and it caps the test at floor(n/5)
# lags. On 55 observations that is 11 lags with no degrees-of-freedom correction,
# which makes this a lenient test. Treat a PASS here as weak evidence.
ljung_box <- checkresiduals(hw_model, plot = FALSE)

cat("\nLjung-Box Test:\n")
cat("Q statistic :", ljung_box$statistic, "\n")
cat("Degrees free:", ljung_box$parameter, "\n")
cat("p-value     :", ljung_box$p.value, "\n")
cat("Verdict     :",
    ifelse(ljung_box$p.value > 0.05,
           "PASS - residuals behave as white noise",
           "FAIL - structure remains in the residuals"), "\n")

png("holt_winters_residual_diagnostics.png",
    width = 1400, height = 900, res = 150)
checkresiduals(hw_model)
dev.off()

ljung_p <- unname(ljung_box$p.value)

# 6. FINAL TEST EVALUATION - the test block is read only here
test_metrics <- evaluate(parts$test$value, hw_model$mean)

cat("\n=== FINAL TEST ACCURACY (2025-08 .. 2026-07) ===\n")
print(round(test_metrics, 4))
cat(sprintf("\nMASE uses the shared denominator %.2f from common.R, NOT the\n", MASE_DENOM))
cat("per-model denominator forecast::accuracy() would compute from this\n")
cat("script's own 2021+ window.\n")

save_model_result(
  model_id     = MODEL_ID,
  model_name   = "Holt-Winters additive",
  window_start = WINDOW_START,
  n_train      = nrow(parts$train),
  n_train_val  = nrow(parts$train_val),
  val_metrics  = val_metrics,
  test_metrics = test_metrics,
  ljung_p      = ljung_p,
  identifiable = TRUE,
  aic          = hw_model$model$aic,
  bic          = hw_model$model$bic,
  notes        = "2021+ window avoids COVID; gamma ~ 0 (series pre-adjusted)"
)

# 7. Forecast Results
forecast_results <- data.frame(
  date     = parts$test$date,
  actual   = parts$test$value,
  forecast = as.numeric(hw_model$mean),
  lower_80 = as.numeric(hw_model$lower[, "80%"]),
  upper_80 = as.numeric(hw_model$upper[, "80%"]),
  lower_95 = as.numeric(hw_model$lower[, "95%"]),
  upper_95 = as.numeric(hw_model$upper[, "95%"]),
  imputed  = parts$test$imputed
) %>%
  mutate(
    error         = actual - forecast,
    abs_pct_error = abs(error / actual) * 100
  )

cat("\nForecast Results (imputed = actual is interpolated, not observed):\n")
print(forecast_results, row.names = FALSE, digits = 5)

write.csv(forecast_results, "holt_winters_test_forecast.csv", row.names = FALSE)

# 8. Actual vs Forecast Plot
test_forecast_plot <- ggplot(forecast_results, aes(x = date)) +
  geom_ribbon(aes(ymin = lower_95, ymax = upper_95), alpha = 0.15) +
  geom_ribbon(aes(ymin = lower_80, ymax = upper_80), alpha = 0.25) +
  geom_line(aes(y = actual, linetype = "Actual"), linewidth = 0.9) +
  geom_line(aes(y = forecast, linetype = "Forecast"), linewidth = 0.9) +
  labs(
    title    = "Holt-Winters Additive Forecast of Female Employment",
    subtitle = "Actual vs Forecast - 12-Month Test Set",
    x = "Date", y = "Employment Level (Thousands)", linetype = ""
  ) +
  theme_minimal()

ggsave("holt_winters_test_forecast_plot.png", test_forecast_plot,
       width = 8, height = 5, dpi = 300)

# 9. Seasonal Plot
seasonal_plot <- ggseasonplot(as_monthly_ts(parts$all), year.labels = TRUE) +
  labs(
    title    = "Seasonal Plot of Female Employment",
    subtitle = "Each line is one year",
    x = "Month", y = "Employment Level (Thousands)"
  ) +
  theme_minimal()

ggsave("holt_winters_seasonal_plot.png", seasonal_plot,
       width = 8, height = 5, dpi = 300)

# 10. Final Model Using the Full Window, then a genuine 12-month forward forecast
# The test months are included here because a real forward forecast has no reason
# to discard the most recent observations.
final_hw_model <- forecast::hw(as_monthly_ts(parts$all), seasonal = "additive",
                               h = HORIZON, level = c(80, 95))

cat("\nFull-window Holt-Winters Additive Model:\n")
print(final_hw_model$model)

future_dates <- seq(
  from = seq(max(data$date), by = "month", length.out = 2)[2],
  by = "month", length.out = HORIZON
)

future_forecast_results <- data.frame(
  date     = future_dates,
  forecast = as.numeric(final_hw_model$mean),
  lower_80 = as.numeric(final_hw_model$lower[, "80%"]),
  upper_80 = as.numeric(final_hw_model$upper[, "80%"]),
  lower_95 = as.numeric(final_hw_model$lower[, "95%"]),
  upper_95 = as.numeric(final_hw_model$upper[, "95%"])
)

cat("\nFuture 12-Month Forecast:\n")
print(future_forecast_results, row.names = FALSE, digits = 6)

write.csv(future_forecast_results, "holt_winters_future_forecast.csv",
          row.names = FALSE)

# 11. Final Forecast Plot
history_data <- data.frame(date = parts$all$date, actual = parts$all$value)

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
    title    = "Holt-Winters Additive 12-Month Forecast of Female Employment",
    subtitle = "Historical Data and Forecast with 80% and 95% Prediction Intervals",
    x = "Year", y = "Employment Level (Thousands)", colour = ""
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title       = element_text(size = 16, face = "bold"),
    plot.subtitle    = element_text(size = 11),
    legend.position  = "bottom",
    panel.grid.minor = element_blank()
  )

ggsave("holt_winters_future_forecast_plot.png", final_forecast_plot,
       width = 9, height = 5, dpi = 300)

cat("\nHolt-Winters analysis completed.\n")
