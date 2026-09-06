# ============================================================
# HOLT-WINTERS ADDITIVE MODEL
# SDG 5: Gender Equality
# Dataset: LNS12000002 - Employment Level: Women
# ============================================================

# 1. Load Packages
library(dplyr)
library(ggplot2)
library(forecast)
library(zoo)

# The Metrics package exports its own accuracy(), rmse(), mae() and mape().
# If it is attached in the same session it masks the forecast versions and this
# script fails with "'list' object cannot be coerced to type 'double'". Detach
# it if present. Restarting R also clears the problem.
if ("package:Metrics" %in% search()) {
  detach("package:Metrics", unload = TRUE)
  cat("Detached the Metrics package to avoid masking forecast::accuracy().\n")
}

# 2. Read Processed Data
data <- read.csv(
  "processed_female_employment.csv"
)

data <- data %>%
  mutate(
    date = as.Date(date),
    female_employment = as.numeric(female_employment)
  ) %>%
  arrange(date)

# 3. Calendar Repair
# The processing script removes rows whose value is NA. For monthly data that
# deletes the MONTH, not just the value, so every later observation shifts one
# position earlier inside a frequency = 12 series. October 2025 is affected in
# the current release. Rebuild the full monthly grid and interpolate.
month_grid <- data.frame(
  date = seq(
    min(data$date),
    max(data$date),
    by = "month"
  )
)

data <- month_grid %>%
  left_join(
    data,
    by = "date"
  ) %>%
  mutate(
    imputed = is.na(female_employment),
    female_employment = na.approx(
      female_employment,
      x = date,
      na.rm = FALSE
    )
  )

cat("\nMonths reinserted:", sum(data$imputed), "\n")

if (any(data$imputed)) {
  cat(
    "Dates reinserted :",
    paste(format(data$date[data$imputed]), collapse = ", "),
    "\n"
  )
}

# 4. Apply Modelling Window
# The 2020 collapse is a structural break, not a trend. Holt-Winters cannot
# model a level shift of that size, so the window starts in January 2021.
window_start <- as.Date("2021-01-01")

data <- data %>%
  filter(date >= window_start)

cat(
  "\nWindow:",
  as.character(min(data$date)),
  "to",
  as.character(max(data$date)),
  "(", nrow(data), "months )\n"
)

# 5. Create Time Series
female_ts <- ts(
  data$female_employment,
  start = c(
    as.numeric(format(min(data$date), "%Y")),
    as.numeric(format(min(data$date), "%m"))
  ),
  frequency = 12
)

# 6. Train-Test Split
# Last 12 months are used as test data.
# subset() is used instead of head()/tail() because head() and tail() strip the
# ts class and the frequency attribute. Once frequency is lost, no seasonal
# model can be estimated correctly.
test_size <- 12

train_ts <- subset(
  female_ts,
  end = length(female_ts) - test_size
)

test_ts <- subset(
  female_ts,
  start = length(female_ts) - test_size + 1
)

cat("\nTraining observations:", length(train_ts), "\n")
cat("Testing observations :", length(test_ts), "\n")

test_imputed <- data$imputed[
  (nrow(data) - test_size + 1):nrow(data)
]

cat("Imputed months in test set:", sum(test_imputed), "\n")

# 7. Fit Holt-Winters Additive Model
hw_model <- forecast::hw(
  train_ts,
  seasonal = "additive",
  h = test_size,
  level = c(80, 95)
)

cat("\nHolt-Winters Additive Model:\n")
print(hw_model$model)

# 8. Model Information
cat("\nSmoothing Parameters:\n")
print(round(hw_model$model$par[c("alpha", "beta", "gamma")], 6))

cat("\nAIC :", hw_model$model$aic, "\n")
cat("AICc:", hw_model$model$aicc, "\n")
cat("BIC :", hw_model$model$bic, "\n")

# Final seasonal indices, one per month
seasonal_columns <- grep(
  "^s",
  colnames(hw_model$model$states)
)

seasonal_indices <- data.frame(
  month = month.abb,
  index = round(
    as.numeric(
      tail(hw_model$model$states[, seasonal_columns], 1)
    ),
    2
  )
)

cat("\nFinal Seasonal Indices (Thousands):\n")
print(seasonal_indices, row.names = FALSE)

# A gamma at or near zero means the seasonal indices are held constant rather
# than updated over time. LNS12000002 is seasonally adjusted at source, so this
# is the expected result and should be reported as a limitation.
if (hw_model$model$par["gamma"] < 0.01) {
  cat(
    "\nNote: gamma is effectively zero. The seasonal component is fixed",
    "\nrather than evolving, consistent with a pre-adjusted series.\n"
  )
}

# 9. Forecast Test Period
cat("\n12-Month Test Forecast:\n")
print(hw_model)

# 10. Forecast Accuracy
# forecast:: is written out in full because the Metrics package also exports an
# accuracy() function, with the signature accuracy(actual, predicted). If
# Metrics is attached later than forecast, it masks this call and the forecast
# object gets passed where a numeric vector is expected, producing:
#   Error: 'list' object cannot be coerced to type 'double'
hw_accuracy <- forecast::accuracy(
  hw_model,
  test_ts
)

cat("\nForecast Accuracy:\n")
print(hw_accuracy)

# MASE is scale-free and is the correct measure for comparing across models,
# but only when every model uses the SAME training window. Use the Test set
# row, not the Training set row.
cat("\nMASE (Training set):", hw_accuracy["Training set", "MASE"], "\n")
cat("MASE (Test set)    :", hw_accuracy["Test set", "MASE"], "\n")

# Overfitting check: test metric divided by training metric.
# The group criterion is a ratio of 1.3 or lower on every metric.
ratio_rmse <- hw_accuracy["Test set", "RMSE"] / hw_accuracy["Training set", "RMSE"]
ratio_mae  <- hw_accuracy["Test set", "MAE"]  / hw_accuracy["Training set", "MAE"]
ratio_mape <- hw_accuracy["Test set", "MAPE"] / hw_accuracy["Training set", "MAPE"]

overfitting_check <- data.frame(
  metric  = c("RMSE", "MAE", "MAPE"),
  train   = c(
    hw_accuracy["Training set", "RMSE"],
    hw_accuracy["Training set", "MAE"],
    hw_accuracy["Training set", "MAPE"]
  ),
  test    = c(
    hw_accuracy["Test set", "RMSE"],
    hw_accuracy["Test set", "MAE"],
    hw_accuracy["Test set", "MAPE"]
  ),
  ratio   = c(ratio_rmse, ratio_mae, ratio_mape),
  verdict = ifelse(
    c(ratio_rmse, ratio_mae, ratio_mape) <= 1.3,
    "PASS",
    "FAIL"
  )
)

cat("\nOverfitting Check (criterion: ratio <= 1.3):\n")
print(overfitting_check, row.names = FALSE, digits = 5)

write.csv(
  overfitting_check,
  "holt_winters_accuracy.csv",
  row.names = FALSE
)

# 11. Residual Diagnostics
# H0: residuals are independently distributed (no autocorrelation).
# The criterion is p > 0.05, so a PASS means we fail to reject H0.
ljung_box <- checkresiduals(
  hw_model
)

cat("\nLjung-Box Test:\n")
cat("Q statistic :", ljung_box$statistic, "\n")
cat("Degrees free:", ljung_box$parameter, "\n")
cat("p-value     :", ljung_box$p.value, "\n")
cat(
  "Verdict     :",
  ifelse(
    ljung_box$p.value > 0.05,
    "PASS - residuals behave as white noise",
    "FAIL - structure remains in the residuals"
  ),
  "\n"
)

ljung_box_results <- data.frame(
  test      = "Ljung-Box",
  Q         = as.numeric(ljung_box$statistic),
  df        = as.numeric(ljung_box$parameter),
  p_value   = as.numeric(ljung_box$p.value),
  threshold = 0.05,
  verdict   = ifelse(ljung_box$p.value > 0.05, "PASS", "FAIL")
)

write.csv(
  ljung_box_results,
  "holt_winters_ljungbox.csv",
  row.names = FALSE
)

# 12. Create Forecast Results
test_dates <- tail(
  data$date,
  test_size
)

forecast_results <- data.frame(
  date     = test_dates,
  actual   = as.numeric(test_ts),
  forecast = as.numeric(hw_model$mean),
  lower_80 = as.numeric(hw_model$lower[, "80%"]),
  upper_80 = as.numeric(hw_model$upper[, "80%"]),
  lower_95 = as.numeric(hw_model$lower[, "95%"]),
  upper_95 = as.numeric(hw_model$upper[, "95%"]),
  imputed  = test_imputed
) %>%
  mutate(
    error         = actual - forecast,
    abs_pct_error = abs(error / actual) * 100
  )

cat("\nForecast Results:\n")
print(forecast_results, row.names = FALSE, digits = 5)

write.csv(
  forecast_results,
  "holt_winters_test_forecast.csv",
  row.names = FALSE
)

# 13. Actual vs Forecast Plot
test_forecast_plot <- ggplot(
  forecast_results,
  aes(x = date)
) +
  
  geom_ribbon(
    aes(
      ymin = lower_95,
      ymax = upper_95
    ),
    alpha = 0.15
  ) +
  
  geom_ribbon(
    aes(
      ymin = lower_80,
      ymax = upper_80
    ),
    alpha = 0.25
  ) +
  
  geom_line(
    aes(
      y = actual,
      linetype = "Actual"
    ),
    linewidth = 0.9
  ) +
  
  geom_line(
    aes(
      y = forecast,
      linetype = "Forecast"
    ),
    linewidth = 0.9
  ) +
  
  labs(
    title = "Holt-Winters Additive Forecast of Female Employment",
    subtitle = "Actual vs Forecast - 12-Month Test Set",
    x = "Date",
    y = "Employment Level (Thousands)",
    linetype = ""
  ) +
  
  theme_minimal()

print(test_forecast_plot)

ggsave(
  filename = "holt_winters_test_forecast_plot.png",
  plot = test_forecast_plot,
  width = 8,
  height = 5,
  dpi = 300
)

# 14. Seasonal Plot
seasonal_plot <- ggseasonplot(
  female_ts,
  year.labels = TRUE
) +
  
  labs(
    title = "Seasonal Plot of Female Employment",
    subtitle = "Each line is one year",
    x = "Month",
    y = "Employment Level (Thousands)"
  ) +
  
  theme_minimal()

print(seasonal_plot)

ggsave(
  filename = "holt_winters_seasonal_plot.png",
  plot = seasonal_plot,
  width = 8,
  height = 5,
  dpi = 300
)

# 15. Final Model Using Full Dataset
# The test months are included here because a genuine forward forecast has no
# reason to discard the most recent observations.
final_hw_model <- forecast::hw(
  female_ts,
  seasonal = "additive",
  h = 12,
  level = c(80, 95)
)

cat("\nFinal Holt-Winters Additive Model:\n")
print(final_hw_model$model)

# 16. Future 12-Month Forecast
cat("\nFuture 12-Month Forecast:\n")
print(final_hw_model)

# Future dates
future_dates <- seq.Date(
  from = seq.Date(
    max(data$date),
    by = "month",
    length.out = 2
  )[2],
  by = "month",
  length.out = 12
)

# Future Forecast Results
future_forecast_results <- data.frame(
  date     = future_dates,
  forecast = as.numeric(final_hw_model$mean),
  lower_80 = as.numeric(final_hw_model$lower[, "80%"]),
  upper_80 = as.numeric(final_hw_model$upper[, "80%"]),
  lower_95 = as.numeric(final_hw_model$lower[, "95%"]),
  upper_95 = as.numeric(final_hw_model$upper[, "95%"])
)

cat("\nFuture Forecast Results:\n")
print(future_forecast_results, row.names = FALSE, digits = 6)

write.csv(
  future_forecast_results,
  "holt_winters_future_forecast.csv",
  row.names = FALSE
)

# 17. Final Forecast Plot
# Historical data - recent years only
history_data <- data.frame(
  date   = data$date,
  actual = data$female_employment
)

future_data <- future_forecast_results

# Connect forecast line to the last historical observation
forecast_line_data <- rbind(
  data.frame(
    date = max(history_data$date),
    forecast = tail(history_data$actual, 1)
  ),
  
  future_data %>%
    select(
      date,
      forecast
    )
)

# Publication-style plot
final_forecast_plot <- ggplot() +
  
  geom_line(
    data = history_data,
    aes(
      x = date,
      y = actual,
      colour = "Historical Data"
    ),
    linewidth = 0.9
  ) +
  
  geom_ribbon(
    data = future_data,
    aes(
      x = date,
      ymin = lower_95,
      ymax = upper_95
    ),
    fill = "#90CAF9",
    alpha = 0.30
  ) +
  
  geom_ribbon(
    data = future_data,
    aes(
      x = date,
      ymin = lower_80,
      ymax = upper_80
    ),
    fill = "#42A5F5",
    alpha = 0.35
  ) +
  
  geom_line(
    data = forecast_line_data,
    aes(
      x = date,
      y = forecast,
      colour = "Forecast"
    ),
    linewidth = 1.2
  ) +
  
  geom_point(
    data = future_data,
    aes(
      x = date,
      y = forecast,
      colour = "Forecast"
    ),
    size = 1.8
  ) +
  
  scale_colour_manual(
    values = c(
      "Historical Data" = "#222222",
      "Forecast" = "#1565C0"
    )
  ) +
  
  labs(
    title = "Holt-Winters Additive 12-Month Forecast of Female Employment",
    subtitle = "Historical Data and Forecast with 80% and 95% Prediction Intervals",
    x = "Year",
    y = "Employment Level (Thousands)",
    colour = ""
  ) +
  
  theme_minimal(base_size = 12) +
  
  theme(
    plot.title = element_text(
      size = 16,
      face = "bold"
    ),
    
    plot.subtitle = element_text(
      size = 11
    ),
    
    axis.title = element_text(
      size = 12
    ),
    
    axis.text = element_text(
      size = 10
    ),
    
    legend.position = "bottom",
    
    panel.grid.minor = element_blank()
  )

print(final_forecast_plot)

ggsave(
  filename = "holt_winters_future_forecast_plot.png",
  plot = final_forecast_plot,
  width = 9,
  height = 5,
  dpi = 300
)

cat("\nHolt-Winters analysis completed successfully!\n")
