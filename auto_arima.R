# ============================================================
# AUTO ARIMA MODEL
# SDG 5: Gender Equality
# Dataset: LNS12000002 - Employment Level: Women
# ============================================================

# 1. Load Packages
library(dplyr)
library(ggplot2)
library(forecast)

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

# 3. Create Time Series
female_ts <- ts(
  data$female_employment,
  start = c(
    as.numeric(format(min(data$date), "%Y")),
    as.numeric(format(min(data$date), "%m"))
  ),
  frequency = 12
)

# 4. Train-Test Split
# Last 12 months are used as test data
test_size <- 12

train_ts <- head(
  female_ts,
  length(female_ts) - test_size
)

test_ts <- tail(
  female_ts,
  test_size
)

# 5. Fit Auto ARIMA Model
auto_arima_model <- auto.arima(
  train_ts,
  seasonal = TRUE,
  stepwise = FALSE,
  approximation = FALSE
)

cat("\nSelected Auto ARIMA Model:\n")
print(auto_arima_model)

# 6. Model Information
cat("\nModel Coefficients:\n")
print(coef(auto_arima_model))

cat("\nAIC :", AIC(auto_arima_model), "\n")
cat("AICc:", auto_arima_model$aicc, "\n")
cat("BIC :", BIC(auto_arima_model), "\n")

# 7. Forecast Test Period
auto_arima_forecast <- forecast(
  auto_arima_model,
  h = test_size,
  level = c(80, 95)
)

cat("\n12-Month Test Forecast:\n")
print(auto_arima_forecast)

# 8. Forecast Accuracy
auto_arima_accuracy <- accuracy(
  auto_arima_forecast,
  test_ts
)

cat("\nForecast Accuracy:\n")
print(auto_arima_accuracy)

# 9. Residual Diagnostics
checkresiduals(
  auto_arima_model
)

# 10. Create Forecast Results
test_dates <- tail(
  data$date,
  test_size
)
forecast_results <- data.frame(
  date = test_dates,
  actual = as.numeric(test_ts),
  forecast = as.numeric(auto_arima_forecast$mean),
  lower_80 = as.numeric(auto_arima_forecast$lower[, "80%"]),
  upper_80 = as.numeric(auto_arima_forecast$upper[, "80%"]),
  lower_95 = as.numeric(auto_arima_forecast$lower[, "95%"]),
  upper_95 = as.numeric(auto_arima_forecast$upper[, "95%"])
)

cat("\nForecast Results:\n")
print(forecast_results)

write.csv(
  forecast_results,
  "auto_arima_test_forecast.csv",
  row.names = FALSE
)

# 11. Actual vs Forecast Plot
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
    title = "Auto ARIMA Forecast of Female Employment",
    subtitle = "Actual vs Forecast - 12-Month Test Set",
    x = "Date",
    y = "Employment Level (Thousands)",
    linetype = ""
  ) +
  
  theme_minimal()

print(test_forecast_plot)

ggsave(
  filename = "auto_arima_test_forecast_plot.png",
  plot = test_forecast_plot,
  width = 8,
  height = 5,
  dpi = 300
)

# 12. Final Model Using Full Dataset
final_auto_arima_model <- auto.arima(
  female_ts,
  seasonal = TRUE,
  stepwise = FALSE,
  approximation = FALSE
)

cat("\nFinal Auto ARIMA Model:\n")
print(final_auto_arima_model)

# 13. Future 12-Month Forecast
final_forecast <- forecast(
  final_auto_arima_model,
  h = 12,
  level = c(80, 95)
)

cat("\nFuture 12-Month Forecast:\n")
print(final_forecast)

# 14. Final Forecast Plot
# Historical data - recent years only
history_data <- data.frame(
  date = data$date,
  actual = data$female_employment
) %>%
  filter(date >= as.Date("2020-01-01"))


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
  date = future_dates,
  forecast = as.numeric(final_forecast$mean),
  lower_80 = as.numeric(final_forecast$lower[, "80%"]),
  upper_80 = as.numeric(final_forecast$upper[, "80%"]),
  lower_95 = as.numeric(final_forecast$lower[, "95%"]),
  upper_95 = as.numeric(final_forecast$upper[, "95%"])
)

cat("\nFuture Forecast Results:\n")
print(future_forecast_results)

write.csv(
  future_forecast_results,
  "auto_arima_future_forecast.csv",
  row.names = FALSE
)

# Prepare Future Forecast Data for Plot
future_data <- data.frame(
  date = future_dates,
  forecast = as.numeric(final_forecast$mean),
  lower_80 = as.numeric(final_forecast$lower[, "80%"]),
  upper_80 = as.numeric(final_forecast$upper[, "80%"]),
  lower_95 = as.numeric(final_forecast$lower[, "95%"]),
  upper_95 = as.numeric(final_forecast$upper[, "95%"])
)

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
    title = "Auto ARIMA 12-Month Forecast of Female Employment",
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
  filename = "auto_arima_future_forecast_plot.png",
  plot = final_forecast_plot,
  width = 9,
  height = 5,
  dpi = 300
)