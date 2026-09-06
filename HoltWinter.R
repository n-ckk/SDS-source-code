# ============================================================
# HOLT-WINTERS ADDITIVE MODEL
# SDG 5: Gender Equality
# Dataset: LNS12000002 - Employment Level: Women
#
# Split, metrics, MASE denominators, benchmarks and the analysis window all come
# from common.R. Do not redefine them here - divergent copies of exactly those
# things are what made the earlier comparison table misleading.
#
# WINDOW. The 2020 collapse is a structural break, not a trend. Holt-Winters has
# no mechanism for a level shift of that size and no way to carry an intervention
# regressor, so this model starts at HW_WINDOW_START (2021-01) and simply avoids
# it. That is a legitimate choice, but it means this model is fitted on 55 months
# against the 943 available to the ARIMA models - which is why AIC, BIC and any
# per-model MASE are not comparable across scripts, and why common.R fixes one
# MASE denominator for everyone.
#
# SEASONALITY - AND WHY THE SEASONAL TERMS ARE PROBABLY DOING HARM.
# LNS12000002 is seasonally adjusted at source, so the fitted gamma comes out at
# about 1e-4: the seasonal indices are frozen at their initial values rather than
# evolving. That is not a harmless curiosity. Eleven free seasonal initial states
# are still estimated, on 43 training observations, and the resulting frozen
# offsets span roughly 450 thousand persons - larger than this model's own test
# RMSE. They are visible as a spurious step in the forward forecast. Section 10
# therefore fits ETS(A,A,N) - Holt's linear trend, the same model without the
# seasonal component - and reports both. If Holt wins, the seasonal terms were
# fitting noise and the report should say so.
# ============================================================

# 1. Load Packages
library(dplyr)
library(ggplot2)
library(forecast)

source("common.R")

MODEL_ID     <- "holt_winters"
WINDOW_START <- HW_WINDOW_START   # defined in common.R; rolling_cv.R reads it too

# The Metrics package exports its own accuracy(), rmse(), mae() and mape(). If it
# is attached in the same session it masks the forecast versions and this script
# fails with "'list' object cannot be coerced to type 'double'". Detach it if
# present. Restarting R also clears the problem.
if ("package:Metrics" %in% search()) {
  detach("package:Metrics", unload = TRUE)
  cat("Detached the Metrics package to avoid masking forecast::accuracy().\n")
}

# 2. Read Processed Data and Split
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

# Final seasonal indices, one per month.
#
# THE COLUMN ORDER IS NOT JANUARY-TO-DECEMBER. An ETS state row stores the
# seasonal vector in REVERSE chronological order: at the last row, column s1 is
# the index for the most recent observation, s2 the one before it, and so on back
# to s12. This script used to paste month.abb straight onto s1..s12, which
# labelled every index with the wrong month - with the series ending in July, s1
# is July and s12 is the following August, so the printed table was effectively
# reversed. Derive the months from the series end instead of assuming.
last_obs_date    <- max(parts$train_val$date)
seasonal_columns <- grep("^s[0-9]+$", colnames(hw_model$model$states))
seasonal_values  <- as.numeric(tail(hw_model$model$states[, seasonal_columns], 1))
seasonal_dates   <- seq(last_obs_date, by = "-1 month",
                        length.out = length(seasonal_values))

seasonal_indices <- data.frame(
  state = colnames(hw_model$model$states)[seasonal_columns],
  month = format(seasonal_dates, "%b"),
  index = round(seasonal_values, 2),
  stringsAsFactors = FALSE
)
# Present in calendar order, which is what a reader expects.
seasonal_indices <- seasonal_indices[order(match(seasonal_indices$month,
                                                 month.abb)), ]

cat("\nFinal Seasonal Indices (Thousands), in calendar order:\n")
cat("(the 'state' column shows which ETS state each index came from - s1 is the\n")
cat(" most recent month,", format(last_obs_date, "%b %Y"),
    ", not January)\n")
print(seasonal_indices, row.names = FALSE)
cat(sprintf("Range: %.1f to %.1f  (spread %.1f)\n",
            min(seasonal_values), max(seasonal_values),
            diff(range(seasonal_values))))

# A gamma at or near zero means the seasonal indices are held constant rather
# than updated over time.
if (hw_model$model$par["gamma"] < 0.01) {
  cat("\nNote: gamma is effectively zero. The seasonal component is FROZEN at its",
      "\nestimated initial values rather than evolving, consistent with a series",
      "\nthat BLS has already seasonally adjusted. Holt-Winters is operating as",
      "\nHolt's linear trend method plus a fixed offset pattern - and that pattern",
      "\nstill costs 11 estimated parameters and injects a spread of",
      sprintf("%.0f", diff(range(seasonal_values))),
      "\nthousand persons into the forecast. Section 10 tests whether it earns",
      "\nits place.\n")
}

# 5. Residual Diagnostics
# H0: residuals are independently distributed. The criterion is p > 0.05, so a
# PASS means we fail to reject H0.
#
# WHY THIS IS NOT JUST checkresiduals(). For an ETS object forecast::
# checkresiduals() applies df = 0, so no degrees of freedom are deducted for the
# parameters the model estimated - here 3 smoothing parameters plus 13 initial
# states. It also caps the test at floor(n/5) lags, which on this window is 11.
# The result is a lenient test whose PASS is weak evidence. There is no settled
# df correction for ETS, so both bounds are reported: df = 0 (the package
# default, most lenient) and df = 3 (deducting the smoothing parameters).
#
# The STRICTER of the two is what goes into the comparison table. The earlier
# version computed both, warned if they disagreed, and then saved the lenient
# one - discarding the work and putting the most flattering number in the shared
# table.
hw_resid <- residuals(hw_model)
hw_resid <- hw_resid[is.finite(hw_resid)]

lb_variants <- lapply(c(0, 3), function(k)
  ljung_box(hw_resid, fitdf = k, on = "level residuals"))

lb_table <- do.call(rbind, lapply(lb_variants, function(v)
  data.frame(fitdf = v$fitdf, lag = v$lag, chisq_df = v$df, p_value = v$p,
             verdict = ifelse(v$p > 0.05, "PASS", "FAIL"),
             row.names = NULL)))

cat("\nLjung-Box on Holt-Winters residuals\n")
cat("(df = 0 is the forecast package default for ETS; df = 3 deducts the",
    "\nsmoothing parameters. No settled correction exists - both are shown, and",
    "\nthe STRICTER one is what reaches model_comparison.csv.)\n")
print(lb_table, row.names = FALSE, digits = 5)

cat("\nParameters actually estimated: 3 smoothing +",
    length(hw_model$model$par) - 3, "initial states =",
    length(hw_model$model$par), "on", length(hw_resid), "observations.\n")
cat("This test has little power either way; treat a PASS as weak evidence.\n")

if (length(unique(lb_table$verdict)) > 1) {
  cat("\nWARNING: the two df conventions DISAGREE. Report both.\n")
}

write.csv(lb_table, "holt_winters_ljungbox.csv", row.names = FALSE)

png("holt_winters_residual_diagnostics.png",
    width = 1400, height = 900, res = 150)
checkresiduals(hw_model)
dev.off()

lb <- lb_variants[[which.min(vapply(lb_variants, function(v) v$p, 0))]]

# 6. FINAL TEST EVALUATION - the test block is read only here
test_metrics <- evaluate(parts$test$value, hw_model$mean,
                         exclude = parts$test$imputed)

cat("\n=== FINAL TEST ACCURACY (2025-08 .. 2026-07) ===\n")
print(round(test_metrics, 4))
cat(sprintf("\nMASE uses the shared lag-1 denominator %.2f from common.R, NOT the\n",
            MASE_DENOM))
cat("per-model denominator forecast::accuracy() would compute from this script's\n")
cat("own 2021+ window. MASE_s uses the seasonal-naive denominator",
    sprintf("%.2f", MASE_DENOM_S), "and\n")
cat("is reported alongside it, never on its own.\n")
cat(sprintf("Scored on %d observed months; 2025-10 is excluded because it is\n",
            unname(test_metrics["N"])))
cat(sprintf("interpolated, not observed. Including it would give RMSE %.2f.\n",
            unname(evaluate(parts$test$value, hw_model$mean)["RMSE"])))

bench <- benchmark_table()
cat("\nAgainst the shared benchmarks (same scored months):\n")
print(round(bench[, c("RMSE", "MAE", "MASE")], 3))
cat(sprintf("Test RMSE vs RW-with-drift: %+.1f%%  (negative = model is better)\n",
            100 * (test_metrics["RMSE"] / bench["RW with drift", "RMSE"] - 1)))

save_model_result(
  model_id     = MODEL_ID,
  model_name   = "Holt-Winters additive",
  window_start = WINDOW_START,
  n_train      = nrow(parts$train),
  n_train_val  = nrow(parts$train_val),
  val_metrics  = val_metrics,
  test_metrics = test_metrics,
  lb           = lb,
  identifiable = TRUE,
  aic          = hw_model$model$aic,
  bic          = hw_model$model$bic,
  spec         = "kind=hw;seasonal=additive",
  notes        = sprintf("2021+ window avoids COVID; gamma=%.1e (frozen seasonals, spread %.0f)",
                         hw_model$model$par["gamma"], diff(range(seasonal_values)))
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
    subtitle = "Each line is one year - on a series BLS has already deseasonalised",
    x = "Month", y = "Employment Level (Thousands)"
  ) +
  theme_minimal()

ggsave("holt_winters_seasonal_plot.png", seasonal_plot,
       width = 8, height = 5, dpi = 300)

# ==============================================================================
# 10. DO THE SEASONAL TERMS EARN THEIR PLACE?
#
# gamma is ~1e-4, so the seasonal component never updates - it is a fixed offset
# pattern estimated from 43 training observations on a series that BLS has
# already seasonally adjusted. The obvious alternative is the same model without
# it: ETS(A,A,N), Holt's linear trend, which drops 11 parameters.
#
# This is a diagnostic, not a competing entry in the comparison table. But if
# Holt is at least as accurate, then the seasonal component is fitting noise and
# the report must not present the seasonal indices as a finding about the series.
# ==============================================================================
cat("\n\n=== DOES THE SEASONAL COMPONENT EARN ITS PLACE? ===\n")

holt_model <- forecast::holt(train_val_ts, h = HORIZON, level = c(80, 95))
holt_metrics <- evaluate(parts$test$value, holt_model$mean,
                         exclude = parts$test$imputed)

holt_val <- evaluate(parts$val$value,
                     forecast::holt(train_ts, h = HORIZON)$mean)

seasonal_check <- data.frame(
  Model     = c("Holt-Winters additive (reported)", "ETS(A,A,N) - Holt, no seasonal"),
  N_par     = c(length(hw_model$model$par), length(holt_model$model$par)),
  Val_RMSE  = c(val_metrics["RMSE"],  holt_val["RMSE"]),
  Test_RMSE = c(test_metrics["RMSE"], holt_metrics["RMSE"]),
  Test_MAE  = c(test_metrics["MAE"],  holt_metrics["MAE"]),
  Test_MASE = c(test_metrics["MASE"], holt_metrics["MASE"]),
  AICc      = c(hw_model$model$aicc,  holt_model$model$aicc),
  row.names = NULL
)
print(seasonal_check, row.names = FALSE, digits = 5)

write.csv(seasonal_check, "holt_winters_seasonal_check.csv", row.names = FALSE)

if (holt_metrics["RMSE"] <= test_metrics["RMSE"]) {
  cat("\nFINDING: dropping the seasonal component is at least as accurate on the\n")
  cat(sprintf("test block (%.1f vs %.1f RMSE) while estimating %d fewer parameters.\n",
              holt_metrics["RMSE"], test_metrics["RMSE"],
              length(hw_model$model$par) - length(holt_model$model$par)))
  cat("The seasonal indices printed in section 4 are therefore fitting noise on a\n")
  cat("pre-adjusted series. Report them as an artefact, NOT as evidence of\n")
  cat("seasonality in female employment.\n")
} else {
  cat("\nFINDING: the seasonal component does improve test accuracy here\n")
  cat(sprintf("(%.1f vs %.1f RMSE). Given gamma ~ 0 and a pre-adjusted series this is\n",
              test_metrics["RMSE"], holt_metrics["RMSE"]))
  cat("more likely a fixed level correction than genuine seasonality - check\n")
  cat("whether the gap survives in rolling_cv.R before claiming otherwise.\n")
}

# 11. Final Model Using the Full Window, then a genuine 12-month forward forecast
# The test months are included here because a real forward forecast has no reason
# to discard the most recent observations.
final_hw_model <- forecast::hw(as_monthly_ts(parts$all), seasonal = "additive",
                               h = HORIZON, level = c(80, 95))

cat("\nFull-window Holt-Winters Additive Model:\n")
print(final_hw_model$model)

future_dates <- seq(
  from = month_add(max(data$date), 1),
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
cat("Any month-to-month step in this forecast is the FROZEN seasonal pattern,\n")
cat("not a prediction about that month - see section 10.\n")

write.csv(future_forecast_results, "holt_winters_future_forecast.csv",
          row.names = FALSE)

# 12. Final Forecast Plot
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
