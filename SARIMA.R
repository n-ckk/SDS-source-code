# ============================================================
# SARIMA - TEAMMATE-STYLE FINAL VERSION
# Train / Validation / Test
# Dataset: LNS12000002 - Employment Level: Women
# ============================================================

library(forecast)

# ============================================================
# 1. LOAD DATA
# ============================================================

data <- read.csv("processed_female_employment.csv")

data$date <- as.Date(data$date)
data$female_employment <- as.numeric(data$female_employment)
data <- data[order(data$date), ]

female_ts <- ts(
  data$female_employment,
  start = c(
    as.numeric(format(min(data$date), "%Y")),
    as.numeric(format(min(data$date), "%m"))
  ),
  frequency = 12
)

# ============================================================
# 2. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

validation_size <- 12
test_size <- 12
n <- length(female_ts)

train_ts <- head(
  female_ts,
  n - validation_size - test_size
)

validation_ts <- window(
  female_ts,
  start = time(female_ts)[n - validation_size - test_size + 1],
  end   = time(female_ts)[n - test_size]
)

test_ts <- tail(
  female_ts,
  test_size
)

train_validation_ts <- head(
  female_ts,
  n - test_size
)

cat("\n=== DATA SPLIT ===\n")
cat("Training   :", length(train_ts), "observations\n")
cat("Validation :", length(validation_ts), "observations\n")
cat("Test       :", length(test_ts), "observations\n")

# ============================================================
# 3. DETERMINE DIFFERENCING ORDERS
# ============================================================

d <- ndiffs(
  train_ts,
  test = "kpss",
  max.d = 2
)

D <- nsdiffs(
  train_ts,
  test = "seas",
  max.D = 1
)

cat("\n=== DIFFERENCING ===\n")
cat("d =", d, "\n")
cat("D =", D, "\n")

# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

calc_accuracy <- function(actual, forecast_values) {
  c(
    RMSE = sqrt(mean((actual - forecast_values)^2)),
    MAE  = mean(abs(actual - forecast_values)),
    MAPE = mean(abs((actual - forecast_values) / actual)) * 100
  )
}

get_ljung_p <- function(model, p, q, P, Q) {
  
  res <- residuals(model)
  res <- res[is.finite(res)]
  
  model_df <- p + q + P + Q
  
  lag_value <- min(
    24,
    floor(length(res) / 5)
  )
  
  lag_value <- max(
    lag_value,
    model_df + 3
  )
  
  lag_value <- min(
    lag_value,
    length(res) - 1
  )
  
  if (lag_value <= model_df) {
    return(NA_real_)
  }
  
  Box.test(
    res,
    lag = lag_value,
    type = "Ljung-Box",
    fitdf = model_df
  )$p.value
}

# ============================================================
# 5. SARIMA SPECIFICATION SEARCH
#    FIT ON TRAIN, SCORE ON VALIDATION
# ============================================================

candidate_table <- data.frame()
candidate_models <- list()
candidate_id <- 0

for (p in 0:4) {
  for (q in 0:4) {
    for (P in 0:2) {
      for (Q in 0:2) {
        
        # Keep at least one seasonal AR/MA term
        if ((P + Q) == 0) next
        
        fit <- tryCatch(
          Arima(
            train_ts,
            order = c(p, d, q),
            seasonal = list(
              order = c(P, D, Q),
              period = 12
            ),
            include.drift = (d == 1 && D == 0),
            method = "ML"
          ),
          error = function(e) NULL
        )
        
        if (is.null(fit)) next
        
        validation_fc <- tryCatch(
          forecast(
            fit,
            h = validation_size
          ),
          error = function(e) NULL
        )
        
        if (is.null(validation_fc)) next
        
        val_acc <- calc_accuracy(
          as.numeric(validation_ts),
          as.numeric(validation_fc$mean)
        )
        
        lb_p <- tryCatch(
          get_ljung_p(
            fit,
            p,
            q,
            P,
            Q
          ),
          error = function(e) NA_real_
        )
        
        candidate_id <- candidate_id + 1
        
        model_name <- sprintf(
          "SARIMA(%d,%d,%d)(%d,%d,%d)[12]",
          p, d, q, P, D, Q
        )
        
        candidate_models[[as.character(candidate_id)]] <- fit
        
        candidate_table <- rbind(
          candidate_table,
          data.frame(
            ID = candidate_id,
            Model = model_name,
            p = p,
            d = d,
            q = q,
            P = P,
            D = D,
            Q = Q,
            Validation_RMSE = val_acc["RMSE"],
            Validation_MAE = val_acc["MAE"],
            Validation_MAPE = val_acc["MAPE"],
            Ljung_Box_p = lb_p,
            AIC = fit$aic,
            AICc = fit$aicc,
            BIC = fit$bic,
            stringsAsFactors = FALSE
          )
        )
      }
    }
  }
}

if (nrow(candidate_table) == 0) {
  stop("No SARIMA candidate could be fitted.")
}

# ============================================================
# 6. MODEL SELECTION RULE
# ============================================================

# QUALIFY:
#   Ljung-Box p-value > 0.05
#
# RANK:
#   Lowest Validation RMSE
#   Then Validation MAE
#   Then Validation MAPE
#   Then AICc
#
# IMPORTANT:
#   Final TEST observations are not used in model selection.

qualified <- candidate_table[
  is.finite(candidate_table$Ljung_Box_p) &
    candidate_table$Ljung_Box_p > 0.05,
]

cat("\n=== MODEL SELECTION PROCESS ===\n")
cat("Candidates fitted :", nrow(candidate_table), "\n")
cat("Qualification     : Ljung-Box p-value > 0.05\n")
cat("Ranking           : Validation RMSE -> MAE -> MAPE -> AICc\n")

if (nrow(qualified) == 0) {
  
  cat("No model passed Ljung-Box qualification.\n")
  cat("Fallback: rank all fitted candidates by validation accuracy.\n")
  
  ranked <- candidate_table[
    order(
      candidate_table$Validation_RMSE,
      candidate_table$Validation_MAE,
      candidate_table$Validation_MAPE,
      candidate_table$AICc
    ),
  ]
  
} else {
  
  cat("Qualified models :", nrow(qualified), "\n")
  
  ranked <- qualified[
    order(
      qualified$Validation_RMSE,
      qualified$Validation_MAE,
      qualified$Validation_MAPE,
      qualified$AICc
    ),
  ]
}

selected <- ranked[1, ]
selected_model <- candidate_models[[as.character(selected$ID)]]

cat("\nTop 10 candidates used for final ranking:\n")

print(
  head(
    ranked[, c(
      "Model",
      "Validation_RMSE",
      "Validation_MAE",
      "Validation_MAPE",
      "Ljung_Box_p",
      "AICc"
    )],
    10
  ),
  row.names = FALSE,
  digits = 6
)

cat("\nSelected model:", selected$Model, "\n")
cat("Validation RMSE:", selected$Validation_RMSE, "\n")
cat("Validation MAE :", selected$Validation_MAE, "\n")
cat("Validation MAPE:", selected$Validation_MAPE, "%\n")
cat("Ljung-Box p    :", selected$Ljung_Box_p, "\n")

write.csv(
  candidate_table,
  "sarima_model_selection.csv",
  row.names = FALSE
)

# ============================================================
# 7. REFIT SELECTED MODEL ON TRAIN + VALIDATION
# ============================================================

final_model <- Arima(
  train_validation_ts,
  order = c(
    selected$p,
    selected$d,
    selected$q
  ),
  seasonal = list(
    order = c(
      selected$P,
      selected$D,
      selected$Q
    ),
    period = 12
  ),
  include.drift = (
    selected$d == 1 &&
      selected$D == 0
  ),
  method = "ML"
)

cat("\n=== FINAL MODEL: REFITTED ON TRAIN + VALIDATION ===\n")
print(final_model)

# ============================================================
# 8. FINAL RESIDUAL DIAGNOSTICS
# ============================================================

final_ljung_p <- get_ljung_p(
  final_model,
  selected$p,
  selected$q,
  selected$P,
  selected$Q
)

cat("\n=== FINAL RESIDUAL DIAGNOSTIC ===\n")
cat("Ljung-Box p-value:", final_ljung_p, "\n")

if (is.finite(final_ljung_p) && final_ljung_p > 0.05) {
  cat("Interpretation: no significant residual autocorrelation detected.\n")
} else {
  cat("Interpretation: residual autocorrelation may remain.\n")
}

png(
  "sarima_final_residual_diagnostics.png",
  width = 1400,
  height = 900,
  res = 150
)

checkresiduals(final_model)

dev.off()

# ============================================================
# 9. FINAL TEST EVALUATION
#    TEST IS USED ONLY HERE
# ============================================================

test_fc <- forecast(
  final_model,
  h = test_size,
  level = c(80, 95)
)

test_accuracy <- accuracy(
  test_fc,
  test_ts
)

train_acc <- test_accuracy["Training set", ]
test_acc <- test_accuracy["Test set", ]

rmse_ratio <- test_acc["RMSE"] / train_acc["RMSE"]
mae_ratio  <- test_acc["MAE"]  / train_acc["MAE"]
mape_ratio <- test_acc["MAPE"] / train_acc["MAPE"]

cat("\n=== FINAL TEST ACCURACY ===\n")
print(test_accuracy)

cat("\n=== TEST / TRAIN RATIOS ===\n")
cat("RMSE ratio :", rmse_ratio, "\n")
cat("MAE ratio  :", mae_ratio, "\n")
cat("MAPE ratio :", mape_ratio, "\n")

final_summary <- data.frame(
  Model = selected$Model,
  
  Validation_RMSE = selected$Validation_RMSE,
  Validation_MAE = selected$Validation_MAE,
  Validation_MAPE = selected$Validation_MAPE,
  
  Test_RMSE = unname(test_acc["RMSE"]),
  Test_MAE = unname(test_acc["MAE"]),
  Test_MAPE = unname(test_acc["MAPE"]),
  
  RMSE_Ratio = unname(rmse_ratio),
  MAE_Ratio = unname(mae_ratio),
  MAPE_Ratio = unname(mape_ratio),
  
  Ljung_Box_p = final_ljung_p,
  
  AIC = final_model$aic,
  AICc = final_model$aicc,
  BIC = final_model$bic,
  
  stringsAsFactors = FALSE
)

write.csv(
  final_summary,
  "sarima_final_summary.csv",
  row.names = FALSE
)

# ============================================================
# 10. TEST FORECAST TABLE + PLOT
# ============================================================

test_dates <- tail(
  data$date,
  test_size
)

test_forecast_table <- data.frame(
  Date = test_dates,
  Actual = as.numeric(test_ts),
  Forecast = as.numeric(test_fc$mean),
  Error = as.numeric(test_ts) -
    as.numeric(test_fc$mean),
  Lower_80 = as.numeric(test_fc$lower[, "80%"]),
  Upper_80 = as.numeric(test_fc$upper[, "80%"]),
  Lower_95 = as.numeric(test_fc$lower[, "95%"]),
  Upper_95 = as.numeric(test_fc$upper[, "95%"])
)

write.csv(
  test_forecast_table,
  "sarima_final_test_forecast.csv",
  row.names = FALSE
)

png(
  "sarima_final_test_forecast_plot.png",
  width = 1400,
  height = 900,
  res = 150
)

plot(
  test_fc,
  main = paste0(
    selected$Model,
    " - Final Test Forecast vs Actual"
  ),
  xlab = "Year",
  ylab = "Female Employment Level (Thousands)",
  include = 60
)

lines(
  test_ts,
  lwd = 2
)

dev.off()

# ============================================================
# 11. SAVE FINAL PARAMETERS
# ============================================================

parameter_table <- data.frame(
  Parameter = names(coef(final_model)),
  Estimate = as.numeric(coef(final_model))
)

write.csv(
  parameter_table,
  "sarima_final_parameters.csv",
  row.names = FALSE
)

# ============================================================
# 12. REFIT ON FULL DATA + FUTURE 12-MONTH FORECAST
# ============================================================

full_model <- Arima(
  female_ts,
  order = c(
    selected$p,
    selected$d,
    selected$q
  ),
  seasonal = list(
    order = c(
      selected$P,
      selected$D,
      selected$Q
    ),
    period = 12
  ),
  include.drift = (
    selected$d == 1 &&
      selected$D == 0
  ),
  method = "ML"
)

future_fc <- forecast(
  full_model,
  h = 12,
  level = c(80, 95)
)

future_dates <- seq(
  from = seq(
    max(data$date),
    by = "month",
    length.out = 2
  )[2],
  by = "month",
  length.out = 12
)

future_table <- data.frame(
  Date = future_dates,
  Forecast = as.numeric(future_fc$mean),
  Lower_80 = as.numeric(future_fc$lower[, "80%"]),
  Upper_80 = as.numeric(future_fc$upper[, "80%"]),
  Lower_95 = as.numeric(future_fc$lower[, "95%"]),
  Upper_95 = as.numeric(future_fc$upper[, "95%"])
)

write.csv(
  future_table,
  "sarima_future_12_month_forecast.csv",
  row.names = FALSE
)

png(
  "sarima_future_12_month_forecast.png",
  width = 1400,
  height = 900,
  res = 150
)

plot(
  future_fc,
  main = paste0(
    selected$Model,
    " - 12-Month Future Forecast"
  ),
  xlab = "Year",
  ylab = "Female Employment Level (Thousands)"
)

dev.off()

cat("\n=== COMPLETED ===\n")
cat("Selected model:", selected$Model, "\n")
cat("Results saved successfully.\n")
