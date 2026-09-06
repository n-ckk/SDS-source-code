# ==============================================================================
# compare_models.R - the group comparison table
#
# The repository previously had no comparison script at all, even though the
# comparison table is the actual deliverable. Each model was read off its own
# console output, and because those outputs used different MASE denominators the
# resulting ranking was wrong: on the scripts' own numbers Holt-Winters beat
# auto.arima, while on one denominator the order is the other way round.
#
# This script reads the one-row *_result.csv that each model writes through
# save_model_result() and puts them side by side. Every row is already on the
# shared split and the shared MASE denominator from common.R, so the columns are
# directly comparable.
#
# Run it after the four model scripts.
# ==============================================================================

source("common.R")

MODEL_FILES <- c(
  auto_arima   = "auto_arima_result.csv",
  sarima       = "sarima_result.csv",
  holt_winters = "holt_winters_result.csv",
  arx_garch    = "arx_garch_result.csv"
)

# Sensitivity rows are reported separately: they are diagnostics, not competing
# models, and mixing them into the ranking would imply they are candidates.
SENSITIVITY_FILES <- c(
  auto_arima_covid = "auto_arima_covid_sensitivity_result.csv",
  sarima_covid     = "sarima_covid_sensitivity_result.csv"
)

present <- MODEL_FILES[file.exists(MODEL_FILES)]
missing <- MODEL_FILES[!file.exists(MODEL_FILES)]

if (length(missing)) {
  cat("NOTE: no result file for:", paste(names(missing), collapse = ", "),
      "\n      Run the corresponding script(s) first.\n\n")
}
if (length(present) == 0) stop("No model results found. Run the model scripts first.")

results <- do.call(rbind, lapply(present, read.csv, stringsAsFactors = FALSE))
rownames(results) <- NULL

# ---- Benchmarks on the identical test window --------------------------------
# A model that cannot beat these is not earning its complexity.
data  <- load_series()
parts <- split_series(data)

actual   <- parts$test$value
last_obs <- tail(parts$train_val$value, 1)
tv       <- parts$train_val$value

bench <- rbind(
  `Naive (last value)` = evaluate(actual, rep(last_obs, HORIZON)),
  `Seasonal naive`     = evaluate(actual, tail(tv, HORIZON)),
  `RW with drift`      = evaluate(
    actual,
    last_obs + (last_obs - tv[1]) / (length(tv) - 1) * seq_len(HORIZON))
)

# ---- The table ---------------------------------------------------------------
main <- results[order(results$Test_RMSE),
                c("Model", "Window", "N_train_val", "Test_RMSE", "Test_MAE",
                  "Test_MAPE", "Test_MASE", "Val_RMSE", "LjungBox_p",
                  "Identifiable")]

cat("\n")
cat("================================================================\n")
cat(" MODEL COMPARISON - test window", format(TEST_START), "to",
    format(max(parts$test$date)), "\n")
cat("================================================================\n")
cat("All rows share one split and one MASE denominator (", round(MASE_DENOM, 2),
    ").\n", sep = "")
cat("Ranked by test RMSE. Val_RMSE is shown so a model that only looks good on\n")
cat("the test block is visible as such.\n\n")
print(main, row.names = FALSE, digits = 5)

cat("\n--- Benchmarks on the same 12 months ---\n")
print(round(bench, 4))

cat("\n--- Interpretation guide ---\n")
cat("* Test RMSE / MAE / MAPE are directly comparable: identical test window.\n")
cat("* Test_MASE is comparable ONLY because common.R fixes one denominator.\n")
cat("  Do NOT paste MASE values from the individual scripts' accuracy() output.\n")
cat("* AIC and BIC are NOT comparable across rows - the models are fitted on\n")
cat("  different windows and, for ARX-GARCH, on differences rather than levels.\n")
cat("  They are omitted from this table on purpose.\n")
cat("* Identifiable = FALSE means the fit has non-finite standard errors. Such a\n")
cat("  model must not be reported however good its test RMSE looks.\n")

if (any(!results$Identifiable, na.rm = TRUE)) {
  cat("\nWARNING: ",
      paste(results$Model[!results$Identifiable], collapse = ", "),
      "\n  returned non-finite standard errors and should not be reported.\n")
}

# How much does the best model actually buy over the simplest benchmark?
best <- main[1, ]
rw   <- bench["RW with drift", "RMSE"]
cat(sprintf("\nBest model (%s) vs random-walk-with-drift: %+.1f%% test RMSE.\n",
            best$Model, 100 * (best$Test_RMSE / rw - 1)))
cat("Report this. A few percent against a naive benchmark is the honest framing\n")
cat("for how much the modelling effort bought.\n")

# ---- Sensitivity rows --------------------------------------------------------
sens_present <- SENSITIVITY_FILES[file.exists(SENSITIVITY_FILES)]
if (length(sens_present)) {
  sens <- do.call(rbind, lapply(sens_present, read.csv, stringsAsFactors = FALSE))
  rownames(sens) <- NULL
  cat("\n--- COVID intervention sensitivity (NOT competing models) ---\n")
  cat("These refit a primary model WITH COVID dummies. They forecast worse but",
      "\nleave far better behaved residuals. See each script's header for the",
      "\nfull eight-configuration experiment behind that statement.\n")
  print(sens[, c("Model", "Test_RMSE", "Test_MAE", "Test_MAPE", "Test_MASE")],
        row.names = FALSE, digits = 5)
  write.csv(sens, "model_comparison_sensitivity.csv", row.names = FALSE)
}

write.csv(results[order(results$Test_RMSE), ], "model_comparison.csv",
          row.names = FALSE)
write.csv(data.frame(Benchmark = rownames(bench), bench, row.names = NULL),
          "model_comparison_benchmarks.csv", row.names = FALSE)

cat("\n--- IMPORTANT: this table rests on twelve observations ---\n")
cat("Models within a few percent of each other cannot be separated here, and the",
    "\nsingle block can flatter a model outright: on this block ARX-GARCH and",
    "\nHolt-Winters both beat the random-walk benchmark, but across 24 forecast",
    "\norigins (rolling_cv.R) both are WORSE than it. Run rolling_cv.R and quote",
    "\nrolling_cv_summary.csv as the primary ranking; this table is the strict",
    "\nread-once holdout, not the most reliable comparison.\n")

cat("\nAlso note: auto_arima.R and SARIMA.R select the SAME specification",
    "\n- ARIMA(2,1,2) with drift, identical AIC to ten significant figures.",
    "\nThey are one model found by two search procedures. Do not report them",
    "\nas independent results that corroborate one another.\n")

cat("\nWrote model_comparison.csv and model_comparison_benchmarks.csv\n")
