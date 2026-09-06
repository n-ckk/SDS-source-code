# ==============================================================================
# compare_models.R - the group comparison table
#
# This script reads the one-row *_result.csv that each model writes through
# save_model_result() and puts them side by side. Every row is already on the
# shared split, the shared MASE denominators and the shared benchmarks from
# common.R, so the columns are directly comparable - with the two documented
# exceptions called out in the interpretation guide below.
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

rows <- lapply(present, read.csv, stringsAsFactors = FALSE)

# rbind() on frames with different columns fails with "numbers of columns of
# arguments do not match", which says nothing about the cause. The cause is
# always the same: one model's *_result.csv was written by an older version of
# save_model_result() than the others, so the schemas differ. Say that.
schemas <- lapply(rows, names)
if (length(unique(lapply(schemas, sort))) > 1) {
  ref <- schemas[[1]]
  for (i in seq_along(rows)) {
    extra   <- setdiff(schemas[[i]], ref)
    missing <- setdiff(ref, schemas[[i]])
    if (length(extra) || length(missing)) {
      cat(present[i], ": ",
          if (length(missing)) paste("missing", paste(missing, collapse = ", ")) else "",
          if (length(extra)) paste("  extra", paste(extra, collapse = ", ")) else "",
          "
", sep = "")
    }
  }
  stop("Result files were written by different versions of save_model_result(). ",
       "Re-run every model script (or `Rscript run_all.R`) so they share one schema.")
}

results <- do.call(rbind, rows)
rownames(results) <- NULL

# ---- Benchmarks on the identical test window --------------------------------
# From common.R, so this is the same "RW with drift" the model scripts and
# rolling_cv.R use. It used to be built here by hand and separately in
# ARX-GARCH.R with a different drift window and a different set of scored
# months, which made one claim come out as two different numbers.
data  <- load_series()
parts <- split_series(data)
drop  <- parts$test$imputed
bench <- benchmark_table()

# ---- Are any two rows the same model? ---------------------------------------
# auto_arima.R and SARIMA.R search different spaces but converge on the same
# specification. When they do, the table has four rows and three models, and
# "the best model" is decided by an optimizer-tolerance gap between two copies of
# one fit. The Spec column makes this checkable instead of being a remark at the
# bottom of the output.
dupes <- list()
if (!is.null(results$Spec)) {
  s <- results$Spec[!is.na(results$Spec)]
  for (v in unique(s[duplicated(s)])) {
    dupes[[v]] <- results$Model[which(results$Spec == v)]
  }
}

# ---- The table ---------------------------------------------------------------
cols <- c("Model", "Window", "N_train_val", "Test_RMSE", "Test_MAE",
          "Test_MAPE", "Test_MASE", "Test_MASE_s", "Val_RMSE",
          "LjungBox_p", "LjungBox_lag", "LjungBox_fitdf", "Identifiable")
cols <- cols[cols %in% names(results)]
main <- results[order(results$Test_RMSE), cols]

cat("\n")
cat("================================================================\n")
cat(" MODEL COMPARISON - test window", format(TEST_START), "to",
    format(max(parts$test$date)), "\n")
cat("================================================================\n")
cat("All rows share one split and one pair of MASE denominators.\n")
cat("  MASE   scales by the lag-1 naive MAE   (", round(MASE_DENOM, 2),
    ") <- the headline\n", sep = "")
cat("  MASE_s scales by the seasonal-naive MAE (", round(MASE_DENOM_S, 2),
    ")\n", sep = "")
cat("Scored on ", sum(!drop), " of ", HORIZON,
    " test months: 2025-10 is interpolated, not observed,\n",
    "and is excluded from every model AND every benchmark.\n", sep = "")
cat("Ranked by test RMSE. Val_RMSE is shown so a model that only looks good on\n")
cat("the test block is visible as such.\n\n")
print(main, row.names = FALSE, digits = 5)

cat("\n--- Benchmarks on the same months ---\n")
print(round(bench, 4))

cat("\n--- Interpretation guide ---\n")
cat("* Test RMSE / MAE / MAPE are directly comparable: identical test window.\n")
cat("* Test_MASE is comparable ONLY because common.R fixes one denominator.\n")
cat("  Do NOT paste MASE values from the individual scripts' accuracy() output.\n")
cat("* MASE ABOVE 1 MEANS WORSE THAN A NAIVE ONE-STEP FORECAST on the scaling\n")
cat("  that fits this series. MASE_s is on the seasonal-naive scaling, which is\n")
cat("  about 5.6x larger here because a 12-month difference on a trending series\n")
cat("  is mostly drift; it makes every model look far better and must never be\n")
cat("  quoted on its own. Both columns are shown so the choice is visible.\n")
cat("* LjungBox_p is NOT comparable across rows, which is why the lag and df are\n")
cat("  printed beside it. The rows test different residuals at different lags on\n")
cat("  different sample sizes - the ARX-GARCH row is on standardised residuals of\n")
cat("  the DIFFERENCED series, the others on level residuals. Read each row as a\n")
cat("  diagnostic of its own model, never as a ranking.\n")
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

if (length(dupes)) {
  cat("\n*** DUPLICATE SPECIFICATIONS IN THIS TABLE ***\n")
  for (v in names(dupes)) {
    cat("  ", paste(dupes[[v]], collapse = "  ==  "), "\n", sep = "")
    cat("    same spec: ", v, "\n", sep = "")
  }
  cat("  These are ONE model reached by different search procedures, not\n")
  cat("  independent results. Any RMSE gap between them is optimizer tolerance.\n")
  cat("  Do not report them as corroborating each other, and do not treat the\n")
  cat("  ordering between them as a finding.\n")
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
  cat("These refit a primary model WITH COVID dummies, holding the orders fixed",
      "\nso that only one thing changes. They forecast worse but leave far better",
      "\nbehaved residuals. Each script measures this on every run.\n")
  scols <- c("Model", "Test_RMSE", "Test_MAE", "Test_MAPE", "Test_MASE")
  print(sens[, scols[scols %in% names(sens)]], row.names = FALSE, digits = 5)
  write.csv(sens, "model_comparison_sensitivity.csv", row.names = FALSE)
}

write.csv(results[order(results$Test_RMSE), ], "model_comparison.csv",
          row.names = FALSE)
write.csv(data.frame(Benchmark = rownames(bench), bench, row.names = NULL),
          "model_comparison_benchmarks.csv", row.names = FALSE)

cat("\n--- IMPORTANT: this table rests on eleven observations ---\n")
cat("Models within a few percent of each other cannot be separated here, and the",
    "\nsingle block can flatter a model outright: on this block ARX-GARCH and",
    "\nHolt-Winters both beat the random-walk benchmark, but across 24 forecast",
    "\norigins (rolling_cv.R) both are WORSE than it. Run rolling_cv.R and read",
    "\nrolling_cv_summary.csv alongside this table - with the leakage caveat that",
    "\nscript states in its own header.\n")

cat("\nWrote model_comparison.csv and model_comparison_benchmarks.csv\n")
