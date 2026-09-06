# ==============================================================================
# run_all.R - reproduce every result in this project from the raw workbook
#
#   Rscript run_all.R
#
# Order matters, and in two directions:
#   - data_processing.R writes the CSV that common.R and every model reads.
#   - compare_models.R AND rolling_cv.R both run last, because both read the
#     one-row *_result.csv each model writes. rolling_cv.R in particular takes
#     the SELECTED specification from those files rather than keeping its own
#     copy, so running it against stale result files scores stale models.
#
# Each script is run in its own environment so nothing leaks between them - one
# script's stray variable can no longer change another's result. session_info.txt
# is written at the end so the package versions behind a set of results are on
# the record.
#
# RUNTIME: roughly ten to thirty minutes end to end, dominated by SARIMA.R's
# 225-specification grid. That grid is cached in sarima_model_selection.csv with
# a fingerprint of the data and the split; an unchanged project reuses it and the
# whole run drops to a few minutes. rolling_cv.R refits every model at 24
# forecast origins and is the other slow step.
# ==============================================================================

SCRIPTS <- c(
  "data_processing.R",
  "auto_arima.R",
  "SARIMA.R",
  "HoltWinter.R",
  "ARX-GARCH.R",
  "compare_models.R",
  "rolling_cv.R"
)

missing <- SCRIPTS[!file.exists(SCRIPTS)]
if (length(missing)) {
  stop("Missing script(s): ", paste(missing, collapse = ", "),
       "\nRun this from the project root.")
}

started <- Sys.time()
timings <- data.frame()

for (s in SCRIPTS) {
  cat("\n\n")
  cat(strrep("=", 78), "\n")
  cat("RUNNING:", s, "\n")
  cat(strrep("=", 78), "\n")

  t0 <- Sys.time()
  ok <- tryCatch({
    # local = new.env() keeps each script's objects to itself.
    source(s, local = new.env(), echo = FALSE)
    TRUE
  }, error = function(e) {
    cat("\n*** FAILED:", s, "-", conditionMessage(e), "\n")
    FALSE
  })
  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "mins"))

  timings <- rbind(timings, data.frame(Script = s, Minutes = round(elapsed, 2),
                                       Status = ifelse(ok, "OK", "FAILED")))

  if (!ok && s == "data_processing.R") {
    stop("data_processing.R failed; nothing downstream can run.")
  }
}

cat("\n\n")
cat(strrep("=", 78), "\n")
cat("RUN SUMMARY\n")
cat(strrep("=", 78), "\n")
print(timings, row.names = FALSE)
cat(sprintf("\nTotal: %.1f minutes\n",
            as.numeric(difftime(Sys.time(), started, units = "mins"))))

if (any(timings$Status == "FAILED")) {
  cat("\nOne or more scripts failed - the comparison table may be incomplete.\n")
}

# Record the environment these results came from.
local({
  source("common.R", local = TRUE)
  write_session_info()
})

cat("\nDone.\n")
