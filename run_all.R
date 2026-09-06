# ==============================================================================
# run_all.R - reproduce every result in this project from the raw workbook
#
#   Rscript run_all.R
#
# Order matters: data_processing.R writes the CSV that common.R and every model
# reads. The comparison table is built last, from the one-row result file each
# model writes.
#
# Each script is run in its own environment so nothing leaks between them - one
# script's stray variable can no longer change another's result. session_info.txt
# is written at the end so the package versions behind a set of results are on
# the record.
#
# RUNTIME: about ten minutes end to end. SARIMA.R fits 225 candidate models
# (cached in sarima_model_selection.csv) and rolling_cv.R refits every model at
# 24 forecast origins.
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
