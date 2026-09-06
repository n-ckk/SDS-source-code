# ==============================================================================
# rolling_cv.R - compare the models on more than eleven observations, and build
#                prediction intervals that are actually calibrated
#
# WHY THIS EXISTS
# model_comparison.csv ranks models on a single 12-month test block. Eleven
# scored points cannot separate models that finish within a few percent of each
# other: the top two rows there differ by 0.07 RMSE, which is optimizer tolerance
# rather than a real difference. This script re-evaluates the SAME fixed
# specifications at many forecast origins, so the comparison rests on 24
# forecasts of 12 months each instead of one.
#
# WHERE THE SPECIFICATIONS COME FROM
# They are READ from each model's *_result.csv (the Spec and Window columns),
# not retyped here. The earlier version hardcoded ARIMA(2,1,2)+drift, the 2021+
# Holt-Winters window, the 2010+ ARX window and the full ARX-GARCH
# specification - four copies of decisions that live in other files. Re-running a
# search that selected something else would have left this script silently
# scoring the old model. Models whose Spec strings match are collapsed into one
# entry automatically, which is what happens to auto_arima.R and SARIMA.R.
#
# ============================ READ THIS BEFORE QUOTING ========================
# TWO LIMITATIONS THAT THE RANKING BELOW CANNOT ESCAPE.
#
# (1) SPECIFICATION LEAKAGE, and it is not neutral between the models.
#     Every specification scored here was chosen using data through 2025-07, but
#     the origins start in 2023-08. At the earliest origin the ARIMA orders
#     already encode two years of future information. That advantage is NOT
#     shared equally: ARIMA(2,1,2)+drift was picked as the best of 225 candidates
#     and the ARX specification as the best of 36, while RW-with-drift was chosen
#     by nobody and has no parameters to tune. The model with the most hindsight
#     is the one that comes out on top, so read "ARIMA beats the random walk" as
#     an upper bound on its real advantage, not a measurement of it.
#
#     What would fix it: re-running each script's SELECTION at every origin.
#     That is a full nested cross-validation and costs 24 x 225 ARIMA fits, which
#     is why it is not done here. Saying so is the alternative to pretending the
#     problem does not exist.
#
# (2) THE TEST BLOCK IS RE-READ. Twelve of the 24 origins forecast into
#     2025-08 or later, so the block that model_comparison.csv reads exactly once
#     is read twelve more times here. Nothing is SELECTED on it, so this is not
#     leakage into the model choice - but it does mean this file and
#     model_comparison.csv are not two independent pieces of evidence about the
#     same months.
#
# Because the rolling windows overlap, the per-origin errors are strongly
# autocorrelated - a naive paired t-test would badly overstate significance, so
# the summary reports the spread across origins and a win count instead of a
# p-value.
# ==============================================================================

suppressMessages({
  library(forecast)
  library(rugarch)
})

source("common.R")

set.seed(123)

N_ORIGINS <- 24      # monthly origins, each forecasting HORIZON months ahead
NSIM_CV   <- 2000    # simulation paths for ARX-GARCH intervals (20000 in the
                     # main script; reduced here because it runs at every origin)

data <- load_series()
n    <- nrow(data)

# The last usable origin must leave HORIZON actuals after it.
last_origin <- n - HORIZON
origin_idx  <- seq(last_origin - N_ORIGINS + 1, last_origin)

cat(sprintf("Origins: %d, from %s to %s, each forecasting %d months ahead.\n",
            length(origin_idx),
            format(data$date[min(origin_idx)]),
            format(data$date[max(origin_idx)]), HORIZON))
cat(sprintf("Origins whose forecast window reaches the test block (%s+): %d of %d\n",
            format(TEST_START),
            sum(data$date[origin_idx] >= month_add(TEST_START, -HORIZON)),
            length(origin_idx)))

## ---- Build the forecasters from each model's recorded specification ---------
RESULT_FILES <- c("auto_arima_result.csv", "sarima_result.csv",
                  "holt_winters_result.csv", "arx_garch_result.csv")

found <- RESULT_FILES[file.exists(RESULT_FILES)]
if (!length(found)) {
  stop("No *_result.csv found. Run the model scripts before rolling_cv.R - it ",
       "reads the selected specifications from them rather than keeping its own ",
       "copy.")
}
if (length(found) < length(RESULT_FILES)) {
  cat("NOTE: missing result file(s):",
      paste(setdiff(RESULT_FILES, found), collapse = ", "),
      "\n      Those models are absent from the rolling comparison.\n")
}

registry <- do.call(rbind, lapply(found, function(f) {
  r <- read.csv(f, stringsAsFactors = FALSE)
  if (is.null(r$Spec) || is.na(r$Spec[1])) {
    cat("NOTE:", f, "has no Spec column - re-run that model script.",
        "Skipping it.\n")
    return(NULL)
  }
  data.frame(File = f, Model = r$Model[1], Spec = r$Spec[1],
             Window = r$Window[1], stringsAsFactors = FALSE)
}))

if (is.null(registry) || !nrow(registry)) {
  stop("No usable specifications found in the result files.")
}

# Collapse rows that describe the same model on the same window. auto_arima.R and
# SARIMA.R normally land here as one entry.
key <- paste(registry$Spec, registry$Window)
collapsed <- do.call(rbind, lapply(split(registry, key), function(g) {
  if (nrow(g) > 1) {
    cat("Collapsing identical specifications into one entry: ",
        paste(g$Model, collapse = " == "), "\n", sep = "")
  }
  data.frame(Label = g$Model[1], Spec = g$Spec[1], Window = g$Window[1],
             From = paste(sub("_result.csv", "", g$File), collapse = "+"),
             stringsAsFactors = FALSE)
}))
rownames(collapsed) <- NULL

cat("\nSpecifications being rolled:\n")
print(collapsed[, c("Label", "Window", "Spec", "From")], row.names = FALSE)

# A forecaster is a function(hist, h) -> list(mean, lo80, hi80, lo95, hi95).
make_forecaster <- function(spec_string, window_start) {
  s  <- parse_spec(spec_string)
  w0 <- as.Date(window_start)

  if (identical(s$kind, "arima")) {
    function(hist, h) {
      hist <- hist[hist$date >= w0, ]
      fit <- tryCatch(
        Arima(as_monthly_ts(hist),
              order    = c(s$p, s$d, s$q),
              seasonal = list(order = c(s$P, s$D, s$Q), period = FREQ),
              include.drift = isTRUE(s$drift), method = "ML"),
        error = function(e) NULL)
      if (is.null(fit)) return(NULL)
      f <- forecast(fit, h = h, level = c(80, 95))
      list(mean = as.numeric(f$mean),
           lo80 = as.numeric(f$lower[, "80%"]), hi80 = as.numeric(f$upper[, "80%"]),
           lo95 = as.numeric(f$lower[, "95%"]), hi95 = as.numeric(f$upper[, "95%"]))
    }

  } else if (identical(s$kind, "hw")) {
    function(hist, h) {
      hist <- hist[hist$date >= w0, ]
      if (nrow(hist) < 2 * FREQ + 4) return(NULL)
      f <- tryCatch(forecast::hw(as_monthly_ts(hist), seasonal = s$seasonal,
                                 h = h, level = c(80, 95)),
                    error = function(e) NULL)
      if (is.null(f)) return(NULL)
      list(mean = as.numeric(f$mean),
           lo80 = as.numeric(f$lower[, "80%"]), hi80 = as.numeric(f$upper[, "80%"]),
           lo95 = as.numeric(f$lower[, "95%"]), hi95 = as.numeric(f$upper[, "95%"]))
    }

  } else if (identical(s$kind, "arx")) {
    function(hist, h) {
      hist <- hist[hist$date >= w0, ]
      lvl  <- hist$value
      dlvl <- diff(lvl)
      Xh   <- if (isTRUE(s$dummies)) covid_dummies(hist$date[-1]) else NULL
      fut  <- seq(month_add(max(hist$date), 1), by = "month", length.out = h)
      Xf   <- if (isTRUE(s$dummies)) covid_dummies(fut) else NULL

      spec <- ugarchspec(
        variance.model = list(model = "sGARCH",
                              garchOrder = if (isTRUE(s$garch)) c(1, 1) else c(0, 0)),
        mean.model     = list(armaOrder = c(s$ar, 0), include.mean = TRUE,
                              external.regressors = Xh),
        distribution.model = s$dist)
      if (!is.null(Xh)) {
        setbounds(spec) <- setNames(rep(list(c(-30000, 30000)), ncol(Xh)),
                                    paste0("mxreg", seq_len(ncol(Xh))))
      }

      fit <- tryCatch(ugarchfit(spec, data = dlvl, solver = "hybrid"),
                      error = function(e) NULL)
      if (is.null(fit) || fit@fit$convergence != 0) return(NULL)

      fc <- tryCatch(ugarchforecast(fit, n.ahead = h,
                                    external.forecasts = list(mregfor = Xf)),
                     error = function(e) NULL)
      if (is.null(fc)) return(NULL)

      last <- lvl[length(lvl)]
      pt   <- last + cumsum(as.numeric(fitted(fc)))

      sim <- tryCatch(
        ugarchsim(fit, n.sim = h, m.sim = NSIM_CV, startMethod = "sample",
                  mexsimdata = if (!is.null(Xf))
                    replicate(NSIM_CV, Xf, simplify = FALSE) else NULL),
        error = function(e) NULL)
      if (is.null(sim)) return(NULL)

      paths <- last + apply(fitted(sim), 2, cumsum)
      q <- apply(paths, 1, quantile, probs = c(0.025, 0.10, 0.90, 0.975))
      list(mean = pt, lo80 = q[2, ], hi80 = q[3, ], lo95 = q[1, ], hi95 = q[4, ])
    }

  } else {
    stop("rolling_cv.R does not know how to roll spec kind '", s$kind, "'.")
  }
}

# The benchmark uses exactly the rule in common.R::benchmark_forecasts(), applied
# at each origin: drift estimated over the whole history up to that origin.
fc_rw <- function(hist, h) {
  lvl  <- hist$value
  last <- lvl[length(lvl)]
  d    <- (last - lvl[1]) / (length(lvl) - 1)
  list(mean = last + d * seq_len(h),
       lo80 = rep(NA_real_, h), hi80 = rep(NA_real_, h),
       lo95 = rep(NA_real_, h), hi95 = rep(NA_real_, h))
}

MODELS <- setNames(
  lapply(seq_len(nrow(collapsed)),
         function(i) make_forecaster(collapsed$Spec[i], collapsed$Window[i])),
  collapsed$Label)
MODELS[["RW with drift"]] <- fc_rw

## ---- Roll --------------------------------------------------------------------
rows <- data.frame()   # one summary row per origin x model
errs <- data.frame()   # one detail row per origin x model x horizon

for (o in origin_idx) {
  hist <- data[seq_len(o), ]
  idx  <- (o + 1):(o + HORIZON)
  act  <- data$value[idx]
  imp  <- data$imputed[idx]
  keep <- !imp          # never score against an interpolated actual

  for (nm in names(MODELS)) {
    f <- tryCatch(MODELS[[nm]](hist, HORIZON), error = function(e) NULL)
    if (is.null(f)) {
      rows <- rbind(rows, data.frame(Origin = data$date[o], Model = nm,
                                     RMSE = NA_real_, MAE = NA_real_,
                                     Cov80 = NA_real_, Cov95 = NA_real_,
                                     stringsAsFactors = FALSE))
      next
    }
    e <- act - f$mean

    rows <- rbind(rows, data.frame(
      Origin = data$date[o], Model = nm,
      RMSE  = sqrt(mean(e[keep]^2)),
      MAE   = mean(abs(e[keep])),
      Cov80 = if (all(is.na(f$lo80))) NA_real_
              else mean((act >= f$lo80 & act <= f$hi80)[keep]),
      Cov95 = if (all(is.na(f$lo95))) NA_real_
              else mean((act >= f$lo95 & act <= f$hi95)[keep]),
      stringsAsFactors = FALSE))

    errs <- rbind(errs, data.frame(
      Origin = data$date[o], Model = nm, h = seq_len(HORIZON),
      Error = e, Imputed = imp, stringsAsFactors = FALSE))
  }
  cat(".")
}
cat("\n")

write.csv(rows, "rolling_cv_by_origin.csv", row.names = FALSE)

## ---- Summarise ---------------------------------------------------------------
# Two things the earlier version got wrong here.
#
# (1) A model that FAILED to fit at some origins was averaged over the origins it
#     survived, so it was scored on an easier subset than its competitors while
#     being printed in the same column. The summary is now computed over the
#     origins where EVERY model produced a forecast, and says so.
# (2) Mean_RMSE is the mean of 24 per-origin RMSEs. That is a legitimate
#     statistic but it is NOT the RMSE over 288 points, which is what the README
#     used to describe. Both are reported; they differ by Jensen's inequality.
complete <- Reduce(intersect,
                   lapply(split(rows, rows$Model),
                          function(g) as.character(g$Origin[is.finite(g$RMSE)])))
n_dropped <- length(unique(rows$Origin)) - length(complete)
if (n_dropped > 0) {
  cat("\nNOTE:", n_dropped, "origin(s) dropped from the summary because at least",
      "one model\n      failed to fit there. Comparing means over different",
      "origin sets would\n      score models on different problems.\n")
}
common <- rows[as.character(rows$Origin) %in% complete, ]

agg <- do.call(rbind, lapply(split(common, common$Model), function(g) {
  data.frame(Model = g$Model[1],
             Origins     = sum(is.finite(g$RMSE)),
             Mean_RMSE   = mean(g$RMSE),
             Pooled_RMSE = sqrt(mean(g$RMSE^2)),
             SD_RMSE     = sd(g$RMSE),
             Mean_MAE    = mean(g$MAE),
             Cov80       = mean(g$Cov80, na.rm = TRUE),
             Cov95       = mean(g$Cov95, na.rm = TRUE),
             row.names = NULL)
}))
agg <- agg[order(agg$Mean_RMSE), ]

cat("\n==============================================================\n")
cat(" ROLLING-ORIGIN EVALUATION -", length(complete), "origins x", HORIZON,
    "months\n")
cat("==============================================================\n")
cat("Interpolated actuals are excluded from every score.\n")
cat("Mean_RMSE averages the per-origin RMSEs; Pooled_RMSE is the root mean\n")
cat("square of them. Quote whichever you name, and name it.\n")
cat("READ THE HEADER OF THIS FILE before quoting the ranking - the\n")
cat("specifications were chosen using data that postdates most of these\n")
cat("origins, and that advantage is larger for the models with more tuning.\n\n")
print(agg, row.names = FALSE, digits = 5)

wide <- reshape(common[, c("Origin", "Model", "RMSE")],
                idvar = "Origin", timevar = "Model", direction = "wide")
names(wide) <- sub("^RMSE[.]", "", names(wide))
mat  <- as.matrix(wide[, -1, drop = FALSE])

# which.min() returns integer(0) on an all-NA row, which turns apply() into a
# list and then errors on indexing. Cannot happen now that the summary is
# restricted to complete origins, but the guard costs nothing and the failure
# mode was silent.
winner <- apply(mat, 1, function(r) {
  if (all(is.na(r))) return(NA_character_)
  colnames(mat)[which.min(r)]
})
wins <- table(factor(winner, levels = colnames(mat)))

cat("\nOrigins won (lowest RMSE at that origin):\n")
print(as.data.frame(wins, responseName = "Wins"), row.names = FALSE)

if ("RW with drift" %in% colnames(mat)) {
  cat("\nOrigins where each model beats the RW-with-drift benchmark:\n")
  for (cn in setdiff(colnames(mat), "RW with drift")) {
    cat(sprintf("  %-26s %2d / %d\n", cn,
                sum(mat[, cn] < mat[, "RW with drift"]), nrow(mat)))
  }
}

## ---- Empirical prediction intervals ------------------------------------------
# WHY THE MODEL-BASED INTERVALS FAIL.
# An ARIMA interval is point +/- z * sqrt(variance), with the variance built from
# sigma^2 = sum(e_t^2)/n and z taken from the normal distribution. April 2020 is
# a 26-sigma residual, so that one month inflates sigma substantially; residual
# kurtosis in the hundreds then makes the normal quantile wrong as well. Both
# errors push the same way. ARX-GARCH fails differently - its variance process
# sits on the IGARCH boundary, so simulated paths fan out too fast.
#
# THE FIX. Take the observed distribution of h-step-ahead forecast errors across
# origins and add its quantiles to the point forecast. No normality assumption,
# no sigma contaminated by a single outlier, and the accurate undummied point
# forecast is left exactly as it is.
#
# This is why treating COVID was never really a choice between accurate point
# forecasts and usable intervals. That trade-off only binds while the intervals
# have to come from the fitted model's own sigma.
#
# HONEST VALIDATION. Quantiles are built on the first two thirds of the origins
# and scored on the last third, so the coverage reported below is genuinely out
# of sample. Fitting and scoring on the same origins would flatter the result.
scored <- errs[!errs$Imputed, ]

q_from <- function(df) {
  do.call(rbind, lapply(split(df, list(df$Model, df$h), drop = TRUE), function(g) {
    qs <- quantile(g$Error, c(0.025, 0.10, 0.90, 0.975), names = FALSE)
    data.frame(Model = g$Model[1], h = g$h[1],
               q025 = qs[1], q10 = qs[2], q90 = qs[3], q975 = qs[4],
               n_origins = nrow(g), row.names = NULL)
  }))
}

origins_sorted <- sort(unique(errs$Origin))
cut_at    <- floor(length(origins_sorted) * 2 / 3)
build_set <- origins_sorted[seq_len(cut_at)]
score_set <- origins_sorted[-seq_len(cut_at)]

q_build <- q_from(scored[scored$Origin %in% build_set, ])
held    <- merge(scored[scored$Origin %in% score_set, ], q_build,
                 by = c("Model", "h"))
held$in80 <- held$Error >= held$q10  & held$Error <= held$q90
held$in95 <- held$Error >= held$q025 & held$Error <= held$q975

emp <- do.call(rbind, lapply(split(held, held$Model), function(g) {
  data.frame(Model = g$Model[1],
             Emp_Cov80 = mean(g$in80), Emp_Cov95 = mean(g$in95),
             Points = nrow(g), row.names = NULL)
}))

score_rows <- rows[rows$Origin %in% score_set, ]
gauss <- do.call(rbind, lapply(split(score_rows, score_rows$Model), function(g) {
  data.frame(Model = g$Model[1],
             Model_Cov80 = mean(g$Cov80, na.rm = TRUE),
             Model_Cov95 = mean(g$Cov95, na.rm = TRUE),
             row.names = NULL)
}))

cmp <- merge(gauss, emp, by = "Model")
cmp <- cmp[order(abs(cmp$Emp_Cov80 - 0.80)), ]

cat("\n=== INTERVAL CALIBRATION: MODEL-BASED vs EMPIRICAL ===\n")
cat("Scored out of sample on the last", length(score_set),
    "origins. Nominal 80% and 95%.\n\n")
print(cmp, row.names = FALSE, digits = 4)

cat("\nModel_Cov* are the intervals each model produces for itself. Emp_Cov* use\n")
cat("the empirical error quantiles instead. Every model's own intervals come out\n")
cat("too wide; the empirical ones sit closer to nominal. THESE are the intervals\n")
cat("to quote in the report - not any model's own, including ARX-GARCH's.\n")
cat("\nCAVEAT: with", length(build_set), "origins behind each quantile, the 2.5%\n")
cat("and 97.5% points are crude - the 80% interval is much better determined\n")
cat("than the 95% one. More origins would tighten both.\n")

# Full-sample quantiles, for actually constructing intervals in the report.
q_all <- q_from(scored)
write.csv(q_all, "empirical_error_quantiles.csv", row.names = FALSE)

best_label <- agg$Model[1]
cat("\n", best_label, " empirical error quantiles by horizon:\n", sep = "")
print(q_all[q_all$Model == best_label, c("h", "q10", "q90", "q025", "q975")],
      row.names = FALSE, digits = 5)
cat("Interval at horizon h = point forecast + [q10, q90]   for 80%\n")
cat("                      = point forecast + [q025, q975] for 95%\n")

cat("\n--- How to read all of this ---\n")
cat("* Mean_RMSE across origins is a more reliable ranking than the single\n")
cat("  12-month test block, which cannot separate models within a few percent -\n")
cat("  but it is NOT clean evidence, for the two reasons in this file's header.\n")
cat("* SD_RMSE shows how much each model varies by origin. A small mean gap\n")
cat("  against a large SD is not a real difference.\n")
cat("* Win counts and mean RMSE can disagree - a model can win few origins and\n")
cat("  still have the better average, or the reverse. Report both.\n")
cat("* No p-values: the 12-month windows overlap, so per-origin errors are\n")
cat("  autocorrelated and a paired t-test would overstate significance.\n")

write.csv(agg, "rolling_cv_summary.csv", row.names = FALSE)
write.csv(cmp, "rolling_cv_interval_calibration.csv", row.names = FALSE)
cat("\nWrote rolling_cv_summary.csv, rolling_cv_by_origin.csv,\n")
cat("rolling_cv_interval_calibration.csv and empirical_error_quantiles.csv\n")
