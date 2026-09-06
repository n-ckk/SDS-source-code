# ==============================================================================
# rolling_cv.R - compare the models on more than twelve observations, and build
#                prediction intervals that are actually calibrated
#
# WHY THIS EXISTS
# model_comparison.csv ranks models on a single 12-month test block. Twelve
# points cannot separate models that finish within a few percent of each other:
# the top two rows there differ by 0.07 RMSE, which is optimizer tolerance rather
# than a real difference. This script re-evaluates the SAME fixed specifications
# at many forecast origins, so the comparison rests on 24 forecasts of 12 months
# each instead of one.
#
# WHAT IT IS NOT
# This is a supplementary evaluation, not a second test set, and nothing is
# selected on it. Every specification below was already fixed by the train /
# validation procedure in its own script; this only re-scores them. Because the
# rolling windows overlap, the per-origin errors are strongly autocorrelated -
# a naive paired t-test would badly overstate significance, so the summary
# reports the spread across origins and a win count instead of a p-value.
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

# tiny month-add helper so this script needs no lubridate
`%m_plus%` <- function(d, k) seq(d, by = "month", length.out = k + 1)[k + 1]

## ---- One forecaster per DISTINCT model ---------------------------------------
# auto_arima.R and SARIMA.R both select ARIMA(2,1,2) with drift - identical AIC
# to ten significant figures - so they are one entry here, not two.

fc_arima <- function(hist, h) {
  y   <- as_monthly_ts(hist)
  fit <- tryCatch(Arima(y, order = c(2, 1, 2), include.drift = TRUE),
                  error = function(e) NULL)
  if (is.null(fit)) return(NULL)
  f <- forecast(fit, h = h, level = c(80, 95))
  list(mean = as.numeric(f$mean),
       lo80 = as.numeric(f$lower[, "80%"]), hi80 = as.numeric(f$upper[, "80%"]),
       lo95 = as.numeric(f$lower[, "95%"]), hi95 = as.numeric(f$upper[, "95%"]))
}

fc_hw <- function(hist, h) {
  hist <- hist[hist$date >= as.Date("2021-01-01"), ]
  if (nrow(hist) < 2 * FREQ + 4) return(NULL)
  f <- tryCatch(forecast::hw(as_monthly_ts(hist), seasonal = "additive",
                             h = h, level = c(80, 95)),
                error = function(e) NULL)
  if (is.null(f)) return(NULL)
  list(mean = as.numeric(f$mean),
       lo80 = as.numeric(f$lower[, "80%"]), hi80 = as.numeric(f$upper[, "80%"]),
       lo95 = as.numeric(f$lower[, "95%"]), hi95 = as.numeric(f$upper[, "95%"]))
}

fc_arx <- function(hist, h) {
  hist <- hist[hist$date >= as.Date("2010-01-01"), ]
  lvl  <- hist$value
  dlvl <- diff(lvl)
  Xh   <- covid_dummies(hist$date[-1])
  fut_dates <- seq(max(hist$date) %m_plus% 1, by = "month", length.out = h)
  Xf   <- covid_dummies(fut_dates)

  spec <- ugarchspec(
    variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
    mean.model     = list(armaOrder = c(0, 0), include.mean = TRUE,
                          external.regressors = Xh),
    distribution.model = "std")
  setbounds(spec) <- setNames(rep(list(c(-30000, 30000)), ncol(Xh)),
                              paste0("mxreg", seq_len(ncol(Xh))))

  fit <- tryCatch(ugarchfit(spec, data = dlvl, solver = "hybrid"),
                  error = function(e) NULL)
  if (is.null(fit) || fit@fit$convergence != 0) return(NULL)

  fc <- tryCatch(ugarchforecast(fit, n.ahead = h,
                                external.forecasts = list(mregfor = Xf)),
                 error = function(e) NULL)
  if (is.null(fc)) return(NULL)

  last <- lvl[length(lvl)]
  pt   <- last + cumsum(as.numeric(fitted(fc)))

  sim <- tryCatch(ugarchsim(fit, n.sim = h, m.sim = NSIM_CV,
                            startMethod = "sample",
                            mexsimdata = replicate(NSIM_CV, Xf, simplify = FALSE)),
                  error = function(e) NULL)
  if (is.null(sim)) return(NULL)

  paths <- last + apply(fitted(sim), 2, cumsum)
  q <- apply(paths, 1, quantile, probs = c(0.025, 0.10, 0.90, 0.975))
  list(mean = pt, lo80 = q[2, ], hi80 = q[3, ], lo95 = q[1, ], hi95 = q[4, ])
}

fc_rw <- function(hist, h) {
  lvl  <- hist$value
  last <- lvl[length(lvl)]
  d    <- (last - lvl[1]) / (length(lvl) - 1)
  list(mean = last + d * seq_len(h),
       lo80 = rep(NA_real_, h), hi80 = rep(NA_real_, h),
       lo95 = rep(NA_real_, h), hi95 = rep(NA_real_, h))
}

MODELS <- list(
  "ARIMA(2,1,2)+drift"    = fc_arima,   # = auto_arima.R = SARIMA.R
  "Holt-Winters additive" = fc_hw,
  "ARX(0)-GARCH(1,1) std" = fc_arx,
  "RW with drift"         = fc_rw
)

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
                                     Cov80 = NA_real_, Cov95 = NA_real_))
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
agg <- do.call(rbind, lapply(split(rows, rows$Model), function(g) {
  data.frame(Model = g$Model[1],
             Origins   = sum(is.finite(g$RMSE)),
             Mean_RMSE = mean(g$RMSE, na.rm = TRUE),
             SD_RMSE   = sd(g$RMSE, na.rm = TRUE),
             Mean_MAE  = mean(g$MAE,  na.rm = TRUE),
             Cov80     = mean(g$Cov80, na.rm = TRUE),
             Cov95     = mean(g$Cov95, na.rm = TRUE),
             row.names = NULL)
}))
agg <- agg[order(agg$Mean_RMSE), ]

cat("\n==============================================================\n")
cat(" ROLLING-ORIGIN EVALUATION -", length(origin_idx), "origins x", HORIZON,
    "months\n")
cat("==============================================================\n")
cat("Interpolated actuals are excluded from every score.\n\n")
print(agg, row.names = FALSE, digits = 5)

wide <- reshape(rows[, c("Origin", "Model", "RMSE")],
                idvar = "Origin", timevar = "Model", direction = "wide")
names(wide) <- sub("^RMSE\\.", "", names(wide))
mat  <- as.matrix(wide[, -1])
wins <- table(factor(colnames(mat)[apply(mat, 1, which.min)],
                     levels = colnames(mat)))

cat("\nOrigins won (lowest RMSE at that origin):\n")
print(as.data.frame(wins, responseName = "Wins"), row.names = FALSE)

## ---- Empirical prediction intervals ------------------------------------------
# WHY THE MODEL-BASED INTERVALS FAIL.
# An ARIMA interval is point +/- z * sqrt(variance), with the variance built from
# sigma^2 = sum(e_t^2)/n and z taken from the normal distribution. April 2020 is
# a 26-sigma residual, so that one month contributes about 676 sigma^2 to a sum
# over 931 terms and inflates sigma by roughly 40%; residual kurtosis of 504 then
# makes the normal quantile wrong as well. Both errors push the same way, and the
# measured consequence is 98% coverage where 80% is nominal.
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
cat("the empirical error quantiles instead. Where the empirical column sits\n")
cat("closer to nominal, quote the empirical interval and say how it was built.\n")
cat("\nCAVEAT: with", length(build_set), "origins behind each quantile, the 2.5%\n")
cat("and 97.5% points are crude - the 80% interval is much better determined\n")
cat("than the 95% one. More origins would tighten both.\n")

# Full-sample quantiles, for actually constructing intervals in the report.
q_all <- q_from(scored)
write.csv(q_all, "empirical_error_quantiles.csv", row.names = FALSE)

cat("\nARIMA(2,1,2)+drift empirical error quantiles by horizon:\n")
print(q_all[q_all$Model == "ARIMA(2,1,2)+drift",
            c("h", "q10", "q90", "q025", "q975")],
      row.names = FALSE, digits = 5)
cat("Interval at horizon h = point forecast + [q10, q90]   for 80%\n")
cat("                      = point forecast + [q025, q975] for 95%\n")

cat("\n--- How to read all of this ---\n")
cat("* Mean_RMSE across origins is a far more reliable ranking than the single\n")
cat("  12-month test block, which cannot separate models within a few percent.\n")
cat("* SD_RMSE shows how much each model varies by origin. A small mean gap\n")
cat("  against a large SD is not a real difference.\n")
cat("* No p-values: the 12-month windows overlap, so per-origin errors are\n")
cat("  autocorrelated and a paired t-test would overstate significance.\n")

write.csv(agg, "rolling_cv_summary.csv", row.names = FALSE)
write.csv(cmp, "rolling_cv_interval_calibration.csv", row.names = FALSE)
cat("\nWrote rolling_cv_summary.csv, rolling_cv_by_origin.csv,\n")
cat("rolling_cv_interval_calibration.csv and empirical_error_quantiles.csv\n")
