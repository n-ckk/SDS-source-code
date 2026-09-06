# ==============================================================================
# rolling_cv.R - compare the models on more than twelve observations
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
#
# Runtime: a few minutes, dominated by refitting the GARCH at every origin.
# ==============================================================================

suppressMessages({
  library(forecast)
  library(rugarch)
})

source("common.R")

set.seed(123)

N_ORIGINS <- 24      # monthly origins, each forecasting HORIZON months ahead
NSIM_CV   <- 2000    # simulation paths for ARX-GARCH intervals (20000 in the
                     # main script; reduced here because it runs 24 times)

data <- load_series()
n    <- nrow(data)

# The last usable origin must leave HORIZON actuals after it.
last_origin  <- n - HORIZON
origin_idx   <- seq(last_origin - N_ORIGINS + 1, last_origin)

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

## ---- Roll ---------------------------------------------------------------------
rows <- data.frame()

for (o in origin_idx) {
  hist <- data[seq_len(o), ]
  act  <- data$value[(o + 1):(o + HORIZON)]

  for (nm in names(MODELS)) {
    f <- tryCatch(MODELS[[nm]](hist, HORIZON), error = function(e) NULL)
    if (is.null(f)) {
      rows <- rbind(rows, data.frame(Origin = data$date[o], Model = nm,
                                     RMSE = NA_real_, MAE = NA_real_,
                                     Cov80 = NA_real_, Cov95 = NA_real_))
      next
    }
    rows <- rbind(rows, data.frame(
      Origin = data$date[o], Model = nm,
      RMSE = sqrt(mean((act - f$mean)^2)),
      MAE  = mean(abs(act - f$mean)),
      Cov80 = if (all(is.na(f$lo80))) NA_real_
              else mean(act >= f$lo80 & act <= f$hi80),
      Cov95 = if (all(is.na(f$lo95))) NA_real_
              else mean(act >= f$lo95 & act <= f$hi95),
      stringsAsFactors = FALSE))
  }
  cat(".")
}
cat("\n")

write.csv(rows, "rolling_cv_by_origin.csv", row.names = FALSE)

## ---- Summarise ----------------------------------------------------------------
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
print(agg, row.names = FALSE, digits = 5)

# Win counts: which model had the lowest RMSE at each origin. Robust to the
# overlapping-window dependence that rules out a naive t-test.
wide <- reshape(rows[, c("Origin", "Model", "RMSE")],
                idvar = "Origin", timevar = "Model", direction = "wide")
names(wide) <- sub("^RMSE\\.", "", names(wide))
mat  <- as.matrix(wide[, -1])
wins <- table(factor(colnames(mat)[apply(mat, 1, which.min)],
                     levels = colnames(mat)))

cat("\nOrigins won (lowest RMSE at that origin):\n")
print(as.data.frame(wins, responseName = "Wins"), row.names = FALSE)

cat("\nInterval coverage, pooled over", length(origin_idx) * HORIZON,
    "forecast points\n")
cat("(nominal 80% and 95%; well above nominal means intervals too WIDE):\n")
print(agg[, c("Model", "Cov80", "Cov95")], row.names = FALSE, digits = 4)

cat("\n--- How to read this ---\n")
cat("* Mean_RMSE here is a far more reliable ranking than the single 12-month\n")
cat("  test block, which cannot separate models within a few percent.\n")
cat("* SD_RMSE shows how much each model's accuracy varies by origin. A small\n")
cat("  mean difference against a large SD is not a real difference.\n")
cat("* No p-values: the 12-month windows overlap, so per-origin errors are\n")
cat("  strongly autocorrelated and a paired t-test would overstate significance.\n")
cat("* Coverage is now measured on hundreds of points rather than twelve, which\n")
cat("  is the only way to tell a miscalibrated interval from a small sample.\n")

write.csv(agg, "rolling_cv_summary.csv", row.names = FALSE)
cat("\nWrote rolling_cv_summary.csv and rolling_cv_by_origin.csv\n")
