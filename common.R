# ==============================================================================
# common.R - shared foundation for every model script in this project
#
# WHY THIS FILE EXISTS
# Each model script used to re-implement its own train/test split, its own
# accuracy function and its own MASE denominator. Those copies drifted apart, and
# the result was a comparison table whose rankings were an artefact of the drift
# rather than of model quality: on the scripts' own numbers Holt-Winters beat
# auto.arima on MASE, while on a single denominator the order is reversed.
# Everything that must be identical across models is now defined here exactly
# once. Do not copy any of it back into a model script.
#
# Usage, after data_processing.R has written the cleaned CSV:
#   source("common.R")
# ==============================================================================

## ---- Configuration ----------------------------------------------------------
# One split for every model. A script MAY start its analysis window later than
# the series does - Holt-Winters uses 2021+, ARX-GARCH uses 2010+, and those are
# legitimate modelling choices - but it may NOT move these two boundaries. The
# validation and test blocks have to be the same twelve months everywhere or the
# comparison table is comparing different questions.
DATA_FILE  <- "processed_female_employment.csv"
VAL_START  <- as.Date("2024-08-01")   # validation block: 2024-08 .. 2025-07
TEST_START <- as.Date("2025-08-01")   # test block:       2025-08 .. 2026-07
FREQ       <- 12
HORIZON    <- 12                      # reported forecast horizon = test length

# COVID shock months, treated as additive outliers by every model whose training
# window spans them. Leaving them untreated puts a 26-sigma residual in the fit
# and makes the resulting prediction intervals meaningless.
COVID_SHOCK_MONTHS <- as.Date(c("2020-03-01", "2020-04-01",
                                "2020-05-01", "2020-06-01"))

## ---- Data -------------------------------------------------------------------
load_series <- function(path = DATA_FILE) {
  if (!file.exists(path)) {
    stop("Cleaned data not found: ", path, ". Run data_processing.R first.")
  }
  df <- read.csv(path, stringsAsFactors = FALSE)
  names(df)[1:2] <- c("date", "value")
  df$date  <- as.Date(df$date)
  df$value <- as.numeric(df$value)

  # data_processing.R records which months it interpolated. Older copies of the
  # CSV predate that column.
  if (is.null(df$imputed)) {
    warning("No 'imputed' column in ", path,
            " - re-run data_processing.R so synthetic months can be flagged.")
    df$imputed <- FALSE
  }
  df$imputed <- as.logical(df$imputed)

  df <- df[order(df$date), ]
  rownames(df) <- NULL
  df
}

# Build a monthly ts from a data frame produced by load_series() or split_series().
as_monthly_ts <- function(df) {
  ts(df$value,
     start = c(as.numeric(format(min(df$date), "%Y")),
               as.numeric(format(min(df$date), "%m"))),
     frequency = FREQ)
}

## ---- The split --------------------------------------------------------------
# Returns train / val / test / train_val / all as data frames. Fit candidates on
# train, score them on val, refit the winner on train_val, and touch test exactly
# once at the very end.
split_series <- function(df, window_start = NULL) {
  if (!is.null(window_start)) {
    df <- df[df$date >= as.Date(window_start), ]
    rownames(df) <- NULL
  }
  parts <- list(
    train     = df[df$date <  VAL_START, ],
    val       = df[df$date >= VAL_START & df$date < TEST_START, ],
    test      = df[df$date >= TEST_START, ],
    train_val = df[df$date <  TEST_START, ],
    all       = df
  )
  stopifnot(nrow(parts$val) == HORIZON, nrow(parts$test) == HORIZON)
  parts
}

describe_split <- function(parts, label = "") {
  cat("\n=== DATA SPLIT", if (nzchar(label)) paste0("- ", label) else "", "===\n")
  fmt <- function(p, nm, note = "") {
    cat(sprintf("%-10s: %s to %s  (%3d months) %s\n", nm,
                format(min(p$date)), format(max(p$date)), nrow(p), note))
  }
  fmt(parts$train, "Train",      "<- fit candidates")
  fmt(parts$val,   "Validation", "<- selection only")
  fmt(parts$test,  "Test",       "<- scored ONCE")
  n_imp <- sum(parts$test$imputed)
  cat(sprintf("Interpolated months inside the test block: %d%s\n", n_imp,
              if (n_imp > 0) {
                paste0(" (",
                       paste(format(parts$test$date[parts$test$imputed]),
                             collapse = ", "),
                       ") - scored against synthetic actuals")
              } else ""))
  invisible(parts)
}

## ---- The single MASE denominator --------------------------------------------
# MASE divides forecast MAE by the in-sample MAE of a naive forecast. That
# denominator depends entirely on the training window, so two models fitted on
# different windows produce MASE values that CANNOT be compared: a model trained
# from 1948 gets a much smaller denominator than one trained from 2010 and looks
# better for that reason alone. forecast::accuracy() computes it per-model and is
# therefore unsafe for cross-model comparison - use evaluate() instead.
#
# One denominator, computed on the full history up to the test start, applied to
# every row of the comparison table. Seasonal-naive is the conventional default;
# the non-seasonal figure is reported alongside it because this series is
# seasonally adjusted at source, which makes a seasonal-naive comparator
# artificially weak.
mase_denominators <- function(df = load_series()) {
  hist <- df$value[df$date < TEST_START]
  c(snaive = mean(abs(diff(hist, lag = FREQ))),
    naive  = mean(abs(diff(hist, lag = 1))))
}

MASE_DENOMS <- mase_denominators()
MASE_DENOM  <- unname(MASE_DENOMS["snaive"])

## ---- One accuracy function --------------------------------------------------
evaluate <- function(actual, forecast, denom = MASE_DENOM) {
  actual   <- as.numeric(actual)
  forecast <- as.numeric(forecast)
  stopifnot(length(actual) == length(forecast), length(actual) > 0)
  err <- actual - forecast
  c(RMSE = sqrt(mean(err^2)),
    MAE  = mean(abs(err)),
    MAPE = 100 * mean(abs(err / actual)),
    MASE = mean(abs(err)) / denom)
}

## ---- COVID intervention regressors ------------------------------------------
# Additive-outlier pulses for the four shock months. The SAME rule serves
# level-space and difference-space models; what differs is the date vector you
# hand it, not the function:
#   levels model      covid_dummies(dates of the level series)
#   differences model covid_dummies(dates of the differenced series)
# Pass future dates to build the forecast regressor matrix. All four shock months
# are historical, so those matrices are correctly all zero - the model still
# requires them to be supplied.
covid_dummies <- function(dates) {
  dates <- as.Date(dates)
  X <- vapply(COVID_SHOCK_MONTHS,
              function(s) as.numeric(dates == s),
              numeric(length(dates)))
  matrix(as.numeric(X), nrow = length(dates),
         dimnames = list(NULL, paste0("AO", format(COVID_SHOCK_MONTHS, "%y%m"))))
}

## ---- Identifiability guard --------------------------------------------------
# An over-parameterised ARIMA can converge to a point where the Hessian is
# singular. forecast::Arima then returns NaN standard errors, and that fit's
# coefficients, information criteria and prediction intervals are all unusable.
# The original SARIMA search accepted exactly such a model - SARIMA(4,1,3)(1,0,0)
# with NaN errors on five of nine coefficients - and it went on to top the table
# on test RMSE. Good holdout error does not make a degenerate model sound, so
# every candidate must clear this before it is allowed to be selected.
is_identifiable <- function(fit) {
  v <- try(suppressWarnings(sqrt(diag(fit$var.coef))), silent = TRUE)
  if (inherits(v, "try-error") || length(v) == 0) return(FALSE)
  all(is.finite(v))
}

## ---- Residual shape -----------------------------------------------------
# Reports the two numbers that decide whether Gaussian prediction intervals mean
# anything: the largest standardised residual and the excess kurtosis. Used to
# document the COVID trade-off in auto_arima.R and SARIMA.R, where treating the
# 2020 break cleans the residuals but costs point accuracy.
residual_summary <- function(fit) {
  r <- as.numeric(residuals(fit))
  r <- r[is.finite(r)]
  c(max_abs_z = max(abs(r / sd(r))),
    kurtosis  = mean((r - mean(r))^4) / sd(r)^4 - 3)
}

## ---- One result-row schema --------------------------------------------------
# Every model writes exactly one of these. compare_models.R concatenates them.
save_model_result <- function(model_id, model_name, window_start,
                              n_train, n_train_val,
                              val_metrics, test_metrics,
                              ljung_p = NA_real_, identifiable = NA,
                              aic = NA_real_, bic = NA_real_, notes = "") {
  row <- data.frame(
    Model        = model_name,
    Window       = format(as.Date(window_start)),
    N_train      = n_train,
    N_train_val  = n_train_val,
    Val_RMSE     = unname(val_metrics["RMSE"]),
    Val_MAE      = unname(val_metrics["MAE"]),
    Val_MAPE     = unname(val_metrics["MAPE"]),
    Test_RMSE    = unname(test_metrics["RMSE"]),
    Test_MAE     = unname(test_metrics["MAE"]),
    Test_MAPE    = unname(test_metrics["MAPE"]),
    Test_MASE    = unname(test_metrics["MASE"]),
    LjungBox_p   = unname(ljung_p),
    Identifiable = identifiable,
    AIC          = unname(aic),
    BIC          = unname(bic),
    Notes        = notes,
    stringsAsFactors = FALSE
  )
  write.csv(row, paste0(model_id, "_result.csv"), row.names = FALSE)
  row
}

## ---- Reproducibility --------------------------------------------------------
write_session_info <- function(path = "session_info.txt") {
  writeLines(c(paste("Generated:", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
               "", capture.output(sessionInfo())), path)
  cat("Wrote", path, "\n")
}

cat(sprintf("common.R loaded | val %s .. %s | test %s onward\n",
            format(VAL_START), format(TEST_START - 1), format(TEST_START)))
cat(sprintf("           MASE denominator (snaive) = %.2f   (naive = %.2f)\n",
            MASE_DENOMS["snaive"], MASE_DENOMS["naive"]))
