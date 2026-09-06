# ==============================================================================
# common.R - shared foundation for every model script in this project
#
# WHY THIS FILE EXISTS
# Each model script used to re-implement its own train/test split, its own
# accuracy function and its own MASE denominator. Those copies drifted apart, and
# the result was a comparison table whose rankings were an artefact of the drift
# rather than of model quality. Everything that must be identical across models
# is now defined here exactly once. Do not copy any of it back into a model
# script - including the benchmarks and the analysis-window constants, which used
# to be re-typed in ARX-GARCH.R, HoltWinter.R and rolling_cv.R.
#
# Usage, after data_processing.R has written the cleaned CSV:
#   source("common.R")
# ==============================================================================

## ---- Configuration ----------------------------------------------------------
# One split for every model. A script MAY start its analysis window later than
# the series does - Holt-Winters uses 2021+, ARX-GARCH uses 2010+, and those are
# legitimate modelling choices - but it may NOT move these boundaries. The
# validation and test blocks have to be the same twelve months everywhere or the
# comparison table is comparing different questions.
DATA_FILE  <- "processed_female_employment.csv"
VAL_START  <- as.Date("2024-08-01")   # validation block: 2024-08 .. 2025-07
TEST_START <- as.Date("2025-08-01")   # test block:       2025-08 .. 2026-07
FREQ       <- 12
HORIZON    <- 12                      # reported forecast horizon = test length

# Each model's analysis window lives here, not in the model script, because
# rolling_cv.R has to reproduce the same windows and a second copy is exactly how
# the earlier version of this project went wrong.
HW_WINDOW_START  <- as.Date("2021-01-01")   # Holt-Winters: starts after COVID
ARX_WINDOW_START <- as.Date("2010-01-01")   # ARX-GARCH: post-1948 growth regime

# End of the test block. Defined explicitly so that publishing one more month of
# data adds a row to `post` instead of breaking every script's split assertion.
# Add k months to a date. k may be NEGATIVE - seq() rejects a negative
# length.out, so the direction goes into `by` and the count is always positive.
month_add <- function(d, k) {
  d <- as.Date(d)
  if (k == 0) return(d)
  seq(d, by = paste(sign(k), "month"), length.out = abs(k) + 1)[abs(k) + 1]
}
TEST_END  <- month_add(TEST_START, HORIZON)   # first month AFTER the test block

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

  # as_monthly_ts() dates observations by COUNTING FORWARD from the first date at
  # frequency 12. It has no way to notice a missing row: one absent month would
  # silently shift every later observation back by one and every date printed
  # downstream would be wrong. data_processing.R interpolates missing VALUES but
  # nothing ever guaranteed the row grid itself was complete, so check it here.
  if (any(duplicated(df$date))) {
    stop("Duplicate dates in ", path, ": ",
         paste(format(unique(df$date[duplicated(df$date)])), collapse = ", "))
  }
  expected <- seq(min(df$date), max(df$date), by = "month")
  if (length(expected) != nrow(df) || !all(df$date == expected)) {
    gaps <- format(expected[!expected %in% df$date])
    stop("Date grid is not contiguous monthly - ", nrow(df), " rows for ",
         length(expected), " months. Missing: ",
         paste(utils::head(gaps, 12), collapse = ", "),
         if (length(gaps) > 12) ", ..." else "",
         "\nEvery ts() built from this would be mis-dated. Fix data_processing.R.")
  }
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
# Returns train / val / test / train_val / all as data frames, plus `post` for
# any months published after the test block closed. Fit candidates on train,
# score them on val, refit the winner on train_val, and touch test exactly once
# at the very end.
#
# `test` is bounded at both ends (TEST_START .. TEST_END). It used to be an
# open-ended ">= TEST_START" with an assertion that it was exactly HORIZON rows
# long, so the first month of new data BLS published would abort every script in
# the project. New months now land in `post` and are ignored by the scoring.
split_series <- function(df, window_start = NULL) {
  if (!is.null(window_start)) {
    df <- df[df$date >= as.Date(window_start), ]
    rownames(df) <- NULL
  }
  parts <- list(
    train     = df[df$date <  VAL_START, ],
    val       = df[df$date >= VAL_START  & df$date < TEST_START, ],
    test      = df[df$date >= TEST_START & df$date < TEST_END, ],
    train_val = df[df$date <  TEST_START, ],
    post      = df[df$date >= TEST_END, ],
    all       = df
  )
  stopifnot(nrow(parts$val) == HORIZON)
  if (nrow(parts$test) != HORIZON) {
    stop("Test block has ", nrow(parts$test), " months, expected ", HORIZON,
         ". The series ends ", format(max(df$date)),
         " - it does not yet cover the whole test window.")
  }
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
  if (nrow(parts$post)) {
    fmt(parts$post, "Post-test", "<- published after the test window; NOT scored")
  }
  n_imp <- sum(parts$test$imputed)
  cat(sprintf("Interpolated months inside the test block: %d%s\n", n_imp,
              if (n_imp > 0) {
                paste0(" (",
                       paste(format(parts$test$date[parts$test$imputed]),
                             collapse = ", "),
                       ") - EXCLUDED from every score")
              } else ""))
  invisible(parts)
}

## ---- The MASE denominators --------------------------------------------------
# MASE divides forecast MAE by the in-sample MAE of a naive forecast. That
# denominator depends entirely on the training window, so two models fitted on
# different windows produce MASE values that CANNOT be compared. One denominator,
# computed on the full history up to the test start, is applied to every row of
# the comparison table. forecast::accuracy() computes it per-model and is
# therefore unsafe for cross-model comparison - use evaluate() instead.
#
# WHICH DENOMINATOR IS THE HEADLINE. The lag-1 one. LNS12000002 is seasonally
# adjusted at source, so Hyndman's non-seasonal convention (mean |y_t - y_{t-1}|)
# is the right scaling, and it is what MASE means here.
#
# The seasonal-naive denominator is reported alongside it as MASE_s, but it must
# NOT be quoted as "the" MASE on this series: on a strongly trending series a
# 12-month difference is dominated by twelve months of drift rather than by
# seasonality, which makes that denominator about 5.6x larger and every model
# look about 5.6x better. Measured on this data:
#
#   denominator            value    auto.arima test MASE
#   lag-1 naive (headline)  186.8   1.22  -> WORSE than a naive forecast
#   seasonal naive          1041.5  0.22  -> "4.5x better than naive"
#
# The earlier version of this file computed both and reported only the flattering
# one, which reversed the qualitative conclusion. Both are now in every table.
mase_denominators <- function(df = load_series()) {
  hist <- df$value[df$date < TEST_START]
  c(naive  = mean(abs(diff(hist, lag = 1))),
    snaive = mean(abs(diff(hist, lag = FREQ))))
}

# Load the series ONCE at source time. Everything below that needs the full
# history - the MASE denominators and the shared benchmarks - reads this rather
# than re-reading the CSV, and it guarantees they all describe the same data.
SHARED_SERIES <- load_series()
SHARED_PARTS  <- split_series(SHARED_SERIES)

MASE_DENOMS  <- mase_denominators(SHARED_SERIES)
MASE_DENOM   <- unname(MASE_DENOMS["naive"])    # headline scaling
MASE_DENOM_S <- unname(MASE_DENOMS["snaive"])   # reported alongside, never alone

## ---- One accuracy function --------------------------------------------------
# `exclude` drops points from the score - pass the imputed flag so models are not
# graded against interpolated values. 2025-10 is blank at source and filled by
# linear interpolation between September and November; because it sits almost
# exactly on the straight line between its neighbours, any smoothly-trending
# forecast passes close to it and earns an artificially small error. It is
# therefore not a neutral distortion: it tilts the ranking toward smooth-trend
# models. Score against observed values only.
#
# EVERY score in this project - models AND benchmarks - must pass the same
# `exclude`. Scoring a model on 11 points and its benchmark on 12 makes the
# comparison meaningless, which is what ARX-GARCH.R used to do.
evaluate <- function(actual, forecast, denoms = MASE_DENOMS, exclude = NULL) {
  actual   <- as.numeric(actual)
  forecast <- as.numeric(forecast)
  stopifnot(length(actual) == length(forecast), length(actual) > 0)

  if (!is.null(exclude)) {
    keep <- !as.logical(exclude)
    stopifnot(length(keep) == length(actual), any(keep))
    actual   <- actual[keep]
    forecast <- forecast[keep]
  }

  err <- actual - forecast
  c(RMSE   = sqrt(mean(err^2)),
    MAE    = mean(abs(err)),
    MAPE   = 100 * mean(abs(err / actual)),
    MASE   = mean(abs(err)) / unname(denoms["naive"]),
    MASE_s = mean(abs(err)) / unname(denoms["snaive"]),
    N      = length(err))
}

## ---- Benchmarks -------------------------------------------------------------
# Defined once, here, so that "the random-walk-with-drift benchmark" means one
# thing in this project. It previously meant two: compare_models.R estimated the
# drift over the full 1948+ history (64.99/month) while ARX-GARCH.R estimated it
# over its own 2010+ window (58.89/month) and scored it on 12 points against a
# model scored on 11. The same claim therefore came out as -8.2% in one file and
# -10.6% in the other.
#
# The drift is estimated over the FULL history up to the test start, because the
# benchmark has to be one fixed comparator for models fitted on three different
# windows. A model that wants to show its own window-local drift may compute one,
# but must label it as such and must never call it "the benchmark".
# NOTE THE MISSING ARGUMENT. These take no `parts`, on purpose. They read
# SHARED_PARTS, which is always the FULL-history split, so a script that sliced
# its own analysis window cannot accidentally benchmark itself against a drift
# estimated over that same short window. Passing a windowed split here was the
# original bug: Holt-Winters would have compared itself against a 2021+ drift of
# 422 RMSE while compare_models.R used the full-history drift of 322, and both
# would have been called "the RW-with-drift benchmark".
benchmark_forecasts <- function(h = HORIZON) {
  tv       <- SHARED_PARTS$train_val$value
  last_obs <- tv[length(tv)]
  list(
    `Naive (last value)` = rep(last_obs, h),
    `Seasonal naive`     = utils::tail(tv, h),
    `RW with drift`      = last_obs +
      (last_obs - tv[1]) / (length(tv) - 1) * seq_len(h)
  )
}

benchmark_table <- function() {
  fcs <- benchmark_forecasts()
  do.call(rbind, lapply(fcs, function(f)
    evaluate(SHARED_PARTS$test$value, f,
             exclude = SHARED_PARTS$test$imputed)))
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
# Every candidate must clear this before it is allowed to be selected.
#
# A model with NO estimated coefficients - ARIMA(0,1,0) without drift - has an
# empty var.coef and used to be reported as unidentifiable. There is nothing to
# be unidentifiable about; it passes.
is_identifiable <- function(fit) {
  v <- try(suppressWarnings(sqrt(diag(fit$var.coef))), silent = TRUE)
  if (inherits(v, "try-error")) return(FALSE)
  if (length(v) == 0) return(TRUE)     # no free coefficients: nothing to check
  all(is.finite(v))
}

## ---- Residual shape ---------------------------------------------------------
# Reports the two numbers that decide whether Gaussian prediction intervals mean
# anything: the largest standardised residual and the excess kurtosis.
residual_summary <- function(fit) {
  r <- as.numeric(residuals(fit))
  r <- r[is.finite(r)]
  c(max_abs_z = max(abs(r / sd(r))),
    kurtosis  = mean((r - mean(r))^4) / sd(r)^4 - 3)
}

# Share of the total squared residual carried by the single largest one.
#
# WHY THIS EXISTS. A Ljung-Box test on squared residuals is supposed to detect
# leftover ARCH structure, but it is destroyed by a single dominant outlier: when
# one point carries most of the sum of squares, the sample autocorrelations of
# the squared series are driven to zero and the test returns p ~ 1 no matter what
# the rest of the series does. The ARX-GARCH search hit exactly this - the nine
# specifications WITHOUT COVID dummies had max|z| between 8 and 40 and "passed"
# the ARCH diagnostic at p = 0.99999, while the nine well-behaved dummied
# specifications (max|z| ~ 5.5) were correctly flagged at p ~ 0.003 and rejected.
# The rule was admitting the worst fits and rejecting the best.
#
# On n observations a well-behaved fit spreads its squared residuals out, so this
# share is O(1/n) inflated by the tail - about 0.15 here. A single 40-sigma
# residual takes it to about 0.90. The gate is a sanity check on whether the
# ARCH test had any power at all, not a test in its own right.
z2_share <- function(z) {
  z <- as.numeric(z)
  z <- z[is.finite(z)]
  max(z^2) / sum(z^2)
}

## ---- One Ljung-Box convention -----------------------------------------------
# The comparison table used to print a LjungBox_p column whose rows were not
# comparable: the ARIMA rows were lag 24 / df 4 on 930 level residuals, the
# Holt-Winters row was lag 11 / df 0 on 55 level residuals, and the ARX-GARCH row
# was lag 24 / df 0 on 186 standardised residuals of the DIFFERENCED series.
# Three different tests under one heading.
#
# They cannot be made identical - the models have different sample sizes and
# different parameter counts - so instead every caller records what it actually
# ran, and compare_models.R prints the lag, the df and the residual space next to
# the p-value.
ljung_box <- function(resid, fitdf, lag = 2 * FREQ, on = "residuals") {
  r <- as.numeric(resid)
  r <- r[is.finite(r)]
  lag <- min(lag, floor(length(r) / 5))
  lag <- max(lag, fitdf + 3)
  lag <- min(lag, length(r) - 1)
  if (lag <= fitdf) {
    return(list(p = NA_real_, lag = lag, fitdf = fitdf, df = NA_integer_,
                on = on, n = length(r)))
  }
  bt <- Box.test(r, lag = lag, type = "Ljung-Box", fitdf = fitdf)
  # `fitdf` is what was DEDUCTED for estimated parameters; `df` is the
  # chi-squared degrees of freedom the p-value is read against (lag - fitdf).
  # Reporting only one of them made the Holt-Winters diagnostic table label its
  # chi-square df as "fitdf" and print 11 and 8 where 0 and 3 were meant.
  list(p = unname(bt$p.value), lag = lag, fitdf = fitdf,
       df = unname(bt$parameter), on = on, n = length(r))
}

## ---- One result-row schema --------------------------------------------------
# Every model writes exactly one of these. compare_models.R concatenates them.
#
# `spec` is a machine-readable encoding of the SELECTED specification, e.g.
#   "kind=arima;p=2;d=1;q=2;P=0;D=0;Q=0;drift=TRUE"
#   "kind=arx;ar=0;dist=std;garch=TRUE;dummies=TRUE"
# rolling_cv.R reads it instead of keeping a second copy of every model's chosen
# orders. Re-run a search that selects something else and the rolling evaluation
# follows automatically instead of silently scoring the old specification.
save_model_result <- function(model_id, model_name, window_start,
                              n_train, n_train_val,
                              val_metrics, test_metrics,
                              lb = NULL, identifiable = NA,
                              aic = NA_real_, bic = NA_real_,
                              spec = NA_character_, notes = "") {
  if (is.null(lb)) lb <- list(p = NA_real_, lag = NA_integer_,
                              fitdf = NA_integer_, df = NA_integer_,
                              on = NA_character_)
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
    Test_MASE_s  = unname(test_metrics["MASE_s"]),
    N_scored     = unname(test_metrics["N"]),
    LjungBox_p     = unname(lb$p),
    LjungBox_lag   = unname(lb$lag),
    LjungBox_fitdf = unname(lb$fitdf),
    LjungBox_df    = unname(lb$df),
    LjungBox_on    = unname(lb$on),
    Identifiable = identifiable,
    AIC          = unname(aic),
    BIC          = unname(bic),
    Spec         = spec,
    Notes        = notes,
    stringsAsFactors = FALSE
  )
  write.csv(row, paste0(model_id, "_result.csv"), row.names = FALSE)
  row
}

# forecast::arimaorder() returns THREE elements for a non-seasonal fit and SEVEN
# for a seasonal one, so indexing positions 4:6 on a non-seasonal model silently
# yields NA - which then propagates into a Ljung-Box fitdf and aborts the script
# with "missing value where TRUE/FALSE needed". Always go through this.
arima_orders <- function(fit) {
  o <- forecast::arimaorder(fit)
  if (length(o) < 6) o <- c(o[1:3], 0, 0, 0)
  setNames(as.integer(o[1:6]), c("p", "d", "q", "P", "D", "Q"))
}

# Degrees of freedom to deduct in a Ljung-Box test of ARIMA residuals: the ARMA
# terms only. This matches forecast::checkresiduals(), which does not count the
# drift term.
arima_fitdf <- function(fit) {
  o <- arima_orders(fit)
  unname(o["p"] + o["q"] + o["P"] + o["Q"])
}

# Encode a fitted forecast::Arima as a spec string, so the orders that were
# actually selected travel to rolling_cv.R instead of being re-typed there.
arima_spec <- function(fit) {
  o <- arima_orders(fit)
  sprintf("kind=arima;p=%d;d=%d;q=%d;P=%d;D=%d;Q=%d;drift=%s",
          o["p"], o["d"], o["q"], o["P"], o["D"], o["Q"],
          "drift" %in% names(coef(fit)))
}

# Parse a `spec` string back into a named list. "TRUE"/"FALSE" and pure integers
# are converted; everything else stays character.
parse_spec <- function(s) {
  if (length(s) != 1 || is.na(s) || !nzchar(s)) stop("Empty model spec.")
  kv <- strsplit(strsplit(s, ";", fixed = TRUE)[[1]], "=", fixed = TRUE)
  out <- lapply(kv, function(p) {
    v <- p[2]
    if (v %in% c("TRUE", "FALSE")) as.logical(v)
    else if (grepl("^-?[0-9]+$", v)) as.integer(v)
    else v
  })
  names(out) <- vapply(kv, `[`, character(1), 1)
  out
}

## ---- Reproducibility --------------------------------------------------------
write_session_info <- function(path = "session_info.txt") {
  writeLines(c(paste("Generated:", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
               "", capture.output(sessionInfo())), path)
  cat("Wrote", path, "\n")
}

cat(sprintf("common.R loaded | val %s .. %s | test %s .. %s\n",
            format(VAL_START), format(TEST_START - 1),
            format(TEST_START), format(TEST_END - 1)))
cat(sprintf("           MASE  denominator = %8.2f  (lag-1 naive - HEADLINE)\n",
            MASE_DENOM))
cat(sprintf("           MASE_s denominator = %8.2f  (seasonal naive - reported, never quoted alone)\n",
            MASE_DENOM_S))
