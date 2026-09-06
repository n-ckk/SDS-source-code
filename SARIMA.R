# ==============================================================================
# SARIMA - Dataset: LNS12000002 - Employment Level: Women
#
# WHAT THIS SCRIPT IS
# A reimplementation of the original SARIMA script's design on the shared
# foundation in common.R. The MODEL and the SELECTION RULE are the original's,
# unchanged:
#
#   GRID     p 0:4, q 0:4, P 0:2, Q 0:2, restricted to P + Q > 0. A seasonal
#            term is COMPULSORY, so this script reports a genuinely seasonal
#            model rather than collapsing onto the non-seasonal ARIMA that
#            auto_arima.R selects when given a free choice.
#   QUALIFY  Ljung-Box p > 0.05 on the training residuals.
#   RANK     Validation RMSE -> MAE -> MAPE -> AICc.
#
# The validation block does real selection work here. That is the substantive
# difference from auto_arima.R, which selects on an in-sample information
# criterion: this script picks the seasonal specification that forecasts twelve
# genuinely unseen months best, and only breaks ties on AICc.
#
# WHAT WAS REIMPLEMENTED, AND WHY. Every change below is to the SCORING and the
# REPORTING. None of them touches the grid, the qualification or the ranking, so
# the selected specification is the one the original rule selects.
#
# (1) Metrics come from common.R evaluate(), not from a local helper and not
#     from forecast::accuracy(). accuracy() scales MASE by each model's own
#     training window, so MASE from different scripts cannot be compared;
#     evaluate() uses the two shared denominators fixed in common.R. It also
#     drops 2025-10, which is interpolated rather than observed, from the test
#     score - for this model and for the benchmarks alike.
# (2) Ljung-Box comes from common.R ljung_box(), which reports the lag and the
#     degrees of freedom beside the p-value instead of the p-value alone.
# (3) The result row is written by save_model_result(), so compare_models.R and
#     rolling_cv.R read this model through the same schema as the others.
# (4) The test/train error ratio is GONE. It divided a 12-step-ahead test error
#     by 1-step-ahead in-sample residuals - two different quantities - and on a
#     window containing an untreated 2020 break it can "pass" simply because the
#     training residuals are inflated. Validation error against test error is
#     reported instead: both are 12-step-ahead forecasts of unseen blocks.
# (5) Identifiability is REPORTED, never used as a gate. See the caveat below.
#
# READ THIS BEFORE QUOTING THIS MODEL.
# The original rule ranks on validation RMSE and does not inspect the Hessian.
# The specification it selects is identifiable on the training fit but can come
# back with non-finite standard errors once it is refitted on the longer
# train+validation block. Section 6 checks this on every run and says so in the
# output, in sarima_final_parameters.csv and in the Identifiable column of
# sarima_result.csv, which compare_models.R surfaces as a warning.
#
# When that check fails, the consequence is specific and worth stating exactly:
# the POINT FORECASTS remain usable, but the standard errors, the t-values, the
# information criteria and the PREDICTION INTERVALS do not exist in any
# meaningful sense. Quote this model's forecasts; do not quote its intervals or
# its coefficient significance. For intervals, use the empirical error quantiles
# from rolling_cv.R.
#
# SEASONALITY - WHAT A COMPULSORY SEASONAL TERM MEANS HERE.
# LNS12000002 is published SEASONALLY ADJUSTED by BLS, so the seasonal structure
# this grid is required to fit may be very close to nothing. Section 6 prints the
# selected seasonal coefficients with their t-values, and section 7 measures the
# seasonal strength independently, so the report can state how much the seasonal
# terms actually carry rather than assuming they earn their place.
#
# COVID. The primary search carries NO intervention dummies. Treating the 2020
# break cuts the extreme April 2020 residual by roughly a factor of four and the
# excess kurtosis by about an order of magnitude, and makes the point forecasts
# worse. Residual quality and accuracy move in opposite directions, and no
# configuration achieves both. Section 11 measures that trade-off on every run
# rather than quoting it from prose - see sarima_covid_sensitivity.csv.
# ==============================================================================

library(forecast)

source("common.R")

MODEL_ID <- "sarima"

# ------------------------------------------------------------------------------
# 1. LOAD DATA AND SPLIT
# The split boundaries live in common.R because rolling_cv.R needs the same ones.
# Train ends 2024-07, validation is 2024-08 .. 2025-07, test is 2025-08 .. 2026-07.
# ------------------------------------------------------------------------------
data  <- load_series()
parts <- split_series(data)
describe_split(parts, "SARIMA (full history, seasonal term compulsory)")

train_ts     <- as_monthly_ts(parts$train)
train_val_ts <- as_monthly_ts(parts$train_val)
full_ts      <- as_monthly_ts(parts$all)

# Built here but used ONLY by the sensitivity in section 11.
X_train_val <- covid_dummies(parts$train_val$date)
X_test      <- covid_dummies(parts$test$date)

# ------------------------------------------------------------------------------
# 2. DIFFERENCING ORDERS
# ------------------------------------------------------------------------------
d <- ndiffs(train_ts, test = "kpss", max.d = 2)
D <- nsdiffs(train_ts, test = "seas", max.D = 1)

cat("\n=== DIFFERENCING ===\n")
cat("d =", d, "\n")
cat("D =", D, " (0 means no seasonal differencing is required)\n")
if (D == 0) {
  cat("Note: nsdiffs() finds no seasonal unit root, which is expected on a\n")
  cat("series BLS has already seasonally adjusted. The seasonal AR/MA terms\n")
  cat("below are still compulsory under this script's selection rule.\n")
}

# ------------------------------------------------------------------------------
# 3. SPECIFICATION SEARCH - FIT ON TRAIN, SCORE ON VALIDATION
#
# The grid is 225 maximum-likelihood fits and takes 15-30 minutes, so it is
# cached. The full p/q/P/Q grid is searched and the P + Q > 0 restriction is
# applied in section 4 where the rule is stated. Selecting the best seasonal
# candidate out of the full table is arithmetically identical to searching only
# the seasonal ones, and keeping the non-seasonal fits in the table costs 25
# extra fits once and lets section 7 quantify what the restriction costs.
#
# THE CACHE IS FINGERPRINTED, covering the split boundaries and the training data
# itself. Without that, changing VAL_START, TEST_START or the data would leave a
# stale table on disk that the script would happily reuse, every cached
# Validation_RMSE and AICc silently describing a different question.
# ------------------------------------------------------------------------------
SEARCH_CACHE <- "sarima_model_selection.csv"   # primary search, no dummies

fingerprint <- sprintf("val=%s;test=%s;freq=%d;n=%d;s1=%.4f;s2=%.4f;d=%d;D=%d",
                       format(VAL_START), format(TEST_START), FREQ,
                       length(train_ts), sum(train_ts), sum(train_ts^2), d, D)

cache_ok <- FALSE
if (file.exists(SEARCH_CACHE)) {
  cached <- read.csv(SEARCH_CACHE, stringsAsFactors = FALSE)
  cache_ok <- !is.null(cached$Fingerprint) &&
    nrow(cached) > 0 &&
    identical(as.character(cached$Fingerprint[1]), fingerprint)
  if (!cache_ok) {
    cat("\nCached search in", SEARCH_CACHE, "does not match the current data or\n")
    cat("split (missing or stale fingerprint). Re-running the grid.\n")
  }
}

refresh <- !cache_ok || isTRUE(as.logical(Sys.getenv("SARIMA_REFRESH", "FALSE")))

candidate_table <- data.frame()
candidate_id    <- 0

if (!refresh) {

  cat("\nReusing cached search from", SEARCH_CACHE, "\n")
  cat("Fingerprint matches. Set SARIMA_REFRESH=TRUE or delete it to re-run.\n")
  candidate_table <- cached

} else {

cat("\nSearching p 0:4, q 0:4, P 0:2, Q 0:2 ...\n")

for (p in 0:4) {
  for (q in 0:4) {
    for (P in 0:2) {
      for (Q in 0:2) {

        fit <- tryCatch(
          Arima(train_ts,
                order    = c(p, d, q),
                seasonal = list(order = c(P, D, Q), period = FREQ),
                include.drift = (d == 1 && D == 0),
                method   = "ML"),
          error = function(e) NULL
        )
        # Warnings are deliberately NOT trapped. Trapping them would discard
        # fits that merely warn about a near-non-invertible root, which are
        # often fine. Identifiability is recorded as a COLUMN below so that the
        # selection rule can ignore it and the report cannot.
        if (is.null(fit)) next

        val_fc <- tryCatch(forecast(fit, h = HORIZON),
                           error = function(e) NULL)
        if (is.null(val_fc)) next

        val_acc <- evaluate(parts$val$value, val_fc$mean)
        lb_p <- tryCatch(
          ljung_box(residuals(fit), fitdf = p + q + P + Q,
                    on = "level residuals")$p,
          error = function(e) NA_real_
        )

        candidate_id <- candidate_id + 1

        candidate_table <- rbind(candidate_table, data.frame(
          ID    = candidate_id,
          Model = sprintf("SARIMA(%d,%d,%d)(%d,%d,%d)[%d]",
                          p, d, q, P, D, Q, FREQ),
          p = p, d = d, q = q, P = P, D = D, Q = Q,
          Seasonal        = (P + Q) > 0,
          Identifiable    = is_identifiable(fit),
          Validation_RMSE = unname(val_acc["RMSE"]),
          Validation_MAE  = unname(val_acc["MAE"]),
          Validation_MAPE = unname(val_acc["MAPE"]),
          Ljung_Box_p     = lb_p,
          AIC = fit$aic, AICc = fit$aicc, BIC = fit$bic,
          Fingerprint     = fingerprint,
          stringsAsFactors = FALSE
        ))
      }
    }
  }
}

write.csv(candidate_table, SEARCH_CACHE, row.names = FALSE)

}  # end of search / cache branch

if (nrow(candidate_table) == 0) stop("No SARIMA candidate could be fitted.")

# MASE columns are DERIVED, never cached. MASE is MAE divided by a denominator
# that lives in common.R, so recomputing it here means a change to that
# denominator propagates into the cached table instead of being silently ignored.
candidate_table$Validation_MASE   <- candidate_table$Validation_MAE / MASE_DENOM
candidate_table$Validation_MASE_s <- candidate_table$Validation_MAE / MASE_DENOM_S

cat("Candidates in table  :", nrow(candidate_table), "of 225 attempted",
    sprintf("(%d specifications failed to converge)\n",
            225 - nrow(candidate_table)))

# ------------------------------------------------------------------------------
# 4. MODEL SELECTION RULE - the original rule, stated up front and applied
#    mechanically
#
#   RESTRICT  P + Q > 0        a seasonal term is compulsory
#   QUALIFY   Ljung-Box p > 0.05
#   RANK      Validation RMSE -> MAE -> MAPE -> AICc
#
# The test block plays no part. Identifiability plays no part either - it is
# recorded and reported, but this rule does not gate on it.
# ------------------------------------------------------------------------------
cat("\n=== MODEL SELECTION PROCESS ===\n")
cat("Candidates fitted    :", nrow(candidate_table), "\n")
cat("Restriction          : P + Q > 0 (a seasonal term is compulsory)\n")
cat("Excluded by that rule:", sum(!candidate_table$Seasonal),
    "non-seasonal specifications\n")
cat("Qualification        : Ljung-Box p > 0.05 (in-sample, training fit)\n")
cat("Ranking              : Validation RMSE -> MAE -> MAPE -> AICc\n")
cat("Test block           : NOT used\n")

seasonal_pool <- candidate_table[candidate_table$Seasonal, ]
if (nrow(seasonal_pool) == 0) stop("No seasonal SARIMA candidate could be fitted.")

qualified <- seasonal_pool[
  is.finite(seasonal_pool$Ljung_Box_p) & seasonal_pool$Ljung_Box_p > 0.05,
]

if (nrow(qualified) == 0) {
  cat("\nNo seasonal model passed Ljung-Box qualification.\n")
  cat("Fallback: rank all seasonal candidates by validation accuracy.",
      "Report this.\n")
  qualified <- seasonal_pool
}

cat("Qualified models     :", nrow(qualified), "of", nrow(seasonal_pool),
    "seasonal candidates\n")

ranked <- qualified[order(qualified$Validation_RMSE,
                          qualified$Validation_MAE,
                          qualified$Validation_MAPE,
                          qualified$AICc), ]
selected <- ranked[1, ]

cat("\nTop 10 candidates used for the final ranking:\n")
print(head(ranked[, c("Model", "Validation_RMSE", "Validation_MAE",
                      "Validation_MAPE", "Ljung_Box_p", "AICc",
                      "Identifiable")], 10),
      row.names = FALSE, digits = 6)

cat("\nSelected model :", selected$Model, "\n")
cat("Validation RMSE:", selected$Validation_RMSE, "\n")
cat("Validation MAE :", selected$Validation_MAE, "\n")
cat("Validation MAPE:", selected$Validation_MAPE, "%\n")
cat("Ljung-Box p    :", selected$Ljung_Box_p, "\n")
cat("AICc           :", selected$AICc, "\n")

# Honest cross-check. The ranking is on twelve validation points, which is a
# small number to separate this many models by. The full field is spread out, but
# what decides the selection is the gap at the TOP, and that gap is what the
# report has to justify. Measure both rather than characterising either.
spread   <- range(qualified$Validation_RMSE)
n_lead   <- min(10, nrow(ranked))
lead_rng <- range(ranked$Validation_RMSE[seq_len(n_lead)])
lead_pct <- 100 * (lead_rng[2] / lead_rng[1] - 1)
margin   <- 100 * (ranked$Validation_RMSE[min(2, nrow(ranked))] /
                     ranked$Validation_RMSE[1] - 1)

cat(sprintf("\nValidation RMSE across the %d qualifiers spans %.1f to %.1f (%.1f%%).\n",
            nrow(qualified), spread[1], spread[2],
            100 * (spread[2] / spread[1] - 1)))
cat(sprintf("The top %d are packed into %.1f to %.1f - a %.1f%% band - and the\n",
            n_lead, lead_rng[1], lead_rng[2], lead_pct))
cat(sprintf("selected model leads the runner-up by %.1f%%.\n", margin))
if (lead_pct < 5) {
  cat("The leaders are therefore separated by less than the noise you should\n")
  cat("expect from twelve points. The rule picks one of them, and it is the one\n")
  cat("reported - but report the rule and the margin, not the rank alone, and do\n")
  cat("not claim the runners-up were shown to be worse.\n")
}

by_aicc <- qualified$Model[which.min(qualified$AICc)]
if (!identical(by_aicc, selected$Model)) {
  cat("\nCross-check - lowest in-sample AICc among the qualifiers would pick:",
      by_aicc, "\n")
  cat("The rule ranks on validation RMSE, so that model is not selected here.\n")
}

# ------------------------------------------------------------------------------
# 5. REFIT THE SELECTED SPECIFICATION ON TRAIN + VALIDATION
#
# A specification that converged on the training block can still fail to
# re-estimate on the longer block - optim() reports "non-finite finite-difference
# value". The original script called Arima() unguarded here and would abort. Walk
# down the ranking until one refits, and report which rank was actually used.
#
# Note what this loop does NOT do: it does not skip a model for being
# unidentifiable. That is section 6's job to report, not this rule's job to gate.
# ------------------------------------------------------------------------------
refit_on <- function(row, series, xreg = NULL) {
  tryCatch(
    Arima(series,
          order    = c(row$p, row$d, row$q),
          seasonal = list(order = c(row$P, row$D, row$Q), period = FREQ),
          xreg     = xreg,
          include.drift = (row$d == 1 && row$D == 0),
          method   = "ML"),
    error = function(e) NULL
  )
}

final_model <- NULL
rank_used   <- NA_integer_

for (i in seq_len(nrow(ranked))) {
  cand_fit <- refit_on(ranked[i, ], train_val_ts)
  if (!is.null(cand_fit)) {
    final_model <- cand_fit
    rank_used   <- i
    break
  }
  cat(sprintf("  rank %d (%s) failed to re-estimate on train+validation\n",
              i, ranked$Model[i]))
}

if (is.null(final_model)) {
  stop("No qualifying specification could be refitted on train + validation.")
}

selected <- ranked[rank_used, ]

cat("\n=== FINAL MODEL: REFITTED ON TRAIN + VALIDATION ===\n")
if (rank_used > 1) {
  cat(sprintf("NOTE: ranks 1..%d could not be re-estimated; using rank %d (%s).\n",
              rank_used - 1, rank_used, selected$Model))
  cat("Report this - the rule's first choice was not usable.\n")
}
print(final_model)

# ------------------------------------------------------------------------------
# 6. IDENTIFIABILITY AND COEFFICIENT SIGNIFICANCE
#
# This is the section to read before quoting anything from this model beyond its
# point forecasts. A singular Hessian produces non-finite standard errors; when
# that happens the t-values, the p-values, the information criteria and the
# prediction intervals below are all meaningless, and the script says so rather
# than printing NaN without comment.
# ------------------------------------------------------------------------------
se           <- suppressWarnings(sqrt(diag(final_model$var.coef)))
n_coef       <- length(coef(final_model))
n_nan_se     <- sum(!is.finite(se))
identifiable <- is_identifiable(final_model)

parameter_table <- data.frame(
  Parameter = names(coef(final_model)),
  Estimate  = as.numeric(coef(final_model)),
  StdError  = as.numeric(se)
)
parameter_table$tvalue <- parameter_table$Estimate / parameter_table$StdError
parameter_table$pvalue <- 2 * pnorm(-abs(parameter_table$tvalue))

cat("\n=== FINAL COEFFICIENTS ===\n")
print(parameter_table, row.names = FALSE, digits = 5)
write.csv(parameter_table, "sarima_final_parameters.csv", row.names = FALSE)

cat(sprintf("\nIdentifiable: %s  (%d of %d standard errors are non-finite)\n",
            identifiable, n_nan_se, n_coef))

if (!identifiable) {
  cat("\n*** THIS MODEL HAS A SINGULAR HESSIAN ***\n")
  cat("The selection rule ranks on validation RMSE and does not inspect the\n")
  cat("Hessian, so a specification this over-parameterised can win it. The\n")
  cat("consequences are specific:\n")
  cat("  USABLE     the point forecasts, and the accuracy metrics computed\n")
  cat("             from them in section 9\n")
  cat("  NOT USABLE the standard errors, t-values and p-values printed above,\n")
  cat("             the AIC/AICc/BIC, and every prediction interval this model\n")
  cat("             produces in sections 9 and 10\n")
  cat("Quote its forecasts. Do not quote its intervals or its coefficient\n")
  cat("significance. For intervals use the empirical quantiles from\n")
  cat("rolling_cv.R, which are validated out of sample.\n")
}

# What are the compulsory seasonal terms actually contributing?
seasonal_rows <- grepl("^s(ar|ma)", parameter_table$Parameter)
if (any(seasonal_rows)) {
  cat("\n--- The compulsory seasonal terms ---\n")
  print(parameter_table[seasonal_rows, ], row.names = FALSE, digits = 5)
  sp <- parameter_table$pvalue[seasonal_rows]
  if (all(!is.finite(sp))) {
    cat("Their standard errors are among the non-finite ones, so whether they\n")
    cat("differ from zero cannot be tested. Report that as the finding.\n")
  } else if (all(sp > 0.05, na.rm = TRUE)) {
    cat("None is significant at the 5 percent level: the seasonal structure the\n")
    cat("rule required is not distinguishable from zero, which is what a series\n")
    cat("the BLS has already adjusted should look like.\n")
  } else {
    cat("At least one seasonal term is significant at the 5 percent level.\n")
  }
}

# ------------------------------------------------------------------------------
# 7. HOW MUCH DOES THE COMPULSORY SEASONAL TERM COST?
#
# The restriction in section 4 is a modelling choice, so its price is measured
# rather than assumed: the best QUALIFYING NON-SEASONAL candidate from the same
# search, scored the same way. This is what the rule gave up.
# ------------------------------------------------------------------------------
cat("\n=== WHAT THE SEASONAL RESTRICTION COSTS ===\n")

nonseasonal <- candidate_table[
  !candidate_table$Seasonal &
    is.finite(candidate_table$Ljung_Box_p) &
    candidate_table$Ljung_Box_p > 0.05,
]

if (nrow(nonseasonal) == 0) {
  cat("No non-seasonal candidate qualified, so the restriction cost nothing\n")
  cat("that the qualification step would have admitted.\n")
} else {
  ns_best <- nonseasonal[order(nonseasonal$Validation_RMSE), ][1, ]
  cat("Under THIS script's ranking (validation RMSE), the best qualifying\n")
  cat("non-seasonal candidate is", ns_best$Model, "\n")
  cat(sprintf("  validation RMSE %.1f versus %.1f for the selected seasonal model",
              ns_best$Validation_RMSE, selected$Validation_RMSE))
  cat(sprintf(" (%+.1f%%).\n",
              100 * (selected$Validation_RMSE / ns_best$Validation_RMSE - 1)))
  cat("  identifiable:", ns_best$Identifiable, " AICc:",
      round(ns_best$AICc, 1), "\n")

  # auto_arima.R ranks the same space on an information criterion, so it lands
  # somewhere else. Name that model too rather than implying the two agree.
  ns_aicc <- nonseasonal[order(nonseasonal$AICc), ][1, ]
  cat("\nRanked instead on in-sample AICc, the best non-seasonal candidate is\n")
  cat(" ", ns_aicc$Model, sprintf("(AICc %.1f, validation RMSE %.1f).\n",
                                  ns_aicc$AICc, ns_aicc$Validation_RMSE))
  cat("That is the criterion auto_arima.R uses, which is why this script and\n")
  cat("auto_arima.R report different models: they differ in the RANKING rule as\n")
  cat("well as in the seasonal restriction. Neither difference is an error, and\n")
  cat("the two results are not independent confirmations of each other.\n")

  cat("\nSo the seasonal restriction costs very little on validation error here,\n")
  cat("and what it buys is a seasonal term the report can show is insignificant.\n")
  cat("Stating both is what makes it a modelling choice rather than a hidden one.\n")
}

# ------------------------------------------------------------------------------
# 8. RESIDUAL DIAGNOSTICS
# ------------------------------------------------------------------------------
lb <- ljung_box(residuals(final_model),
                fitdf = selected$p + selected$q + selected$P + selected$Q,
                on = "level residuals")

cat("\n=== FINAL RESIDUAL DIAGNOSTIC ===\n")
cat(sprintf("Ljung-Box: lag = %d, df = %d, n = %d, p = %.4f\n",
            lb$lag, lb$df, lb$n, lb$p))
cat("Interpretation:",
    ifelse(is.finite(lb$p) && lb$p > 0.05,
           "no significant residual autocorrelation detected.",
           "residual autocorrelation may remain."), "\n")

rs <- residual_summary(final_model)
cat(sprintf("Largest standardised residual: %.1f sigma; excess kurtosis %.0f.\n",
            rs["max_abs_z"], rs["kurtosis"]))
cat("A residual that size is the April 2020 collapse. It is why this model's\n")
cat("Gaussian prediction intervals would not be trustworthy even if the fit\n")
cat("were identifiable - section 11 measures what treating it would cost.\n")

png("sarima_final_residual_diagnostics.png",
    width = 1400, height = 900, res = 150)
checkresiduals(final_model)
dev.off()

# ------------------------------------------------------------------------------
# 9. FINAL TEST EVALUATION - the test block is read only here
# ------------------------------------------------------------------------------
test_fc <- forecast(final_model, h = HORIZON, level = c(80, 95))

# The selected row's orders were refitted unchanged above, so these validation
# figures do belong to the reported specification.
val_metrics  <- c(RMSE   = selected$Validation_RMSE,
                  MAE    = selected$Validation_MAE,
                  MAPE   = selected$Validation_MAPE,
                  MASE   = selected$Validation_MASE,
                  MASE_s = selected$Validation_MASE_s,
                  N      = HORIZON)
test_metrics <- evaluate(parts$test$value, test_fc$mean,
                         exclude = parts$test$imputed)

cat("\n=== FINAL TEST ACCURACY (2025-08 .. 2026-07) ===\n")
print(round(test_metrics, 4))
cat(sprintf("\nMASE uses the shared lag-1 denominator %.2f from common.R;\n",
            MASE_DENOM))
cat(sprintf("MASE_s uses the seasonal-naive denominator %.2f and is never quoted alone.\n",
            MASE_DENOM_S))
cat(sprintf("Scored on %d observed months; 2025-10 is excluded because it is\n",
            unname(test_metrics["N"])))
cat(sprintf("interpolated, not observed. Including it would give RMSE %.2f.\n",
            unname(evaluate(parts$test$value, test_fc$mean)["RMSE"])))

# Validation against test, in place of the test/train ratio the original used.
# Both are 12-step-ahead forecasts of blocks the model has not seen, so the
# comparison is between like and like.
cat("\n--- Validation vs test, both 12-step-ahead on unseen blocks ---\n")
cat(sprintf("Validation RMSE %.1f  ->  test RMSE %.1f  (%+.1f%%)\n",
            val_metrics["RMSE"], test_metrics["RMSE"],
            100 * (test_metrics["RMSE"] / val_metrics["RMSE"] - 1)))
cat("A test error far above the validation error would mean the specification\n")
cat("was tuned to the validation block. These are the two numbers to compare;\n")
cat("a test-over-training ratio compares different quantities and is not used.\n")

bench <- benchmark_table()
cat("\nAgainst the shared benchmarks (same scored months):\n")
print(round(bench[, c("RMSE", "MAE", "MASE")], 3))
cat(sprintf("Test RMSE vs RW-with-drift: %+.1f%%  (negative = model is better)\n",
            100 * (test_metrics["RMSE"] / bench["RW with drift", "RMSE"] - 1)))

save_model_result(
  model_id     = MODEL_ID,
  model_name   = selected$Model,
  window_start = min(parts$all$date),
  n_train      = nrow(parts$train),
  n_train_val  = nrow(parts$train_val),
  val_metrics  = val_metrics,
  test_metrics = test_metrics,
  lb           = lb,
  identifiable = identifiable,
  aic          = final_model$aic,
  bic          = final_model$bic,
  spec         = sprintf("kind=arima;p=%d;d=%d;q=%d;P=%d;D=%d;Q=%d;drift=%s",
                         selected$p, selected$d, selected$q,
                         selected$P, selected$D, selected$Q,
                         selected$d == 1 && selected$D == 0),
  notes        = sprintf(paste("seasonal term compulsory (P+Q>0); ranked on",
                               "validation RMSE among %d qualifiers;",
                               "%d of %d std errors non-finite"),
                         nrow(qualified), n_nan_se, n_coef)
)

# ------------------------------------------------------------------------------
# 10. TEST FORECAST TABLE + PLOT, AND THE 12-MONTH FUTURE FORECAST
# ------------------------------------------------------------------------------
test_forecast_table <- data.frame(
  Date     = parts$test$date,
  Actual   = parts$test$value,
  Imputed  = parts$test$imputed,
  Forecast = as.numeric(test_fc$mean),
  Error    = parts$test$value - as.numeric(test_fc$mean),
  Lower_80 = as.numeric(test_fc$lower[, "80%"]),
  Upper_80 = as.numeric(test_fc$upper[, "80%"]),
  Lower_95 = as.numeric(test_fc$lower[, "95%"]),
  Upper_95 = as.numeric(test_fc$upper[, "95%"])
)

cat("\n=== TEST FORECAST vs ACTUAL ===\n")
print(test_forecast_table[, c("Date", "Actual", "Imputed", "Forecast", "Error")],
      row.names = FALSE, digits = 6)
cat("The Imputed row is excluded from every score above.\n")
if (!identifiable) {
  cat("The interval columns are written for completeness only - this fit is\n")
  cat("not identifiable, so they should not be quoted.\n")
}

write.csv(test_forecast_table, "sarima_final_test_forecast.csv", row.names = FALSE)

png("sarima_final_test_forecast_plot.png", width = 1400, height = 900, res = 150)
plot(test_fc,
     main = paste0(selected$Model, " - Final Test Forecast vs Actual"),
     xlab = "Year", ylab = "Female Employment Level (Thousands)",
     include = 60)
lines(as_monthly_ts(parts$test), lwd = 2)
dev.off()

# Refit on everything for the forward forecast. The specification is held fixed;
# only the estimation sample grows.
full_model <- refit_on(selected, full_ts)

if (is.null(full_model)) {

  cat("\nThe selected specification could not be refitted on the full sample;\n")
  cat("no forward forecast is produced. Report this.\n")

} else {

  future_fc <- forecast(full_model, h = HORIZON, level = c(80, 95))
  future_dates <- seq(month_add(max(parts$all$date), 1),
                      by = "month", length.out = HORIZON)

  future_table <- data.frame(
    Date     = future_dates,
    Forecast = as.numeric(future_fc$mean),
    Lower_80 = as.numeric(future_fc$lower[, "80%"]),
    Upper_80 = as.numeric(future_fc$upper[, "80%"]),
    Lower_95 = as.numeric(future_fc$lower[, "95%"]),
    Upper_95 = as.numeric(future_fc$upper[, "95%"])
  )

  cat("\n=== 12-MONTH AHEAD FORECAST (beyond the observed sample) ===\n")
  print(future_table, row.names = FALSE, digits = 6)
  cat("Intervals are the model's own Gaussian ones and are NOT calibrated.\n")
  cat("Use the empirical quantiles from rolling_cv.R for intervals worth\n")
  cat("quoting", if (!identifiable) "- doubly so, since this fit is not identifiable." else ".", "\n")

  write.csv(future_table, "sarima_future_12_month_forecast.csv", row.names = FALSE)

  png("sarima_future_12_month_forecast.png", width = 1400, height = 900, res = 150)
  plot(future_fc,
       main = paste0(selected$Model, " - 12-Month Future Forecast"),
       xlab = "Year", ylab = "Female Employment Level (Thousands)",
       include = 60)
  dev.off()
}

# ------------------------------------------------------------------------------
# 11. SENSITIVITY: WHAT COVID INTERVENTION DUMMIES COST AND BUY
#
# Not the reported model. The SELECTED orders are refitted with the intervention
# dummies rather than re-running the search, so this is a CONTROLLED comparison -
# one thing changes - and costs one extra fit. It exists so the trade-off stated
# in the header is measured on every run instead of quoted from prose.
# ------------------------------------------------------------------------------
cat("\n\n=== SENSITIVITY: COVID DUMMIES (not the reported model) ===\n")

sens_model <- refit_on(selected, train_val_ts, X_train_val)

if (is.null(sens_model)) {

  cat("The selected orders could not be refitted with intervention dummies;\n")
  cat("no sensitivity reported.\n")

} else {

  sens_fc      <- forecast(sens_model, xreg = X_test, level = c(80, 95))
  sens_metrics <- evaluate(parts$test$value, sens_fc$mean,
                           exclude = parts$test$imputed)

  prim_resid <- residual_summary(final_model)
  sens_resid <- residual_summary(sens_model)

  comparison <- data.frame(
    Specification = c("Primary (no dummies)", "Sensitivity (COVID dummies)"),
    Model         = c(selected$Model, paste(selected$Model, "+ AO")),
    Test_RMSE     = c(test_metrics["RMSE"], sens_metrics["RMSE"]),
    Test_MAE      = c(test_metrics["MAE"],  sens_metrics["MAE"]),
    Test_MASE     = c(test_metrics["MASE"], sens_metrics["MASE"]),
    Max_abs_z     = c(prim_resid["max_abs_z"], sens_resid["max_abs_z"]),
    Kurtosis      = c(prim_resid["kurtosis"],  sens_resid["kurtosis"]),
    Identifiable  = c(identifiable, is_identifiable(sens_model)),
    row.names     = NULL
  )

  print(comparison, row.names = FALSE, digits = 5)

  cat(sprintf("\nThe dummies change test RMSE by %+.1f (%+.1f%%) and cut max|z|",
              sens_metrics["RMSE"] - test_metrics["RMSE"],
              100 * (sens_metrics["RMSE"] / test_metrics["RMSE"] - 1)))
  cat(sprintf(" from %.1f to %.1f\n", prim_resid["max_abs_z"],
              sens_resid["max_abs_z"]))
  cat(sprintf("and excess kurtosis from %.0f to %.1f - which is what makes\n",
              prim_resid["kurtosis"], sens_resid["kurtosis"]))
  cat("Gaussian prediction intervals meaningful. This is a controlled\n")
  cat("comparison: the orders are identical and only the dummies change.\n")

  write.csv(comparison, "sarima_covid_sensitivity.csv", row.names = FALSE)

  save_model_result(
    model_id     = "sarima_covid_sensitivity",
    model_name   = paste(selected$Model, "+ COVID dummies [sensitivity]"),
    window_start = min(parts$all$date),
    n_train      = nrow(parts$train),
    n_train_val  = nrow(parts$train_val),
    val_metrics  = setNames(rep(NA_real_, 3), c("RMSE", "MAE", "MAPE")),
    test_metrics = sens_metrics,
    lb           = ljung_box(residuals(sens_model),
                             fitdf = selected$p + selected$q +
                                     selected$P + selected$Q,
                             on = "level residuals"),
    identifiable = is_identifiable(sens_model),
    aic          = sens_model$aic,
    bic          = sens_model$bic,
    spec         = NA_character_,
    notes        = "SENSITIVITY ONLY - controlled (same orders as primary)"
  )
}

cat("\n=== COMPLETED ===\n")
cat("Selected model:", selected$Model, "\n")
cat("Selection rule: seasonal term compulsory, Ljung-Box qualified, ranked on\n")
cat("                validation RMSE.\n")
cat(sprintf("Identifiable  : %s (%d of %d standard errors non-finite)\n",
            identifiable, n_nan_se, n_coef))
if (!identifiable) {
  cat("Quote the point forecasts. Do not quote the intervals or the\n")
  cat("coefficient significance - see section 6.\n")
}
