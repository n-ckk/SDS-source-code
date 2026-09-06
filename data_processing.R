# ============================================================
# DATA PROCESSING
# SDG 5: Gender Equality
# Dataset: LNS12000002 - Employment Level: Women
# ============================================================

# 1. Load Packages
library(readxl)
library(dplyr)
library(ggplot2)
library(forecast)
library(tseries)
library(zoo)

# 2. Read Excel Dataset
data_raw <- read_excel(
  "LNS12000002.xlsx",
  sheet = "Monthly"
)

# View the original structure
print(head(data_raw))
print(str(data_raw))

# 3. Rename Columns
data <- data_raw %>%
  rename(
    date = observation_date,
    female_employment = LNS12000002
  )

# 4. Data Type Conversion
data <- data %>%
  mutate(
    date = as.Date(date),
    female_employment = as.numeric(female_employment)
  ) %>%
  arrange(date)

# 5. Basic Data Information
cat("Number of observations:", nrow(data), "\n")
cat("Number of variables:", ncol(data), "\n")

cat(
  "Date range:",
  as.character(min(data$date)),
  "to",
  as.character(max(data$date)),
  "\n"
)

# 6. Check and Handle Missing Values
cat("\nMissing values before treatment:\n")
print(colSums(is.na(data)))

# Remove rows only if date itself is missing
data <- data %>%
  filter(!is.na(date))

# Record which months are interpolated BEFORE filling them. This flag is written
# to the CSV so every downstream model can report - or exclude - synthetic points
# instead of re-deriving the flag and getting it wrong. 2025-10 is blank at source
# (it falls inside every model's test window, so it is scored as if observed).
data$imputed <- is.na(data$female_employment)

# Fill missing female employment values using linear interpolation
data$female_employment <- na.approx(
  data$female_employment,
  x = data$date,
  na.rm = FALSE
)

cat("\nMonths interpolated:", sum(data$imputed), "\n")
if (any(data$imputed)) {
  cat("Dates interpolated :",
      paste(format(data$date[data$imputed]), collapse = ", "), "\n")
}

cat("\nMissing values after interpolation:\n")
print(colSums(is.na(data)))

# 7. Check Duplicate Dates
duplicate_dates <- data %>%
  group_by(date) %>%
  summarise(n = n()) %>%
  filter(n > 1)

cat("\nDuplicate dates:\n")
print(duplicate_dates)

# 8. Descriptive Statistics
cat("\nDescriptive Statistics:\n")

summary_stats <- data %>%
  summarise(
    Minimum = min(female_employment),
    Q1 = quantile(female_employment, 0.25),
    Median = median(female_employment),
    Mean = mean(female_employment),
    Q3 = quantile(female_employment, 0.75),
    Maximum = max(female_employment),
    SD = sd(female_employment)
  )

print(summary_stats)

# 9. Plot Original Time Series
ggplot(data, aes(x = date, y = female_employment)) +
  geom_line() +
  labs(
    title = "Female Employment Level in the United States",
    subtitle = "LNS12000002",
    x = "Year",
    y = "Employment Level (Thousands)"
  ) +
  theme_minimal()

# 10. Create Time Series Object
female_ts <- ts(
  data$female_employment,
  start = c(
    as.numeric(format(min(data$date), "%Y")),
    as.numeric(format(min(data$date), "%m"))
  ),
  frequency = 12
)

cat("\nTime Series:\n")
print(female_ts)

# 11. Decompose Time Series
decomposition <- decompose(
  female_ts,
  type = "additive"
)

plot(decomposition)

# 12. Check Stationarity
cat("\nADF Test - Original Series:\n")

adf_original <- adf.test(
  female_ts,
  alternative = "stationary"
)

print(adf_original)

# 13. First Difference
female_diff <- diff(female_ts)

cat("\nADF Test - First Differenced Series:\n")

adf_diff <- adf.test(
  female_diff,
  alternative = "stationary"
)

print(adf_diff)

# Plot differenced series
plot(
  female_diff,
  main = "First Differenced Female Employment",
  ylab = "Differenced Employment",
  xlab = "Year"
)

# 14. Train-Test Split
# REMOVED. The authoritative split lives in common.R (VAL_START / TEST_START) and
# is shared by every model. This script previously printed a second, two-way split
# that no model actually used and that disagreed with the three-way split the
# models apply - exactly the kind of drift common.R exists to prevent.

# 15. Save Processed Data
write.csv(
  data,
  "processed_female_employment.csv",
  row.names = FALSE
)