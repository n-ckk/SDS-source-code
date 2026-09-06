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

# Fill missing female employment values using linear interpolation
data$female_employment <- na.approx(
  data$female_employment,
  x = data$date,
  na.rm = FALSE
)

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
# Last 12 months are used as test data
test_size <- 12

train_ts <- head(
  female_ts,
  length(female_ts) - test_size
)

test_ts <- tail(
  female_ts,
  test_size
)

cat("\nTraining observations:", length(train_ts), "\n")
cat("Testing observations:", length(test_ts), "\n")

cat(
  "Training period:",
  start(train_ts),
  "to",
  end(train_ts),
  "\n"
)

cat(
  "Testing period:",
  start(test_ts),
  "to",
  end(test_ts),
  "\n"
)

# 15. Save Processed Data
write.csv(
  data,
  "processed_female_employment.csv",
  row.names = FALSE
)