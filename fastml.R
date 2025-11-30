renv::status()
renv::snapshot()
library(fastml)

url_red <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

wine_red <- read.csv(url_red, sep = ";")
library(fastml)

data <- wine_red
data$quality <- as.numeric(data$quality)

set.seed(123)
model_fastml <- fastml(
  data = wine_red,
  label = "quality",
  algorithms = c("rand_forest", "xgboost", "linear_reg", "elastic_net", "lasso_reg")
)


model_fastml
perf <- model_fastml$performance

best <- model_fastml$best_model_name
best


vi <- explain_dalex(
  model_fastml,
  features = NULL,
  vi_iterations = 20   # daha stabil VI için arttırılabilir
)


model_fastml$performance

plot(model_fastml, type = "all")
