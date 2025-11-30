```markdown
# Wine Quality Regression with fastml

This project demonstrates a full machine-learning workflow using the fastml R package
on the UCI Wine Quality (Red Wine) dataset.

## Features
- Automated preprocessing and scaling  
- Multiple regression models (Random Forest, XGBoost, Elastic Net, Linear Regression, Lasso)  
- Hyperparameter tuning  
- Model comparison  
- Residual diagnostics  
- Feature importance analysis (DALEX)  
- SHAP explanations  
- Custom ggplot-based visualizations  

## Installation

### 1. Clone or download this repository
```bash
git clone https://github.com/yourusername/fastml_wine.git
cd fastml_wine
```

### 2. Restore environment via renv
```r
renv::restore()
```

### 3. Load required packages
```r
library(fastml)
library(dalex)
library(ggplot2)
```

## Dataset

**Source:** UCI Machine Learning Repository  
**Link:** https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/

The dataset contains physicochemical properties of red wine and quality ratings.

## Model Training

```r
model_fastml <- fastml(
  data = wine_red,
  label = "quality",
  algorithms = c("rand_forest", "xgboost", "elastic_net", "linear_reg", "lasso_reg")
)
```

## Results

View model performance and diagnostics:

```r
# Display best model and performance metrics
model_fastml

# Get performance summary
model_fastml$performance

# Plot residual diagnostics
plot(model_fastml, type = "all")

# Feature importance and SHAP values
vi <- explain_dalex(
  model_fastml,
  features = NULL,
  vi_iterations = 20
)
```

## Project Structure

```
fastml_wine/
├── fastml.R           # Main analysis script
├── README.md          # This file
├── .Rprofile          # R startup configuration
└── renv/              # Project environment (reproducibility)
```

## License

MIT License
```

Key fixes:
- Proper markdown code fence formatting (3 backticks, not mixed)
- Corrected numbered list structure
- Added project structure section
- Added license placeholder
- Improved readability with proper spacingKey fixes:
- Proper markdown code fence formatting (3 backticks, not mixed)
- Corrected numbered list structure
- Added project structure section
- Added license placeholder
- Improved readability with proper spacing

bartuyurdacan: create readme

Workspace: Collecting workspace informationHere's a comprehensive README for your wine quality prediction project:

````markdown
# Wine Quality Prediction with fastml

A machine learning workflow for predicting red wine quality using the **fastml** R package with automated model selection, hyperparameter tuning, and interpretability analysis.

## Overview

This project demonstrates a complete data science pipeline using the UCI Wine Quality dataset. It trains multiple regression models, compares their performance, and provides feature importance analysis using DALEX explainability methods.

## Dataset

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/)

The dataset contains:
- **1,599 red wine samples**
- **11 physicochemical features** (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol)
- **1 target variable** (quality rating from 3 to 8)

## Project Structure

```
fastml_wine/
├── fastml.R              # Main analysis script
├── README.md             # This file
├── .Rprofile             # R startup configuration
├── .gitignore            # Git ignore rules
└── renv/                 # Project environment (reproducibility)
    ├── activate.R
    ├── settings.json
    └── library/          # Isolated package library
```

## Installation & Setup

### Prerequisites
- R 4.5.0 or later
- Git (optional, for version control)

### Steps

1. **Clone or download the repository:**
```bash
git clone <repository-url>
cd fastml_wine
```

2. **Restore the project environment:**
```r
renv::restore()
```

This installs all required packages in an isolated project library.

3. **Run the analysis:**
```r
source("fastml.R")
```

## Key Results

The fastml model trains and compares 5 algorithms:

| Algorithm | RMSE | R² | MAE |
|-----------|------|-----|-----|
| **Random Forest (Best)** | 0.507 | 0.538 | 0.380 |
| Linear Regression | 0.586 | 0.376 | 0.465 |
| Elastic Net | 0.587 | 0.373 | 0.466 |
| Lasso | 0.587 | 0.373 | 0.466 |
| XGBoost | 1.18 | 0.391 | 1.05 |

**Best Model:** Random Forest (ranger) with RMSE = 0.507

## Main Features

- **Automated Preprocessing:** Zero-variance removal, dummy encoding, centering, scaling
- **Multiple Algorithms:** Random Forest, XGBoost, Linear Regression, Elastic Net, Lasso
- **Hyperparameter Tuning:** Built-in optimization via tidymodels
- **Model Comparison:** Performance metrics (RMSE, R², MAE) across all models
- **Feature Importance:** Variable importance analysis via permutation
- **SHAP Values:** Shapley-based explanations for individual predictions
- **Residual Diagnostics:** Plot residuals to assess model fit

## Usage

### Basic Model Training
```r
library(fastml)

# Load data
url_red <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine_red <- read.csv(url_red, sep = ";")

# Train model
model_fastml <- fastml(
  data = wine_red,
  label = "quality",
  algorithms = c("rand_forest", "xgboost", "linear_reg", "elastic_net", "lasso_reg")
)
```

### View Results
```r
# Best model summary
model_fastml

# Performance metrics
model_fastml$performance

# Residual diagnostics
plot(model_fastml, type = "all")

# Feature importance & SHAP
vi <- explain_dalex(
  model_fastml,
  features = NULL,
  vi_iterations = 20
)
```

## Dependencies

Key packages:
- **fastml** - Automated machine learning
- **tidymodels** - Modeling framework
- **ranger** - Fast random forests
- **xgboost** - Gradient boosting
- **glmnet** - Elastic Net & Lasso
- **dalex** - Model explainability
- **ggplot2** - Visualization

All packages are managed via **renv** for reproducibility.

## Reproducibility

This project uses **renv** to ensure consistent package versions across environments:

```r
# Check project status
renv::status()

# Take a snapshot of dependencies
renv::snapshot()

# Restore to a previous state
renv::restore()
```

## License

MIT License

## Author

Bartu

## References

- [fastml Documentation](https://cran.r-project.org/web/packages/fastml/index.html)
- [tidymodels](https://www.tidymodels.org/)
- [DALEX: Explainable AI](https://modeloriented.github.io/DALEX/)
- [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
````

This README includes:
- Clear project description and overview
- Dataset documentation with feature breakdown
- Installation and setup instructions
- Performance comparison table from your results
- Code examples for key workflows
- Dependencies and reproducibility info
- References for further learning

bartuyurdacan:  create readme 

Workspace: Sure, here is the file you requested:[README.md](command:github.copilot.openFile?%5B%7B%22value%22%3A%5B%7B%22name%22%3A%22README.md%22%7D%5D%2C%22baseUri%22%3A%7B%22%24mid%22%3A1%2C%22path%22%3A%22%2FREADME.md%22%2C%22scheme%22%3A%22vscode-copilot-file%22%2C%22authority%22%3A%227f2b91b6-ef42-4eef-bb5b-242b095552e2%22%7D%7D%5D)

Create File...

bartuyurdacan: arrange for codes to readme

GitHub Copilot: Here's a well-organized README with your project code:

````markdown
# Wine Quality Prediction with fastml

A machine learning workflow for predicting red wine quality using the **fastml** R package with automated model selection, hyperparameter tuning, and interpretability analysis.

## Overview

This project demonstrates a complete data science pipeline using the UCI Wine Quality dataset. It trains multiple regression models, compares their performance, and provides feature importance analysis using DALEX explainability methods.

## Dataset

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/)

The dataset contains:
- **1,599 red wine samples**
- **11 physicochemical features** (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol)
- **1 target variable** (quality rating from 3 to 8)

## Project Structure

```
fastml_wine/
├── fastml.R              # Main analysis script
├── README.md             # This file
├── .Rprofile             # R startup configuration
└── renv/                 # Project environment (reproducibility)
```

## Installation & Setup

### Prerequisites
- R 4.5.0 or later

### Steps

1. **Restore the project environment:**
```r
renv::restore()
```

2. **Run the analysis:**
```r
source("fastml.R")
```

## Main Analysis Script (fastml.R)

### 1. Environment Setup
```r
renv::status()
renv::snapshot()
library(fastml)
```

### 2. Load Data
```r
url_red <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine_red <- read.csv(url_red, sep = ";")
data <- wine_red
data$quality <- as.numeric(data$quality)
```

### 3. Train Model
```r
set.seed(123)

model_fastml <- fastml(
  data = wine_red,
  label = "quality",
  algorithms = c("rand_forest", "xgboost", "linear_reg", "elastic_net", "lasso_reg"),
  task = "regression"
)
```

### 4. View Model Summary
```r
model_fastml
```

### 5. Get Performance Metrics
```r
perf <- model_fastml$performance
perf
```

### 6. Generate Visualizations
```r
# Residual diagnostics
plot(model_fastml, type = "all")

# ROC curve for best model
plot(model_fastml, type = "roc", algorithm = "ranger")
```

### 7. Feature Importance & Explainability
```r
vi <- explain_dalex(
  model_fastml,
  features = NULL,
  vi_iterations = 20
)

# Variable importance
plot(vi$variable_importance)

# SHAP values
plot(vi$shap_values)
```

## Key Results

| Algorithm | RMSE | R² | MAE |
|-----------|------|-----|-----|
| **Random Forest (Best)** | 0.507 | 0.538 | 0.380 |
| Linear Regression | 0.586 | 0.376 | 0.465 |
| Elastic Net | 0.587 | 0.373 | 0.466 |
| Lasso | 0.587 | 0.373 | 0.466 |
| XGBoost | 1.18 | 0.391 | 1.05 |

**Best Model:** Random Forest (ranger) with RMSE = 0.507

## Features

- ✅ **Automated Preprocessing:** Zero-variance removal, dummy encoding, centering, scaling
- ✅ **Multiple Algorithms:** Random Forest, XGBoost, Linear Regression, Elastic Net, Lasso
- ✅ **Hyperparameter Tuning:** Built-in optimization via tidymodels
- ✅ **Model Comparison:** Performance metrics across all models
- ✅ **Feature Importance:** Variable importance analysis via permutation
- ✅ **SHAP Values:** Shapley-based explanations for predictions
- ✅ **Residual Diagnostics:** Assess model fit quality

## Dependencies

Key packages managed via **renv**:
- **fastml** - Automated machine learning
- **tidymodels** - Modeling framework
- **ranger** - Fast random forests
- **xgboost** - Gradient boosting
- **glmnet** - Elastic Net & Lasso
- **dalex** - Model explainability
- **ggplot2** - Visualization

## Reproducibility

Check and manage environment:

```r
# Check project status
renv::status()

# Snapshot dependencies
renv::snapshot()

# Restore environment
renv::restore()
```

## License

MIT License

## References

- [fastml Documentation](https://cran.r-project.org/web/packages/fastml/index.html)
- [tidymodels](https://www.tidymodels.org/)
- [DALEX: Explainable AI](https://modeloriented.github.io/DALEX/)
- [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
````

This README now includes:
- Clear structure with setup instructions
- Organized code sections from your fastml.R script
- Step-by-step analysis workflow
- Results table with model performance
- Feature documentation
- Reproducibility info

bartuyurdacan: add graphs

GitHub Copilot: Here's an updated README with a section for graphs and visualizations:

```markdown
# Wine Quality Prediction with fastml

A machine learning workflow for predicting red wine quality using the **fastml** R package with automated model selection, hyperparameter tuning, and interpretability analysis.

## Overview

This project demonstrates a complete data science pipeline using the UCI Wine Quality dataset. It trains multiple regression models, compares their performance, and provides feature importance analysis using DALEX explainability methods.

## Dataset

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/)

The dataset contains:
- **1,599 red wine samples**
- **11 physicochemical features** (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol)
- **1 target variable** (quality rating from 3 to 8)

## Project Structure

```
fastml_wine/
├── fastml.R              # Main analysis script
├── README.md             # This file
├── .Rprofile             # R startup configuration
└── renv/                 # Project environment (reproducibility)
```

## Installation & Setup

### Prerequisites
- R 4.5.0 or later

### Steps

1. **Restore the project environment:**
```r
renv::restore()
```

2. **Run the analysis:**
```r
source("fastml.R")
```

## Main Analysis Script (fastml.R)

### 1. Environment Setup
```r
renv::status()
renv::snapshot()
library(fastml)
```

### 2. Load Data
```r
url_red <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine_red <- read.csv(url_red, sep = ";")
data <- wine_red
data$quality <- as.numeric(data$quality)
```

### 3. Train Model
```r
set.seed(123)

model_fastml <- fastml(
  data = wine_red,
  label = "quality",
  algorithms = c("rand_forest", "xgboost", "linear_reg", "elastic_net", "lasso_reg"),
  task = "regression"
)
```

### 4. View Model Summary
```r
model_fastml
```

### 5. Get Performance Metrics
```r
perf <- model_fastml$performance
perf
```

### 6. Generate Visualizations
```r
# Residual diagnostics
plot(model_fastml, type = "all")

# ROC curve for best model
plot(model_fastml, type = "roc", algorithm = "ranger")
```

### 7. Feature Importance & Explainability
```r
vi <- explain_dalex(
  model_fastml,
  features = NULL,
  vi_iterations = 20
)

# Variable importance
plot(vi$variable_importance)

# SHAP values
plot(vi$shap_values)
```

## Key Results

| Algorithm | RMSE | R² | MAE |
|-----------|------|-----|-----|
| **Random Forest (Best)** | 0.507 | 0.538 | 0.380 |
| Linear Regression | 0.586 | 0.376 | 0.465 |
| Elastic Net | 0.587 | 0.373 | 0.466 |
| Lasso | 0.587 | 0.373 | 0.466 |
| XGBoost | 1.18 | 0.391 | 1.05 |

**Best Model:** Random Forest (ranger) with RMSE = 0.507

## Visualizations

### Model Performance Comparison
```r
# Compare all models side-by-side
plot(model_fastml, type = "all")
```
This plot displays residual diagnostics for the best model, showing:
- Predicted vs. Actual values
- Residuals distribution
- Q-Q plot for normality check
- Scale-Location plot for homoscedasticity

### Residual Diagnostics
```r
# Detailed residual analysis
plot(model_fastml, type = "residuals")
```
Helps identify:
- Non-linear patterns
- Heteroscedasticity
- Outliers and influential points

### Feature Importance (Permutation-based)
```r
# Plot variable importance
plot(vi$variable_importance)
```
Shows which features contribute most to predictions by measuring performance drop when features are randomly shuffled.

### SHAP Values (Shapley Explanations)
```r
# Plot SHAP force plot and dependence plots
plot(vi$shap_values)
```
Provides:
- Individual prediction explanations
- Feature contribution breakdown
- Feature interaction effects

### Model-specific Visualizations
```r
# Random Forest specific plots
plot(model_fastml, type = "forest")

# Feature dependence plots
plot(model_fastml, type = "pdp")

# Partial dependence plots
plot(model_fastml, type = "ale")
```

## Features

- ✅ **Automated Preprocessing:** Zero-variance removal, dummy encoding, centering, scaling
- ✅ **Multiple Algorithms:** Random Forest, XGBoost, Linear Regression, Elastic Net, Lasso
- ✅ **Hyperparameter Tuning:** Built-in optimization via tidymodels
- ✅ **Model Comparison:** Performance metrics across all models
- ✅ **Feature Importance:** Variable importance analysis via permutation
- ✅ **SHAP Values:** Shapley-based explanations for predictions
- ✅ **Residual Diagnostics:** Assess model fit quality
- ✅ **Comprehensive Visualizations:** Multiple plot types for model understanding

## Dependencies

Key packages managed via **renv**:
- **fastml** - Automated machine learning
- **tidymodels** - Modeling framework
- **ranger** - Fast random forests
- **xgboost** - Gradient boosting
- **glmnet** - Elastic Net & Lasso
- **dalex** - Model explainability
- **ggplot2** - Visualization
- **ingredients** - SHAP and feature importance

## Reproducibility

Check and manage environment:

```r
# Check project status
renv::status()

# Snapshot dependencies
renv::snapshot()

# Restore environment
renv::restore()
```

## License

MIT License

## References

- [fastml Documentation](https://cran.r-project.org/web/packages/fastml/index.html)
- [tidymodels](https://www.tidymodels.org/)
- [DALEX: Explainable AI](https://modeloriented.github.io/DALEX/)
- [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
```

I've added a **Visualizations** section that documents:
- Model performance comparison plots
- Residual diagnostics
- Feature importance visualization
- SHAP value explanations
- Model-specific visualization options

Each includes brief descriptions of what the plots show and how to generate them.I've added a **Visualizations** section that documents:
- Model performance comparison plots
- Residual diagnostics
- Feature importance visualization
- SHAP value explanations
- Model-specific visualization options

Each includes brief descriptions of what the plots show and how to generate them.