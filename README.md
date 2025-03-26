# Advanced House Price Prediction with Ensemble Regression

## Overview
This project tackles the **Kaggle Advanced House Price Prediction** competition, where the goal is to predict house prices in Ames, Iowa, using a dataset with 79 explanatory variables. The solution employs advanced feature engineering and an ensemble of regression models to achieve accurate predictions, evaluated using Root-Mean-Squared-Error (RMSE) on the logarithm of the predicted and actual sale prices.

This is an ideal project for data science students or practitioners with some experience in Python and machine learning, looking to enhance their skills in feature engineering and advanced regression techniques.

## Dataset
The Ames Housing dataset, compiled by Dean De Cock, contains detailed information about residential homes, including features like lot size, overall quality, living area, and more. The dataset is split into training and test sets, with the task of predicting the `SalePrice` for each house in the test set.

- **Training Data**: Contains 79 features + `SalePrice`.
- **Test Data**: Contains 79 features, with `SalePrice` to be predicted.
- **Evaluation Metric**: RMSE between `log(predicted SalePrice)` and `log(actual SalePrice)`.

## Approach
### 1. Data Preprocessing
- Handle missing values in key columns (`LotFrontage`, `MasVnrArea`, `GarageYrBlt`) using median imputation or zero-filling.
- Factorize categorical variables and compute z-scores for selected numerical features to normalize them.

### 2. Feature Engineering
- Create aggregate features (e.g., `TotalBath`, `TotalSF`, `OverallGrade`).
- Incorporate neighborhood statistics (mean, median, std of `SalePrice` by neighborhood).
- Apply clustering (KMeans) and dimensionality reduction (PCA) to capture latent patterns.
- Transform skewed numerical features using the Yeo-Johnson method.
- Add polynomial and interaction terms for important features like `OverallQual` and `GrLivArea`.

### 3. Modeling
- Use an ensemble of three regression models combined via `VotingRegressor`:
  - **RandomForestRegressor**: 1000 trees, bagging-based.
  - **XGBRegressor**: 1000 estimators, learning rate 0.05, gradient boosting.
  - **LGBMRegressor**: 1000 estimators, learning rate 0.05, light gradient boosting.
- Train on log-transformed `SalePrice` and evaluate using 5-fold cross-validation.

### 4. Prediction
- Generate predictions on the test set, inverse-transform them (`expm1`), and save them to a submission file (`submission.csv`).
