# Business Risk Categorization Model

This notebook demonstrates an end-to-end Machine Learning pipeline to classify businesses into risk categories (`low`, `medium`, and `high`) using demographic, operational, financial, and behavioral survey data.

## Table of Contents
1. [Project Objective](#project-objective)
2. [Data Exploration & Insights](#data-exploration--insights)
3. [Data Preprocessing & Cleaning](#data-preprocessing--cleaning)
4. [Model Development & Architecture](#model-development--architecture)
5. [Model Evaluation & Comparison](#model-evaluation--comparison)
6. [Feature Association & Importance](#feature-association--importance)

---

## Project Objective
The goal is to build an effective classification model capable of predicting the risk classification of a business (`Target`) while robustly handling severe class imbalance, high percentages of missing data, and mixed-type inputs.

## Data Exploration & Insights
* **Class Imbalance**: The target variable is heavily skewed towards the `low` class (~65%), with `medium` making up ~30% and `high` making up only ~5% of the entries.
* **Skewness**: Numerical features such as `personal_income`, `business_expenses`, and `business_turnover` exhibited extreme positive skewness, requiring log transformations.

## Data Preprocessing & Cleaning
* **Unicode Cleaning**: Removed invisible characters (e.g., `\u200e`) from categorical features.
* **Standardization**: Text features were normalized to lowercase and stripped of leading/trailing spaces.
* **Categorical Mapping**: Standardized responses such as mapping various forms of "yes" or "no" into consistent representations.
* **Missing Value Imputation**:
  * Categorical missing values were globally filled with the value `'missing'`.
  * Numerical missing values were imputed using the median grouped by the respective `country`.
* **Scaling**: Applied a $log(1+x)$ transformation on highly skewed numerical columns followed by `RobustScaler` within the pipeline.

## Model Development & Architecture
We utilized an `imblearn.pipeline.Pipeline` to structure our processing and estimation steps:
1. **ColumnTransformer** to split categorical (`OneHotEncoder`) and numerical processing.
2. **XGBoost Classifier** as our primary model, optimized with `mlogloss` objective and custom class weights passed via `sample_weight` to address the severe class imbalance.
3. **Random Forest** and **K-Nearest Neighbors (KNN)** classifiers as comparison baselines.

## Model Evaluation & Comparison

Evaluation is performed using **5-Fold Stratified Cross-Validation**. The primary evaluation metric is **Macro F1-Score** to ensure proper weight is given to the minority `high` risk class.

### 1. XGBoost Classifier (Champion Model)
* **Mean Cross-Validated Macro F1**: **~0.803**
* **Accuracy**: **88%**
* **Key Metrics per Class**:
  * `high`: Precision: 0.90 | Recall: 0.58 | F1: 0.71
  * `low`: Precision: 0.88 | Recall: 0.99 | F1: 0.93
  * `medium`: Precision: 0.88 | Recall: 0.69 | F1: 0.78

### 2. Random Forest Classifier (Baseline Model)
* **Mean Cross-Validated Macro F1**: **~0.796**
* **Accuracy**: **88%**
* **Key Metrics per Class**:
  * `high`: Precision: 0.89 | Recall: 0.56 | F1: 0.69
  * `low`: Precision: 0.88 | Recall: 0.99 | F1: 0.93
  * `medium`: Precision: 0.88 | Recall: 0.69 | F1: 0.78

### 3. K-Nearest Neighbors Classifier (Baseline Model)
* **Mean Cross-Validated Macro F1**: **~0.641**
* **Accuracy**: **78%**

## Feature Association & Importance
* **Categorical Associations**: Cramr's V analysis was applied to map structural associations directly to the target variable.
* **Model Feature Importance**: Built-in XGBoost feature importances were visualized to determine operational drivers that dictate business risk profiles.
