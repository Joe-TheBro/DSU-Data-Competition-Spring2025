# Health Insurance Claims Prediction Competition

## Team Information
- **Team Name:** pipsucks
- **Team Members :** Samyam Aryal, Joseph Hammond
- **Contact Person:** Joseph Hammond
- **Contact Email:** joe.hammond@trojans.dsu.edu
- **Institution/Organization:** Dakota State University

## Overview
This project addresses two primary objectives:
1. **High-Dollar Claims Prediction:** Develop a model to predict High-Dollar claims.
2. **Delayed Claims Prediction:** Estimate the number of "delayed" claims per month. (Delayed claims are identified as those falling into outlier categorization based on processing delays.)

The provided code primarily details the development of the High-Dollar Claims Prediction model with feature engineering, hyperparameter tuning, and custom early stopping. The methodology for delayed claims prediction can follow a similar approach with additional outlier detection and regression analysis techniques.

## Methodology

### Data Preprocessing and Feature Engineering
- **Data Ingestion:**  
  - The dataset is loaded from a Parquet file (`DSU-Dataset.parquet`) using pandas.
- **Date Conversions:**  
  - Columns for 'Service Date', 'Recieved Date', and 'Paid Date' are converted to datetime objects.
- **Feature Creation:**  
  - **Delay Features:**  
    - `recieve delay`: Difference in days between the "Recieved Date" and the "Service Date".  
    - `payment delay`: Difference in days between the "Paid Date" and the "Recieved Date".  
  - **Temporal Features:**  
    - Extracted the month and day of the week from the "Service Date".
- **Handling Missing Values:**  
  - Claims with a missing `High Cost Claim` label are separated for future prediction.
  - All rows with missing values in `High Cost Claim` are dropped for training/ validation.
- **Data Reduction & Encoding:**  
  - Dropped unnecessary columns such as `Claim ID`, `Patient ID`, date fields, `Modifiers`, and the target label from features.
  - Applied one-hot encoding to all categorical features.
  - Renamed columns with invalid characters (e.g., renaming `Member Age_< 1 Yrs Old` to `Member Age_lessthan 1 Yrs Old`).

## High-Dollar Claims Prediction Model

#### Model Development
- **Data Splitting:**  
  - The processed dataset is split into training, validation, and testing sets using `train_test_split` to ensure robust evaluation.
- **Baseline Model:**  
  - Implemented an XGBoost classifier (`XGBClassifier`) using a set of initial hyperparameters:
    - Learning rate (`eta`): 0.295  
    - Maximum depth: 3  
    - Maximum leaves: 4  
    - Regularization parameters (`lambda`, `alpha`, `gamma`) and sampling parameters (`subsample`, `colsample_bytree`)
- **Evaluation Metrics:**  
  - Model performance is evaluated using metrics including AUC-PR and F1 score.

#### Hyperparameter Tuning and Custom Early Stopping
- **Optuna Integration:**  
  - Hyperparameter optimization is performed using Optuna. The objective function tunes parameters such as learning rate, max_depth, max_leaves, regularization (L1 & L2), and sampling fractions.
- **Custom Early Stopping Callback:**  
  - A custom early stopping callback is defined to stop training if the improvement in the validation AUC-PR is less than a specified threshold (`min_delta`) over 50 rounds. This ensures efficient training and prevents overfitting.
- **Final Training & Feature Importance:**  
  - The model is retrained with the best-found parameters and the custom early stopping mechanism.
  - Feature importance is visualized using XGBoost’s built-in plotting functions.


## Delayed Claims Prediction

For delayed claims, the goal is to flag claims with unusually long processing times by detecting outliers in key delay metrics. The methodology for this component is as follows:

#### Data Ingestion and Feature Engineering
- **Efficient Data Loading:**  
  - The dataset is loaded from a Parquet file using [Polars](https://www.pola.rs/), which is optimized for high-performance data processing.
- **Delay Metric Computation:**  
  - Three delay metrics are computed:
    - **Paid - Recieved:** Number of days between the "Paid Date" and the "Recieved Date".
    - **Paid - Service:** Number of days between the "Paid Date" and the "Service Date".
    - **Service - Recieved:** Number of days between the "Recieved Date" and the "Service Date".
- **Outlier Thresholds via IQR:**  
  - For both the "Paid - Service" and "Service - Recieved" metrics, thresholds are calculated using the Interquartile Range (IQR) method:
    - **Threshold = 75th Percentile + IQR**
  - Two binary features, `p-s outlier` and `s-r outlier`, are then created to indicate whether a claim's delay exceeds its respective threshold.

#### Data Preparation
- **Temporal Features:**  
  - Additional features such as the month and day of the week are extracted from the "Service Date" to capture seasonal and weekly patterns.
- **Handling Categorical Data:**  
  - Categorical columns (e.g., "Network Status", "Service Code", "Claim Category") are identified.
  - One-Hot Encoding is applied using scikit-learn’s `OneHotEncoder` to convert these categorical variables into a numeric format.
- **Feature Consolidation:**  
  - The numerical features and the one-hot encoded categorical features are combined to form the final feature set (X).
- **Target Definition:**  
  - Two target variables are defined:
    - `y_p_s` for the `p-s outlier` (Paid - Service outlier).
    - `y_s_r` for the `s-r outlier` (Service - Recieved outlier).

#### Modeling and Evaluation
- **Model Training:**  
  - Two separate Decision Tree Classifiers are trained:
    - One for predicting the `p-s outlier`.
    - Another for predicting the `s-r outlier`.
- **Data Splitting:**  
  - The dataset is split into training and test sets using scikit-learn’s `train_test_split`, ensuring reproducibility with a fixed random state.
- **Performance Metrics:**  
  - **Accuracy:** Evaluates overall classification performance.
  - **Classification Report:** Provides detailed precision, recall, and F1-score metrics.
  - **ROC-AUC Score:** Assesses the model’s ability to distinguish between classes.
  - **Precision-Recall AUC:** Offers insight into performance on imbalanced data.

- **Inference**
  - Run python inference.py 'data_path' from terminal at the path of your project. make sure data is in csv format and also make sure path is in quotes.


This approach allows for the identification of delayed claims based on statistically defined outlier thresholds. The binary predictions for both delay types can be aggregated to estimate the monthly number of delayed claims, enabling targeted interventions and process improvements.

### Tools and Libraries
- **Programming Language:** Python
- **Key Libraries:**  
  - [pandas](https://pandas.pydata.org/) for data manipulation  
  - [xgboost](https://xgboost.readthedocs.io/) for model training and evaluation  
  - [scikit-learn](https://scikit-learn.org/) for preprocessing and evaluation metrics  
  - [optuna](https://optuna.org/) for hyperparameter tuning  
  - [matplotlib](https://matplotlib.org/) for plotting feature importance

## File Structure
- **`high_dollar_model.py` / `high_dollar_model.ipynb`:**  
  - Contains the code for data preprocessing, training the High-Dollar Claims Prediction model, hyperparameter tuning with Optuna, and plotting feature importance.
- **`delayed_claims_model.py` / `delayed_claims_model.ipynb`:**  
  - (If available) Contains the methodology for predicting delayed claims per month using outlier detection and regression analysis.
- **`data/`:**  
  - Directory containing the input dataset (`DSU-Dataset.parquet`).
- **`requirements.txt`:**  
  - Lists all required Python packages.
- **`README.md`:**  
  - This document outlining the approach and submission details.

## Instructions for Running the Code

1. **Environment Setup:**  
   - Install the required libraries:
     ```bash
     pip install -r requirements.txt
     ```
2. **Execution:**  
   - For the High-Dollar Claims Prediction model:
     ```bash
     python high_dollar_model.py
     ```
     or open the `high_dollar_model.ipynb` notebook in your preferred environment.
   - (If applicable) For the Delayed Claims Prediction model:
     ```bash
     python delayed_claims_model.py
     ```
     or open the corresponding notebook.
3. **Notes:**  
   - Ensure the dataset file (`DSU-Dataset.parquet`) is located in the `data/` directory.
   - Adjust any file paths if necessary, based on your local environment.

## Submission Guidelines
- **ZIP File Naming:**  
  - Consolidate all relevant files into a single ZIP/compressed file. Name the file as:
    ```
    [TeamName]_[SubmissionName].zip
    ```
    *Example: TeamAnalytics_Solution.zip*
- **File Size Note:**  
  - If the ZIP file exceeds 20MB, please upload it to a file hosting service (e.g., Dropbox, Google Drive) and include the download link in your submission email.

## Contact
For any questions or further clarifications, please reach out to:
- **Contact Person:** Joseph Hammond
- **Email:** [joe.hammond@trojans.dsu.edu]
