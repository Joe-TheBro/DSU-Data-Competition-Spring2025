import polars as pl
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_recall_curve, auc

# Load Data
df = pd.read_parquet("data/DSU-Dataset.parquet")  

# Custom Transformer for Date Feature Engineering
class DateFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):


        X['Paid Date'] = pd.to_datetime(X['Paid Date'])
        X['Recieved Date'] = pd.to_datetime(X['Recieved Date'])
        X['Service Date'] = pd.to_datetime(X['Service Date'])


        X['Paid - Recieved'] = ((X['Paid Date']) - X['Recieved Date']).dt.days
        X['Paid - Service'] = (X['Paid Date'] - X['Service Date']).dt.days
        X['Service - Recieved'] = (X['Recieved Date'] - X['Service Date']).dt.days

        return X

# Custom Transformer for Outlier Detection
class OutlierDetector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):

        # Calculate IQR thresholds
        ps_iqr = X['Paid - Service'].quantile(0.75) - X['Paid - Service'].quantile(0.25)
        self.ps_threshold = X['Paid - Service'].quantile(0.75) + ps_iqr

        sr_iqr = X['Service - Recieved'].quantile(0.75) - X['Service - Recieved'].quantile(0.25)
        self.sr_threshold = X['Service - Recieved'].quantile(0.75) + sr_iqr

        return self

    def transform(self, X):

        X["p-s outlier"] = (X["Paid - Service"] > self.ps_threshold).astype(int)
        X["s-r outlier"] = (X["Service - Recieved"] > self.sr_threshold).astype(int)
        return X

# Custom Transformer for Feature Selection and Date Expansion
class FeatureSelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['Month'] = pd.to_datetime(X['Service Date']).dt.month
        X['Day of Week'] = pd.to_datetime(X['Service Date']).dt.dayofweek

        relevant_cols = [
            'ICD10 Code 1', 'ICD10 Code 2', 'ICD10 Code 3', 'ICD10 Code 4', 'ICD10 Code 5',
            'ICD10 Code 6', 'ICD10 Code 7', 'ICD10 Code 8', 'ICD10 Code 9', 'ICD10 Code 10',
            'Month', 'Day of Week', 'Network Status', 'Service Code', 'Claim Category',
            'p-s outlier', 's-r outlier'
        ]

        return X[relevant_cols]

# Define categorical and numerical columns
categorical_cols = ["Network Status", "Service Code", "Claim Category"]


# Define Full Pipeline
pipeline = Pipeline(steps=[
    ('date_features', DateFeatureEngineer()),  
    ('outliers', OutlierDetector()),  
    ('feature_selection', FeatureSelector())
    ])

'''
    ('encoding', ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'
)
'''

# Apply pipeline transformations
df_transformed = pipeline.fit_transform(df)
df_transformed = pd.DataFrame(df_transformed)

print(df_transformed.shape)


# Separate Features and Targets
y_p_s = df_transformed["p-s outlier"]
y_s_r = df_transformed["s-r outlier"]
X = df_transformed.drop(["p-s outlier", "s-r outlier"], axis=1)

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X = encoder.fit_transform(X[categorical_cols])


# Train-Test Split
X_train, X_test, y_p_s_train, y_p_s_test = train_test_split(X, y_p_s, test_size=0.2, random_state=42)
X_train, X_test, y_s_r_train, y_s_r_test = train_test_split(X, y_s_r, test_size=0.2, random_state=42)



# Train Decision Tree Models
clf_p_s = DecisionTreeClassifier(random_state=42)
clf_s_r = DecisionTreeClassifier(random_state=42)

clf_p_s.fit(X_train, y_p_s_train)
clf_s_r.fit(X_train, y_s_r_train)

# Predictions
y_p_s_pred = clf_p_s.predict(X_test)
y_s_r_pred = clf_s_r.predict(X_test)

# Accuracy Scores
acc_p_s = accuracy_score(y_p_s_test, y_p_s_pred)
acc_s_r = accuracy_score(y_s_r_test, y_s_r_pred)

print(f"Accuracy for 'p-s outlier': {acc_p_s:.4f}")
print(f"Accuracy for 's-r outlier': {acc_s_r:.4f}")

# Classification Reports
print("Classification Report for 'p-s outlier':\n", classification_report(y_p_s_test, y_p_s_pred, digits=4))
print("Classification Report for 's-r outlier':\n", classification_report(y_s_r_test, y_s_r_pred, digits=4))

# ROC-AUC Scores
roc_auc_p_s = roc_auc_score(y_p_s_test, clf_p_s.predict_proba(X_test)[:, 1])
roc_auc_s_r = roc_auc_score(y_s_r_test, clf_s_r.predict_proba(X_test)[:, 1])

print(f"ROC-AUC Score for 'p-s outlier': {roc_auc_p_s:.4f}")
print(f"ROC-AUC Score for 's-r outlier': {roc_auc_s_r:.4f}")

# Precision-Recall AUC
precision_p_s, recall_p_s, _ = precision_recall_curve(y_p_s_test, clf_p_s.predict_proba(X_test)[:, 1])
pr_auc_p_s = auc(recall_p_s, precision_p_s)

precision_s_r, recall_s_r, _ = precision_recall_curve(y_s_r_test, clf_s_r.predict_proba(X_test)[:, 1])
pr_auc_s_r = auc(recall_s_r, precision_s_r)

print(f"PR-AUC Score for 'p-s outlier': {pr_auc_p_s:.4f}")
print(f"PR-AUC Score for 's-r outlier': {pr_auc_s_r:.4f}")

