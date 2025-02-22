import xgboost as xgb
from xgboost import XGBClassifier, plot_tree
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle

# Custom transformer to handle datetime columns and delay features
class DateFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert date columns to datetime
        X[['Service Date', 'Recieved Date', 'Paid Date']] = X[['Service Date', 'Recieved Date', 'Paid Date']].apply(pd.to_datetime)

        # Compute receive delay and payment delay
        X['recieve delay'] = (X['Recieved Date'] - X['Service Date']).dt.days
        X['payment delay'] = (X['Paid Date'] - X['Recieved Date']).dt.days

        # Engineering datetime features
        X['Month'] = X['Service Date'].dt.month
        X['Day of Week'] = X['Service Date'].dt.dayofweek

        return X
def categorical_selector_function(df):
    return df.select_dtypes(include=['object']).columns

# Custom transformer to handle NaN values and remove unnecessary columns
class FeatureRemoval(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cols_to_drop = ['Claim ID', 'Patient ID', 'Service Date', 'Recieved Date', 'Paid Date', 'Modifiers', 'High Cost Claim']
        X = X.drop(cols_to_drop, axis=1)

        return X
# Main pipeline
pipeline = Pipeline(steps=[
    ('date_features', DateFeatureEngineer()),  # Process dates and engineer features
    ('data_clean', FeatureRemoval()),  # Clean data and extract labels
    ('onehotencoder', ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), 
             categorical_selector_function,
             #lambda X: X.select_dtypes(include=['object']).columns
             )  # Only apply OneHotEncoder to categorical columns
        ], 
        remainder='passthrough'
    )),
    ])

if __name__ == '__main__':

    df = pd.read_parquet("./data/train_set.parquet")
    train_pipeline = Pipeline(steps=[
        ('preprocess', pipeline),  # Process dates and engineer features
        ('model', XGBClassifier(verbosity=2, eval_metric='aucpr', early_stopping_rounds=5))]
    )

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df, df['High Cost Claim'])
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)

    X_train_transformed = pipeline.fit_transform(X_train)
    X_val_transformed = pipeline.transform(X_val)
    X_test_transformed = pipeline.transform(X_test)

    print(X_train_transformed.shape, X_val_transformed.shape)

    train_pipeline.fit(X_train, y_train, model__eval_set=[(X_val_transformed, y_val)]) 

    pickle.dump(train_pipeline, open('train_pipeline.pkl', 'wb'))
