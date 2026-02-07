import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

class FeatureEngineer:
    def __init__(self, target_column: str):
        self.target_column = target_column
        self.pipeline = None

class FeatureEngineer:
    def __init__(self, target_column: str):
        self.target_column = target_column
        self.preprocessor = None
        self.feature_names = None

    def _prepare_data(self, df: pd.DataFrame, is_training: bool = True):
        # Prevent data leakage
        leakage_cols = ["min_salary", "max_salary", "Salary Estimate"]
        df_clean = df.drop(columns=leakage_cols, errors="ignore")
        
        if is_training:
            if self.target_column not in df_clean.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in dataframe")
            X = df_clean.drop(columns=[self.target_column])
            y = df_clean[self.target_column]
            return X, y
        else:
            X = df_clean.drop(columns=[self.target_column], errors="ignore")
            return X

    def fit_transform(self, df: pd.DataFrame):
        X, y = self._prepare_data(df, is_training=True)

        # Define feature groups
        num_features = ["Rating", "age", "desc_len", "num_comp"]
        cat_features = [
            "job_simp", "seniority", "job_state", "Industry", 
            "Sector", "Type of ownership", "Size", "Revenue"
        ]
        bin_features = ["python_yn", "R_yn", "spark", "aws", "excel"]

        # Filter features that actually exist in X
        num_features = [f for f in num_features if f in X.columns]
        cat_features = [f for f in cat_features if f in X.columns]
        bin_features = [f for f in bin_features if f in X.columns]

        # Preprocessing for numerical data
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean'))
        ])

        # Preprocessing for categorical data
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Preprocessing for binary data (ensure numeric/imputed)
        binary_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=0))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_features),
                ('cat', categorical_transformer, cat_features),
                ('bin', binary_transformer, bin_features)
            ],
            remainder='drop'
        )

        # Fit and transform the features
        X_processed = self.preprocessor.fit_transform(X)
        
        # Keep track of column names
        try:
            cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
            cat_cols = cat_encoder.get_feature_names_out(cat_features).tolist()
            self.feature_names = num_features + cat_cols + bin_features
        except:
            self.feature_names = None

        return X_processed, y

    def transform(self, df: pd.DataFrame):
        if self.preprocessor is None:
            raise ValueError("FeatureEngineer must be fitted before calling transform.")
        
        X = self._prepare_data(df, is_training=False)
        return self.preprocessor.transform(X)
