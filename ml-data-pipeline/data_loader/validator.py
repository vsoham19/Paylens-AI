import pandas as pd

req_col = {"id", "name", "age", "salary"}

def validator(df: pd.DataFrame) -> None:
    req_cols = {"age", "salary"}

    missing_cols = req_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Required columns are missing: {missing_cols}")

    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values")

    if not pd.api.types.is_numeric_dtype(df["age"]):
        raise ValueError("Column 'age' must be numeric")

    if not pd.api.types.is_numeric_dtype(df["salary"]):
        raise ValueError("Column 'salary' must be numeric")

