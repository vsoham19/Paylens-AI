import pandas as pd


import pandas as pd
import logging

logger = logging.getLogger("DataPipeline")

def validate_schema(df: pd.DataFrame, required_columns) -> None:
    # Check for presence of required columns
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Log missing values instead of failing
    null_counts = df[required_columns].isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        logger.warning(f"Dataset contains {total_nulls} missing values across required columns.")
        for col, count in null_counts[null_counts > 0].items():
            logger.info(f"Column '{col}' has {count} missing values.")

    # Validate numeric vs categorical (examples)
    numeric_cols = ["Rating", "age", "desc_len", "num_comp", "avg_salary"]
    for col in numeric_cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(f"Column '{col}' is expected to be numeric but is {df[col].dtype}.")
