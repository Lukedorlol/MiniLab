"""
Preprocessing utilities for the regression exercises.
You have to implement the different functions (according to the exercises).

This module provides functions for:
- Cleaning raw housing data (handling missing values, removing outliers)
- Encoding categorical features into numeric representations
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Encode categorical features (required to fully implement for Task 1.3)
# ---------------------------------------------------------------------
def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical (object/bool) columns to numeric.
    The transformation should be simple and deterministic.

    Example:
    - 'condition' → string length
    - boolean features → {False: 0, True: 1}
    """
    df_encoded = df.copy()

    # Example categorical feature: 'condition'
    condition_mapping = {
        "well_kept": 0,            
        "modernized": 1,         
        "fully_renovated": 2,     
        "first_time_use_after_refurbishment": 4,
        "refurbished": 3,
        "first_time_use": 6,       
        "mint_condition": 5,       
        "unknown": 1
    }

    df_encoded["condition_num"] = df_encoded["condition"].map(condition_mapping)
    df_encoded["condition_num"] = df_encoded["condition_num"].fillna(3)


    # Example categorical feature: 'condition'

    # Boolean → 0/1
    bool_cols = df_encoded.select_dtypes(include=["bool"]).columns
    for col in bool_cols:
        df_encoded[col + "_num"] = df_encoded[col].astype(int)

    # Optional: encode heatingType or interiorQual if present
    for cat_col in ["heatingType", "interiorQual"]:
        if cat_col in df_encoded.columns:
            df_encoded[cat_col + "_num"] = df_encoded[cat_col].astype(str).str.len()

    return df_encoded


# ---------------------------------------------------------------------
# Clean full dataset (required to fully implement for Task 1.1)
# ---------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform complete cleaning pipeline:
    - Remove invalid rows (e.g., livingSpace <= 10 && > 1000)
    - Fill missing values (median for numeric, mode for categorical)
    - Encode categorical features
    - Keep only numeric columns
    """
    

    df_clean = df.copy()
    print("Dataframe before cleaning: ", df_clean.shape)
    #print(df_clean.describe())
    print("Cleaning Data....")

    # Filter invalid rows
    if "livingSpace" in df_clean.columns:
        df_clean = df_clean[(df_clean["livingSpace"] > 10) & (df_clean['livingSpace'] < 1000)]
    print("Only regarding livingSpace > 10 m² & < 1000m² Shape: ", df_clean.shape)
    
    # Fill serviceCharge with Median
    print("Missing Values serviceCharge", df_clean['serviceCharge'].isnull().sum())
    df_clean['serviceCharge'] = df['serviceCharge'].fillna(df_clean['serviceCharge'].median())
    print("Missing values serviceCharge after filling:", df_clean['serviceCharge'].isnull().sum())

    # Fill totalRent by baseRent + serviceCharge
    df_clean['totalRent'] = df['totalRent'].fillna(df_clean["baseRent"]+ df_clean["serviceCharge"])    

    for col in df_clean.select_dtypes(include=[np.number]).columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    df_clean['condition'] = df_clean['condition'].fillna("unknown")

    # Fill non-numeric columns (mode)
    for col in df_clean.select_dtypes(exclude=[np.number]).columns:
        try:

            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        except Exception:
            df_clean[col] = df_clean[col].fillna("unknown")



    # Encode categorical columns
    df_clean = encode_categorical(df_clean)

    # Keep only numeric columns
    df_clean = df_clean.select_dtypes(include=[np.number])
    #Remove Columns with to many NaNs
    print("Removing columns with to many NaNs")
    df_clean.drop(['telekomHybridUploadSpeed', 'heatingCosts', 'electricityBasePrice', 'electricityKwhPrice'], axis = 1, inplace = True)

    # Final safety check
    df_clean = df_clean.dropna(axis=0, how="any")
    print("Dataframe nach dem Cleanen: ", df_clean.shape)
    return df_clean

# ---------------------------------------------------------------------
# Inspect missing values
#
# Function showcasing further panda DataFrame handling.
# ---------------------------------------------------------------------
def inspect_missing_values(df: pd.DataFrame):
    """
    Returns a summary of missing values per column.
    Useful for exploration and debugging.
    """
    missing = df.isnull().sum()
    total = len(df)
    percent = (missing / total * 100).round(2)
    summary = pd.DataFrame({"missing": missing, "percent": percent})
    return summary[summary["missing"] > 0].sort_values(by="percent", ascending=False)
