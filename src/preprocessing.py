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
    - boolean features → {False: 0, True: 1}
    """
    df_encoded = df.copy()



    # categorical feature: 'condition'
    """ condition_mapping = {
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

     """

    print("Encoding categorical features...")
    # Boolean → 0/1
    bool_cols = df_encoded.select_dtypes(include=["bool"]).columns
    for col in bool_cols:
        df_encoded[col + "_num"] = df_encoded[col].astype(int)

    # Optional: encode heatingType or interiorQual if present
    for cat_col in ["condition","heatingType","interiorQual","firingTypes","typeOfFlat","regio3"]:
        if cat_col in df_encoded.columns:
            mean_prices = df_encoded.groupby(cat_col)["totalRent"].mean()
            mean_prices_sorted = mean_prices.sort_values()
            mapping = {kategorie: rang for rang, kategorie in enumerate(mean_prices_sorted.index)}
            df_encoded[cat_col + "_num"] = df_encoded[cat_col].map(mapping)
            median = int(len(mapping)/2)
            df_encoded[cat_col + "_num"] = df_encoded[cat_col + "_num"].fillna(median)
            
            #df_encoded[cat_col + "_num"] = df_encoded[cat_col].astype(str).str.len()

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
    print(f"Shape: {df_clean.shape}Cleaning Data....")

    # Filter invalid rows
    if "livingSpace" in df_clean.columns:
        df_clean = df_clean[(df_clean["livingSpace"] > 10) & (df_clean['livingSpace'] < 1000)]
    if 'geo_plz' in df_clean.columns:
        df_clean = df_clean[(df_clean['geo_plz'] < 48470)]
    # Fill serviceCharge with Median
    df_clean['serviceCharge'] = df['serviceCharge'].fillna(df_clean['serviceCharge'].median())

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
    df_clean.drop(['telekomHybridUploadSpeed', 'picturecount','heatingCosts', 'electricityBasePrice', 'electricityKwhPrice','scoutId'], axis = 1, inplace = True)

    # Final safety check
    df_clean = df_clean.dropna(axis=0, how="any")
    print("Dataframe cleaned, Shape: ", df_clean.shape)
    for col in df_clean.columns:
        print(col)
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
