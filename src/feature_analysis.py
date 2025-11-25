"""
Implements the feature ranking and stepwise selection procedures
for Task 2 of the ENN Minilab project.

This module provides functions for:
  - Compute single-feature performance (analyze_single_features)
  - Implement stepwise feature selection (stepwise_selection)
  
Current version: Dummy implementation returning example data
so that tests can be applied (but will fail).
"""

import pandas as pd
import numpy as np
from src.utils import feature_best_fit_list
from src.baseline_model import train_baseline_model, evaluate_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
# ---------------------------------------------------------------------
# Single-feature analysis (Task 2.1)
# ---------------------------------------------------------------------
def analyze_single_features(df: pd.DataFrame, target_col: str = "totalRent"):
    """
    Analyze each feature independently to assess its correlation
    (R² performance) with the target variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset including numeric and encoded categorical columns.
    target_col : str, default="totalRent"
        Name of the target variable.
    
    Returns
    -------
    list of dict
        Example format:
        [
            {"feature": "livingSpace", "r2": 0.75},
            {"feature": "numberOfRooms", "r2": 0.68},
            ...
        ]
        Sorted in descending order by R².
    """
    print("Creating Dict of feature correlation to totalRent...")
    results = []

    for col in df:
        if col in ['totalRent','baseRent','serviceCharge','baseRentRange']: continue

        df_tmp = df.dropna(subset=[col, "totalRent"])
        X = df_tmp[[col]].to_numpy()
        y = df_tmp['totalRent'].to_numpy()
        model_full = LinearRegression().fit(X, y)
        y_predict = model_full.predict(X)
        r2_full = round(r2_score(y, y_predict),4)
        rmse = root_mean_squared_error(y, y_predict)
        results.append((col, r2_full,rmse))

    results.sort(key=lambda x: x[1], reverse=True)
    results_dict = [{"feature": f, "r2": r, 'rmse': m} for f,r,m in results]    
    return results_dict


# ---------------------------------------------------------------------
# Stepwise feature selection (Task 2.2)
# ---------------------------------------------------------------------
def stepwise_selection(df_train: pd.DataFrame, df_val: pd.DataFrame, flag: bool=False):
    """
    Simulate a stepwise feature selection process that gradually adds features
    and evaluates the model performance.

    Parameters
    ----------
    df_train : pd.DataFrame
        Cleaned training dataset.
    df_val : pd.DataFrame
        Cleaned validation dataset.

    Returns
    -------
    list of dict
        Example format:
        [
            {"n_features": 1, "features": ["livingSpace"], "r2": 0.70, "rmse": 250.0},
            {"n_features": 2, "features": ["livingSpace", "numberOfRooms"], "r2": 0.78, "rmse": 210.0},
            ...
        ]
    """
    feature_list = analyze_single_features(df_train)
    initial_feature = feature_list[0]
    features_red_list = [initial_feature['feature']]
    X_train_ = df_train[features_red_list].to_numpy()
    y_train_ = df_train['totalRent'].to_numpy()
    X_val_ = df_val[features_red_list].to_numpy()
    y_val_ = df_val["totalRent"].to_numpy()
    model_ = train_baseline_model(X_train_, y_train_)
    eval_result_ = evaluate_model(model_, X_val_, y_val_)
    features_used = [{"n_features": "1", "features": [initial_feature['feature']], "r2": eval_result_['r2'], 'rmse': eval_result_['rmse']}]
    
    for i in range (15):
        new_feature_list = feature_best_fit_list(df_train, df_val, features_red_list, feature_list)
        if new_feature_list[0]['r2'] < features_used[-1]['r2'] and flag == False:
            print(f"Neues Feature: {new_feature_list[0]['r2']} < aktuelle Feature: {features_used[-1]['r2']}")
            break
        features_red_list.append(new_feature_list[0]['feature'])
        features_used.append({"n_features": i+2, "features": features_red_list.copy(), 'r2': new_feature_list[0]['r2'], 'rmse': new_feature_list[0]['rmse']})
    
    print(features_used)    
    return features_used
