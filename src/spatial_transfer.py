"""
Spatial Transfer

Functions that shall apply the LinearRegression model
on the different training and validation sets 
(testing transfer between different cities).
"""

import numpy as np
from src.baseline_model import BaselineLinearModel, train_baseline_model, evaluate_model
from sklearn.metrics import r2_score, root_mean_squared_error
from src.polynomial_analysis import build_polynomial_design_matrix


def train_and_eval(X_train,y_train,X_val,y_val):
    # Apply training to regression model
    # and afterwards evaluate on R2 and RMSE for both data sets

    model = BaselineLinearModel().fit(X_train,y_train)
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    return {
        "r2_train": float(r2_score(y_train, y_pred_train)),
        "rmse_train": float(root_mean_squared_error(y_train, y_pred_train)),
        "r2_val": float(r2_score(y_val, y_pred_val)),
        "rmse_val": float(root_mean_squared_error(y_val, y_pred_val)),
    }

def run_cross_domain_evaluation_spatial(X_MS_train,y_MS_train,X_MS_val,y_MS_val,
                                X_BI_train,y_BI_train,X_BI_val,y_BI_val):
    return {
        "MS_train_MS_val":     train_and_eval(X_MS_train,y_MS_train,X_MS_val,y_MS_val),
        "MS_train_BI_val":     train_and_eval(X_MS_train,y_MS_train,X_BI_val,y_BI_val),
        "BI_train_BI_val":     train_and_eval(X_BI_train,y_BI_train,X_BI_val,y_BI_val),
    }