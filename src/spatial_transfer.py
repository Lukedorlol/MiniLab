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
    
    # # Removes total_rent, serviceCharge and baseRent
    # #print("--------------------------------")
    # #print(X_train)
    # #print(X_val)
    # X_train = np.delete(X_train, [0, 4, 9], axis = 0)
    # X_val = np.delete(X_val, [0, 4, 9], axis = 0)
    # y_train = np.delete(y_train, [0, 4, 9], axis = 0)
    # y_val = np.delete(y_val, [0, 4, 9], axis = 0)

    # #print("------------------------")
    # #print(X_train)
    # #print(X_val)


    X_train_mat = build_polynomial_design_matrix(X_train, 10, 1)
    X_val_mat = build_polynomial_design_matrix(X_val, 10, 1)



    model = train_baseline_model(X_train_mat, y_train)
    
    X_train_results = evaluate_model(model, X_train_mat,y_train)
    X_val_results = evaluate_model(model, X_val_mat,y_val)

    return {
        "r2_train": X_train_results["r2"],
        "rmse_train": X_train_results["rmse"],
        "r2_val": X_val_results["r2"],
        "rmse_val": X_val_results["rmse"]
    }

def run_cross_domain_evaluation_spatial(X_MS_train,y_MS_train,X_MS_val,y_MS_val,
                                X_BI_train,y_BI_train,X_BI_val,y_BI_val):
    return {
        "MS_train_MS_val":     train_and_eval(X_MS_train,y_MS_train,X_MS_val,y_MS_val),
        "MS_train_BI_val":     train_and_eval(X_MS_train,y_MS_train,X_BI_val,y_BI_val),
        "BI_train_BI_val":     train_and_eval(X_BI_train,y_BI_train,X_BI_val,y_BI_val),
    }