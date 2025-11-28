"""
Polynomial Modeling and Model Selection (reference implementation for task 3)

Dummy version - contains only hardcoded values: returns fixed example data 
structures in the correct format so that tests and example notebooks can 
run successfully.
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from src.baseline_model import train_baseline_model, evaluate_model

def get_candidate_features():
    """
    Return a fixed list of Top 3 features (obtained from earlier tasks)
    """
    return [
        "livingSpace",
        "geo_plz",
        "lift_num",
    ]


def build_polynomial_design_matrix(X_full, n_features, degree):
    """
    Given is an array of ordered features as a numpy array, 
    a feature list, and a polynomial degree:
    Goal is to return the design matrix:
    - extract the first n_features (gives the number of features)
    - expands them into polynomial features (of degree degree)
    Returns X_poly
    The method can be called for X_train or X_val or X_test.
    """

    # Extract the first n_features columns
    X_sub = X_full[:, :n_features]
    poly = PolynomialFeatures(degree=degree,include_bias=False)
    X_poly = poly.fit_transform(X_sub)
    return X_poly

# ---------------------------------------------------------------------
# Polynomial analysis
# ---------------------------------------------------------------------
def evaluate_polynomial_models(X_train=None, y_train=None, X_val=None, y_val=None, feature_names=None, max_degree=6):
    """
    Evaluate polynomial regression models for degrees 1–6.
    Dummy function returning example results for polynomial model evaluation.

    Parameters
    ----------
    X_train, X_val : np.ndarray
        Training and validation input features.
    y_train, y_val : np.ndarray
        Target variables.
    feature_names : list of str
        Names of the features used. T
        This is required to name the used features in the list.
    max_degree : int
        Maximum polynomial degree to evaluate.

    Returns
    -------
    list of dict
        [
            {"features": ["livingSpace"], "degree": 1, "r2_train": ..., "r2_val": ..., "rmse_train": ..., "rmse_val": ...},
            ...
        ]
        """
    results = []
    total_features = len(feature_names)
    for n_feat in range(1,total_features+1):
        features_tmp = feature_names[:n_feat]
        for degree in range(1,max_degree + 1):
            X_train_mat = build_polynomial_design_matrix(X_train, n_feat, degree)
            X_val_mat = build_polynomial_design_matrix(X_val, n_feat, degree)

            model = train_baseline_model(X_train_mat, y_train)
            train_results = evaluate_model(model, X_train_mat,y_train)
            val_results = evaluate_model(model, X_val_mat,y_val)
            results.append({
                'features': features_tmp,
                'degree': degree,
                'r2_train': train_results['r2'],
                'rmse_train': train_results['rmse'],
                'r2_val': val_results['r2'],
                'rmse_val': val_results['rmse']
            })

    return results


""" 
    dummy_results = [
        {"features": ["livingSpace"], "degree": 1, "r2_train": 0.68, "r2_val": 0.63, "rmse_train": 240, "rmse_val": 260},
        {"features": ["livingSpace"], "degree": 2, "r2_train": 0.79, "r2_val": 0.73, "rmse_train": 210, "rmse_val": 220},
        {"features": ["livingSpace"], "degree": 3, "r2_train": 0.85, "r2_val": 0.77, "rmse_train": 180, "rmse_val": 205},
        {"features": ["livingSpace"], "degree": 4, "r2_train": 0.90, "r2_val": 0.75, "rmse_train": 160, "rmse_val": 230},
        {"features": ["livingSpace"], "degree": 5, "r2_train": 0.94, "r2_val": 0.68, "rmse_train": 140, "rmse_val": 270},
        {"features": ["livingSpace"], "degree": 6, "r2_train": 0.96, "r2_val": 0.61, "rmse_train": 125, "rmse_val": 320},
    ]
    print("Dummy polynomial results returned.")
    return dummy_results """

#---------------------------------------------------------------------
# Analyze Performance of Polynomial Models (Task 3.1)
# ---------------------------------------------------------------------
def analyze_polynomial_performance(poly_results):
    """
    Analyze polynomial model evaluation results and determine
    the optimal and overfitting degrees for each feature combination.

    Parameters
    ----------
    poly_results : list of dict
        Output of evaluate_polynomial_models(), containing keys:
        'features', 'degree', 'r2_val', 'rmse_val', etc.

    Returns
    -------
    list of dict
        [
            {
                "features": ["livingSpace"],
                "optimal_degree": 3,
                "optimal_r2_val": 0.77,
                "optimal_rmse_val": 205,
                "overfit_degree": 5,
                "overfit_r2_val": 0.68,
                "overfit_rmse_val": 270
            },
            ...
        ]
    """
    summary = []
    for entry in poly_results:
        continue



    # Task 3.1: You have to iterate over the poly_results structure
    # (which comes from evaluate_polynomial_models)
    # group entries by feature combination
    dummy_summary = [
        {
            "features": ["livingSpace"],
            "optimal_degree": 3,
            "optimal_r2_val": 0.77,
            "optimal_rmse_val": 205,
            "overfit_degree": 5,
            "overfit_r2_val": 0.68,
            "overfit_rmse_val": 270
        },
        {
            "features": ["livingSpace", "noRooms"],
            "optimal_degree": 2,
            "optimal_r2_val": 0.82,
            "optimal_rmse_val": 190,
            "overfit_degree": 5,
            "overfit_r2_val": 0.71,
            "overfit_rmse_val": 255
        }
    ]

    return dummy_summary

# ---------------------------------------------------------------------
# Task 3.4 — Dummy function for “best model” selection
# ---------------------------------------------------------------------
def get_best_polynomial_model(results_list):
    """
    Dummy implementation returning a fixed ‘best model’ entry.

    Students should later replace this with a real selection from
    the evaluated results (e.g., by highest validation R²).
    """
    """best_model = {
        "degree": 2,
        "features": ["livingSpace", "numberOfRooms"],
        "r2_train": 0.7,
        "r2_val": 0.7,
        "rmse_val": 160,
    }"""
    best_model = results_list[0]
    for model in results_list:
        if model["r2_val"] > best_model["r2_val"]:
            best_model = model

    return best_model
