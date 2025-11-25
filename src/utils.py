import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from src.baseline_model import train_baseline_model, evaluate_model

def load_df(csv_path):
    df = pd.read_csv(csv_path)
    print("Dataframe loaded")
    return df

def feature_best_fit_list(df_clean, df_val, features_used, features_all):
    #print("Creating list for Features assendin by ascending R² value on Validation set...")
    possible_features_dict = []
    for col_name in features_all:
        if col_name['feature'] in ['serviceCharge','baseRent'] + features_used:
            continue
        feature_cols_tmp = features_used + [col_name['feature']]
        df_tmp = df_clean.dropna(subset=feature_cols_tmp + ['totalRent'])
        X_train = df_tmp[feature_cols_tmp].to_numpy()
        y_train = df_tmp['totalRent'].to_numpy()
        X_val = df_val[feature_cols_tmp].to_numpy()
        y_val = df_val['totalRent'].to_numpy()
        model = train_baseline_model(X_train, y_train)
        eval_result = evaluate_model(model, X_val, y_val)
        possible_features_dict.append({'feature': col_name['feature'], 'r2': eval_result['r2'], 'rmse': eval_result['rmse']}) 
        
    
    possible_features_dict.sort(key=lambda x: x['r2'], reverse=True)
    return possible_features_dict  