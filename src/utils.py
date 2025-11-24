import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error

def load_df(csv_path):
    df = pd.read_csv(csv_path)
    print("Dataframe loaded")
    return df

def feature_best_fit_list(df_clean, features_used, features_all):
    #print("Creating list for Features assendin by ascending R² value...")
    possible_features_dict = []
    for col_name in features_all:
        if col_name['feature'] in ['serviceCharge','baseRent'] + features_used:
            continue
        feature_cols_tmp = features_used + [col_name['feature']]
        df_tmp = df_clean.dropna(subset=feature_cols_tmp + ['totalRent'])
        X_train = df_tmp[feature_cols_tmp].values
        y_train = df_tmp['totalRent'].values
        model_multi = LinearRegression()
        model_multi.fit(X_train, y_train)
        y_train_pred = model_multi.predict(X_train)
        train_r2 = r2_score(y_train, y_train_pred)
        train_rmse = root_mean_squared_error(y_train,y_train_pred)
        possible_features_dict.append({'feature': col_name['feature'], 'r2': train_r2, 'rmse': train_rmse}) 
        
    
    possible_features_dict.sort(key=lambda x: x['r2'], reverse=True)
    return possible_features_dict  