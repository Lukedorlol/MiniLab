import pandas as pd

def load_df(csv_path):
    df = pd.read_csv(csv_path)
    print("Dataframe loaded")
    return df