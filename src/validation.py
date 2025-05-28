import pandas as pd


def validate_data(df_path,cfg):
    df = pd.read_csv(df_path)
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())
    print("Data validation completed.")