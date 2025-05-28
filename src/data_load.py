import pandas as pd
def load_csv(file_path: str) -> pd.DataFrame:   
    try:
        df = pd.read_csv(file_path)
        print(f"CSV file loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return pd.DataFrame()  

    