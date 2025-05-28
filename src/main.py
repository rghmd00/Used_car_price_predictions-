# src/pipeline.py
from src.data_load import load_csv
from src.preprocess import wrangle
from src.train import train_model


def main():
    # Step 1: Load data
    train_raw = load_csv("data/raw/train.csv")
    preprocessed_df = wrangle(train_raw)
    df_target = preprocessed_df['price']
    predictor = train_model(preprocessed_df)
    
if __name__ == "__main__":
    main()
