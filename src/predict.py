import os
import pickle
import pandas as pd
from src.preprocess import wrangle
from src.train import train_model
from src.data_load import load_csv
from src.eda import explatory_data_analysis
from src.validation import validate_data
from omegaconf import OmegaConf

MODEL_PATH = "/teamspace/studios/this_studio/Used_car_price_predictions-/models/lgb_model.pkl"
CONFIG_PATH = "/teamspace/studios/this_studio/Used_car_price_predictions-/configs/config.yaml"  

def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
            print("Model loaded from file.")
    else:
        print("No trained model found. Training a new one...")
        cfg = OmegaConf.load(CONFIG_PATH)
        df = load_csv(cfg.data.path)
        explatory_data_analysis(cfg.data.path)
        validate_data(cfg.data.path, cfg)
        pre_df = wrangle(df)
        train_model(pre_df, cfg)
        with open(cfg.model.save_path, "rb") as f:
            model = pickle.load(f)
    return model

def predict_price(input_dict: dict):
    df = pd.DataFrame([input_dict])
    df_processed = wrangle(df)
    model = load_or_train_model()
    prediction = model.predict(df_processed)[0]
    return prediction
