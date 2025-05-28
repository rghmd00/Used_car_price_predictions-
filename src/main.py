# src/pipeline.py
from src.data_load import load_csv
from src.preprocess import wrangle
from src.train import train_model
from eda import explatory_data_analysis
from src.validation import validate_data
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    train_raw = load_csv(cfg.data.path)
    explatory_data_analysis(cfg.data.path)
    validate_data(cfg.data.path, cfg)
    preprocessed_df = wrangle(train_raw)

    train_model(preprocessed_df,cfg)
    
if __name__ == "__main__":
    main()
