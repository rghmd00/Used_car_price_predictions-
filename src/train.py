import lightgbm as lgb
import pickle

def train_model(df_train,cfg):
    # Ensure categorical columns are labeled properly
    for col in df_train.select_dtypes(include='object').columns:
        df_train[col] = df_train[col].astype('category')

    X = df_train.drop(columns=cfg.target_column)
    y = df_train[cfg.target_column]

    model = lgb.LGBMRegressor()
    model.fit(X, y)
    print("Model trained successfully.")
    # Save the model
    with open(cfg.model.save_path, 'wb') as f:
        pickle.dump(model, f)    

