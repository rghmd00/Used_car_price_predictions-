import lightgbm as lgb
import pickle

def train_model(df_train):
    # Ensure categorical columns are labeled properly
    for col in df_train.select_dtypes(include='object').columns:
        df_train[col] = df_train[col].astype('category')

    X = df_train.drop(columns='price')
    y = df_train['price']

    model = lgb.LGBMRegressor()
    model.fit(X, y)
    print("Model trained successfully.")
    # Save the model
    with open('models/lgb_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model
    



