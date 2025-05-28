import pandas as pd

def wrangle(DataFrame, fuel_type_map=None):
    df = DataFrame.copy()
    
    if fuel_type_map is None:
        fuel_type_map = {
            'Tesla': 'Electric',  
            'Rivian': 'Electric',
            'Lucid': 'Electric',
            'Ford': 'Gasoline',  
            'Dodge': 'Gasoline',
            'Porsche': 'Gasoline',
            'Audi': 'Gasoline',
            'Nissan': 'Gasoline',  
            'Mazda': 'Gasoline',
            'Mercedes-Benz': 'Gasoline',  
            'BMW': 'Gasoline',
            'Kia': 'Gasoline',
            'Chevrolet': 'Gasoline',
            'Cadillac': 'Gasoline',
            'Chrysler': 'Gasoline',
            'Volkswagen': 'Gasoline',
            'Hyundai': 'Gasoline', 
            'Jeep': 'Gasoline',
            'GMC': 'Gasoline',
            'Acura': 'Gasoline',
            'Volvo': 'Gasoline', 
            'Toyota': 'Gasoline', 
            'Honda': 'Gasoline', 
            'Jaguar': 'Gasoline',
            'McLaren': 'Gasoline',  
        }
    
    df['fuel_type'] = df.apply(
        lambda row: fuel_type_map.get(row['brand'], row['fuel_type']) if pd.isna(row['fuel_type']) else row['fuel_type'], 
        axis=1
    )
    df['fuel_type'] = df['fuel_type'].replace(['', '–', 'N/A', 'Not Available', 'NaN'], pd.NA)
    df['fuel_type'] = df['fuel_type'].fillna('Unknown')
    
    df.loc[df['fuel_type'].isin(["–", "not supported"]), 'fuel_type'] = df['brand'].map(fuel_type_map)
    
    df['accident'] = df['accident'].map({'None reported': 0, 'Reported': 1}) 
    df['accident'] = df['accident'].fillna(0).astype(int)
    
    def simplify_transmission(value):
        if pd.isna(value):
            return 'Unknown'
        elif 'manual' in value.lower() or "m/t" in value.lower():
            return 'M/T'
        elif 'automatic' in value.lower() or "a/t" in value.lower() or "cvt" in value.lower():
            return 'A/T'
        return 'Unknown'
    
    df['transmission'] = df['transmission'].apply(simplify_transmission)
    
    df['horsepower'] = df['engine'].str.extract(r'(\d+\.?\d*)HP').astype('float').fillna(-1)
    df['displacement'] = df['engine'].str.extract(r'(\d+(\.\d+)?)L')[0].astype(float).fillna(-1)
    
    df['age_of_car'] = 2025 - df['model_year']
    
    df = df.drop(columns=['clean_title', 'id'])

    df.to_csv('./data/processed/processed.csv', index=False)
    
    return df