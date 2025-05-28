import matplotlib.pyplot as plt
import pandas as pd

def explatory_data_analysis(path):
    df = pd.read_csv(path)
    plt.figure(figsize=(10, 6))
    plt.hist(df['price'], bins=50, color='blue', alpha=0.7)
    plt.title('Distribution of Car Prices')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()


    top_brands = df['brand'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    top_brands.plot(kind='bar', color='orange', alpha=0.7)
    plt.title('Top 10 Car Brands by Frequency')
    plt.xlabel('Brand')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.75)
    plt.show()



    top_fuel_types = df['fuel_type'].value_counts().head(5)
    plt.figure(figsize=(8, 5))
    top_fuel_types.plot(kind='bar', color='green', alpha=0.7)
    plt.title('Top 5 Fuel Types by Frequency')
    plt.xlabel('Fuel Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.75)
    plt.show()