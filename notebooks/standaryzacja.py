import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Min-max, można spróbować
# Opis metod i coś pokazać
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "../data/marketing_campaign.csv")
df = pd.read_csv("marketing_campaign.csv", sep='\t')


df['Age'] = 2025 - df['Year_Birth']


columns_to_scale = [
    'Age', 'Income', 'Recency',
    'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
    'MntSweetProducts', 'MntGoldProds',
    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
    'NumStorePurchases', 'NumWebVisitsMonth'
]


scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])


df_scaled.to_csv("marketing_campaign_scaled.csv", index=False)


print("Średnie po standaryzacji:")
print(df_scaled[columns_to_scale].mean())
print("\nOdchylenia standardowe po standaryzacji:")
print(df_scaled[columns_to_scale].std())

for x in columns_to_scale:
    print(df_scaled[x].head())
