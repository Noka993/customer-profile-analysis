import pandas as pd
from sklearn.preprocessing import StandardScaler


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
