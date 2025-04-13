import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv("C:/Users/weron/Studia/Semestr4/Metody analizy danych/marketing_campaign.csv", sep='\t')
data.columns = data.columns.str.strip()
data['Age'] = 2021 - data['Year_Birth']

continuous_vars = [
    'Age', 'Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts',
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
    'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth'
]

continuous_data = data[continuous_vars]

minmax_scaler = MinMaxScaler()
continuous_minmax = minmax_scaler.fit_transform(continuous_data)

standard_scaler = StandardScaler()
continuous_standard = standard_scaler.fit_transform(continuous_data)

robust_scaler = RobustScaler()
continuous_robust = robust_scaler.fit_transform(continuous_data)

categorical_vars = [
    'Kidhome', 'Teenhome', 'Complain', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3',
    'AcceptedCmp4', 'AcceptedCmp5', 'Response'
]

encoder = OneHotEncoder(drop='first', sparse_output=False)
categorical_data = data[categorical_vars]
encoded_categorical_data = encoder.fit_transform(categorical_data)
encoded_categorical_df = pd.DataFrame(encoded_categorical_data, columns=encoder.get_feature_names_out(categorical_vars))

continuous_data_minmax_df = pd.DataFrame(continuous_minmax, columns=continuous_vars)
continuous_data_standard_df = pd.DataFrame(continuous_standard, columns=continuous_vars)
continuous_data_robust_df = pd.DataFrame(continuous_robust, columns=continuous_vars)

corr_minmax = continuous_data_minmax_df.corr()
corr_standard = continuous_data_standard_df.corr()
corr_robust = continuous_data_robust_df.corr()
corr_encoded_categorical = encoded_categorical_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_minmax, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=0.5)
plt.title("Macierz korelacji po normalizacji min-max")
plt.tight_layout()
plt.savefig("C:/Users/weron/Studia/Semestr4/Metody analizy danych/macierze_korelacji/proper/matrix_minmax.png")

plt.figure(figsize=(12, 10))
sns.heatmap(corr_standard, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=0.5)
plt.title("Macierz korelacji po standaryzacji")
plt.tight_layout()
plt.savefig("C:/Users/weron/Studia/Semestr4/Metody analizy danych/macierze_korelacji/proper/matrix_standard.png")

plt.figure(figsize=(12, 10))
sns.heatmap(corr_robust, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=0.5)
plt.title("Macierz korelacji po normalizacji RobustScaler")
plt.tight_layout()
plt.savefig("C:/Users/weron/Studia/Semestr4/Metody analizy danych/macierze_korelacji/proper/matrix_robust.png")

plt.figure(figsize=(12, 10))
sns.heatmap(corr_encoded_categorical, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=0.5)
plt.title("Macierz korelacji po One-Hot Encoding dla zmiennych kategorycznych")
plt.tight_layout()
plt.savefig("C:/Users/weron/Studia/Semestr4/Metody analizy danych/macierze_korelacji/proper/matrix_one_hot.png")

