import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler
import os


def read_preprocessed_data(file_name="marketing_campaign.csv", std=True, minmax=False, robust=False, le=True, he=False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)
    file_path = os.path.join(root_dir, "data/", file_name)
    print(file_path)
    df = pd.read_csv(file_path, sep="\t")

    # Brakujące dane
    df = df.dropna()

    # Bierzemy 2021, bo wtedy był ostatnio modyfikowany plik
    df["Age"] = 2021 - df["Year_Birth"]

    # Zamiana dat na ilość dni bycia klientem
    df["Dt_Customer"] = pd.to_datetime(
        df["Dt_Customer"], format="%d-%m-%Y", errors="coerce"
    )
    reference_date = pd.to_datetime("08-02-2021", format="%d-%m-%Y", errors="coerce")
    df["Dt_Customer"] = (reference_date - df["Dt_Customer"]).dt.days

    # Dodajemy sumę wydanych pieniędzy w okresie dwóch lat
    df["Spent"] = (
        df["MntWines"]
        + df["MntFruits"]
        + df["MntMeatProducts"]
        + df["MntFishProducts"]
        + df["MntSweetProducts"]
        + df["MntGoldProds"]
    )

    columns_to_scale = [
        "Age",
        "Income",
        "Recency",
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
        "NumDealsPurchases",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
        "NumWebVisitsMonth",
        "Dt_Customer",
        "Spent"
    ]

    df = remove_outliers(df)

    # Ujednolicenie statusów cywilnych
    df["Marital_Status"] = df["Marital_Status"].replace(
        {
            "Married": "Partner",
            "Together": "Partner",
            "Absurd": "Alone",
            "Widow": "Alone",
            "YOLO": "Alone",
            "Divorced": "Alone",
            "Single": "Alone",
        }
    )

    # Ujednolicenie edukacji
    df["Education"] = df["Education"].replace(
        {
            "Basic": "Undergraduate",
            "2n Cycle": "Undergraduate",
            "Graduation": "Graduate",
            "Master": "Postgraduate",
            "PhD": "Postgraduate",
        }
    )

    if he:
        # Dane kategoryczne, one hot encoding
        s = df.dtypes == "object"
        object_cols = list(s[s].index)

        # Przekształcamy na dane numeryczne
        df = pd.get_dummies(df, columns=object_cols)
    
    if le:
        le = LabelEncoder()
        s = df.dtypes == "object"
        object_cols = list(s[s].index)

        for col in object_cols:
            df[col] = le.fit_transform(df[col])

    if std:
        scaler = StandardScaler()
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    if minmax:
        scaler = MinMaxScaler()
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    if robust:
        scaler = RobustScaler()
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    del (
        df["Z_CostContact"],
        df["Z_Revenue"],
        df["ID"],
        df["Year_Birth"],
        df["Dt_Customer"],
    )

    return df

def outliers_statistics(df):

    outliers_count = []

    for col in df.columns:
        if df[col].dtypes in [float, int]:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            IQR = q3 - q1
            outliers = df[(df[col] < (q1 - 1.5 * IQR)) | (df[col] > (q3 + 1.5 * IQR))]
            outliers_count.append(outliers.shape[0])
        else:
            outliers_count.append(0)

    outliers_percentage = []
    for value in outliers_count:
        outliers_percentage.append(float(value)/len(df))

    outliers_df = pd.DataFrame([outliers_count, outliers_percentage], columns=df.columns)
    outliers_df.index = ['Ilość wartości skrajnych','Procent wartości skrajnych']

    return outliers_df

def remove_outliers(df2):
    outliers_percentages=outliers_statistics(df2)
    df=df2.copy()
    for i, col in enumerate(df.columns):
        if col in ['Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Age', 'Spent']:
            if df[col].dtypes in [float, int]:
                q1 = df2[col].quantile(0.25)
                q3 = df2[col].quantile(0.75)
                IQR = q3 - q1
                upper_bound=(q1 - 1.5 * IQR)
                lower_bound=(q3 + 1.5 * IQR)
                upper_limit = df2[col].mean() + 3*df2[col].std()
                lower_limit = df2[col].mean() - 3*df2[col].std()
                outliers_percentage = outliers_percentages.iloc[1,i]
                if outliers_percentage < 0.05:
                    df[col] = df[col].where(~((df[col] < (q1 - 1.5 * IQR)) | ( df[col] > (q3 + 1.5 * IQR))))
                else:
                    df[col] = df[col].clip(lower=lower_limit,upper=upper_limit)
    cleaned_df=df.dropna()

    return cleaned_df
def general_statistics(df):
    stat_cols = [col for col in df.columns if df[col].nunique()>9]
    num_cols = pd.DataFrame(df,columns= stat_cols)#df.select_dtypes(include=['int', 'float'])
    statystyki = {
    'Średnia':num_cols.mean(),
    'Mediana':num_cols.median(),
    'Minimum':num_cols.min(),
    'Maksimum':num_cols.max(),
    'Odchylenie Standardowe':num_cols.std(),
    'Skośność':num_cols.skew()
    }
    return statystyki
def jaccard_res(df1,df2):
    
    jaccard_values = []
    
    for c1 in clusters1.columns:
        for c2 in clusters2.columns:
            intersection = len(c1.intersection(c2))
            union = len(c1.union(c2))
            if union > 0:  
                jaccard_values.append(intersection / union)
    
    # Return the maximum Jaccard index (best matching)
    return max(jaccard_values) if jaccard_values else 0.0
if __name__ == "__main__":
    df = read_preprocessed_data()
    print(df.info())


