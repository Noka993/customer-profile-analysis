import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import os


def read_preprocessed_data(file_name="marketing_campaign.csv", std=True, le=True):
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

    if std:
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

        scaler = StandardScaler()
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

        # df.to_csv("marketing_campaign_scaled.csv", index=False)

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

    # Usuwamy kolumny, których nie potrzebujemy do analizy
    del (
        df["Z_CostContact"],
        df["Z_Revenue"],
        df["ID"],
        df["Year_Birth"],
        df["Dt_Customer"],
    )

    if le:
        # Dane kategoryczne
        s = df.dtypes == "object"
        object_cols = list(s[s].index)

        # Przekształcamy na dane numeryczne
        LE = LabelEncoder()
        for i in object_cols:
            df[i] = df[[i]].apply(LE.fit_transform)
    
    print(df.head)

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


if __name__ == "__main__":
    df = read_preprocessed_data()
    print(df["Age"].head())


