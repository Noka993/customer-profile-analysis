import pandas as pd
import os


def read_preprocessed_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "../data/marketing_campaign.csv")
    df = pd.read_csv(file_path, sep="\t")
    df.head()

    # Brakujące dane
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]

    print(missing_values)

    # Zamiana dat na ilość dni bycia klientem
    df["Dt_Customer"] = pd.to_datetime(
        df["Dt_Customer"], format="%d-%m-%Y", errors="coerce"
    )
    reference_date = pd.to_datetime("08-02-2021", format="%d-%m-%Y", errors="coerce")
    df["Dt_Customer"] = (reference_date - df["Dt_Customer"]).dt.days

    # Ujednolicenie statusów cywilnych
    df["Living_With"] = df["Marital_Status"].replace(
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

    # Usuwamy kolumny, których nie potrzebujemy do analizy
    del df["Z_CostContact"], df["Z_Revenue"], df["ID"]

    return df
