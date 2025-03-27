import pandas as pd


def read_preprocessed_data():
    df = pd.read_csv("../marketing_campaign.csv", sep="\t")
    df.head()

    # Brakujące dane
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0] 

    print(missing_values)

    # Zamiana dat na ilość dni bycia klientem
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y', errors='coerce')
    reference_date = pd.to_datetime('08-02-2021', format='%d-%m-%Y', errors='coerce')
    df['Dt_Customer'] = (reference_date - df['Dt_Customer']).dt.days

    # Usunięcie niepotrzebnych statusów
    df = df.drop(df[df['Marital_Status'].isin(['YOLO', 'Absurd'])].index)
    df['Marital_Status'].value_counts()

    # Usuwamy kolumny, których nie potrzebujemy do analizy
    del df['Z_CostContact'], df['Z_Revenue'], df["ID"]

    return df
