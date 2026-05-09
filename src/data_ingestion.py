import os
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

RAW_DATA_PATH = "data/raw/churn.csv"

def data_ingestion():
    """
    Load Telco Churn dataset from Kaggle and save to data/raw/
    """

    print("Data ingestion start...")

    # Load dataset from Kaggle
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "blastchar/telco-customer-churn",
        "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    )

    print(f"Data Loaded from Kaggle — Shape: {df.shape}")
    print(f"First 5 Records:\n{df.head()}")
    print(f"Null Values:\n{df.isnull().sum()}")
    print(f"Duplicated Rows: {df.duplicated().sum()}")

    # Create raw data directory
    os.makedirs("data/raw", exist_ok=True)

    # Save raw dataset
    df.to_csv(RAW_DATA_PATH, index=False)

    print(f"Raw data saved to {RAW_DATA_PATH}")

    return df


if __name__ == "__main__":
    data_ingestion()