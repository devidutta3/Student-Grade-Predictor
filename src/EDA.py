import pandas as pd


def load_dataset(path):
    return pd.read_csv(path)


def dataset_info(df):
    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nDataset Shape:")
    print(df.shape)

    print("\nDataset Info:")
    print(df.info())

    print("\nSummary Statistics:")
    print(df.describe())


def main():
    df = load_dataset("data/student_data.csv")
    dataset_info(df)


if __name__ == "__main__":
    main()