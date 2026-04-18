import pandas as pd


def load_data(path):

    data = pd.read_csv(path)

    return data


def preprocess_data(data):

    # Take sample for faster execution
    data = data.sample(n=20000, random_state=42)

    # Clean column names
    data.columns = data.columns.str.strip()

    # Target column
    target_column = "is_fraud"

    # Drop useless columns
    data = data.drop(
        [
            "Unnamed: 0",
            "trans_date_trans_time",
            "cc_num",
            "first",
            "last",
            "street",
            "trans_num",
        ],
        axis=1,
    )

    # Convert categorical → numerical
    data = pd.get_dummies(data, drop_first=True)

    # Split features & target
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    return X, y
