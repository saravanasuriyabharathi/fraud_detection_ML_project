import numpy as np

def engineer_features(df):
    df = df.copy()

    df["Amount_log"] = np.log1p(df["Amount"])
    df["Amount_ratio"] = df["Amount"] / df["Amount"].mean()
    df["High_amount"] = (df["Amount"] > 200).astype(int)
    df["Amount_squared"] = df["Amount"] ** 2

    return df
