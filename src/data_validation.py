def validate_data(df):
    # Check missing values
    if df.isnull().sum().sum() > 0:
        raise ValueError("Missing values found")

    # Handle duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"[INFO] Removing {duplicate_count} duplicate rows")
        df.drop_duplicates(inplace=True)

    # Check target column
    if "Class" not in df.columns:
        raise ValueError("Target column 'Class' missing")

    return df
