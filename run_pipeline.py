import pandas as pd
import sys

from src.data_validation import validate_data
from src.feature_engineering import engineer_features
from src.preprocessing import split_data
from src.train import train_model
from src.evaluate import evaluate_with_threshold

# ----------------------------
# MODE: before or after
# ----------------------------
MODE = sys.argv[1]  # "before" or "after"

# Load data
df = pd.read_csv("data/fraud.csv")

# Validate & clean
df = validate_data(df)

# Feature engineering
df = engineer_features(df)

# Split
X_train, X_test, y_train, y_test = split_data(df)

# Train
model, cv_f1 = train_model(X_train, y_train)

print("CV F1:", cv_f1)

# Evaluation
if MODE == "before":
    f1 = evaluate_with_threshold(
        model,
        X_test,
        y_test,
        threshold=0.5,   # default threshold
        file_name="reports/before.json"
    )
    print("Test F1 (Before):", f1)

elif MODE == "after":
    f1 = evaluate_with_threshold(
        model,
        X_test,
        y_test,
        threshold=0.3,   # tuned threshold
        file_name="reports/after.json"
    )
    print("Test F1 (After):", f1)

else:
    raise ValueError("Mode must be 'before' or 'after'")
