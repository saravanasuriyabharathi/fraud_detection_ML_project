from sklearn.metrics import f1_score, classification_report
import json
import numpy as np

def evaluate_with_threshold(model, X_test, y_test, threshold, file_name):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > threshold).astype(int)

    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    with open(file_name, "w") as f:
        json.dump(report, f, indent=4)

    return f1
