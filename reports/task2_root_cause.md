## Task 2 – Root Cause Analysis

### Observed Issues
- Low and unstable F1 score
- Poor recall for fraud class
- High variance across evaluation runs

### Root Causes

1. **Severe class imbalance**
   - Fraud cases are extremely rare (~0.2%)
   - Model becomes biased towards majority class

2. **Default decision threshold**
   - Threshold = 0.5 is unsuitable for fraud detection
   - Causes most fraud cases to be missed

3. **Model sensitivity to rare samples**
   - Logistic Regression struggles with sparse positive class
   - Small data changes lead to prediction instability

4. **Evaluation sensitivity**
   - F1 score heavily impacted by small changes in fraud predictions
