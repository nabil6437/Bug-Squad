# Accuracy Comparison Report

## Decision Tree Model vs Rule-Based Expert System

| Metric      | Decision Tree | Expert System |
|-------------|--------------|---------------|
| Accuracy    | 0.7705       | ~0.64         |
| Precision   | 0.7500       | ~0.57         |
| Recall      | 0.7500       | ~0.93         |
| F1-Score    | 0.7500       | ~0.70         |

> Decision Tree metrics measured on held-out 20% test set (61 samples, random_state=42).
> Expert System metrics are approximate, evaluated on same test records using risk-level thresholding.

## Decision Tree Configuration
- **Best Parameters**: criterion=entropy, max_depth=5, min_samples_leaf=4, min_samples_split=2
- **Tuning Method**: GridSearchCV (5-fold CV, scoring=F1)
- **Train / Test Split**: 80% / 20% (stratified)
- **Preprocessing**: Duplicate removal, missing value imputation, IQR capping, One-Hot Encoding, MinMaxScaler

## Analysis

### Decision Tree Model
- Learns patterns directly from data (305 unique records after deduplication)
- Handles complex feature interactions automatically via OHE + scaling
- Hyperparameter-tuned with GridSearchCV for optimal generalization
- Moderate interpretability via feature importance

### Rule-Based Expert System
- Uses 20 human-defined clinical rules (10 HIGH, 5 MEDIUM, 5 LOW)
- Fully transparent and explainable — every decision is traceable
- No training data required
- High recall (catches most disease cases) at the cost of lower precision

### Conclusion
The Decision Tree achieves **77.1% accuracy** while the Expert System achieves approximately **64% accuracy**.
The Expert System's higher recall makes it more suitable as a screening tool,
while the Decision Tree offers better overall balanced performance.
