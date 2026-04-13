# Heart Disease Detection System

A heart disease risk prediction system combining a Rule-Based Expert System (Experta) and a Decision Tree Classifier (Scikit-Learn), with an interactive Streamlit dashboard.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Train the Decision Tree Model
```bash
python ml_model/train_model.py
```

### Step 2: Launch the Streamlit App
```bash
streamlit run ui/app.py
```

### Step 3 (Optional): Run Expert System via CLI
```bash
python rule_based_system/expert_system.py
```

### Step 4 (Optional): Run Data Analysis
```bash
python notebooks/data_analysis.py
```

## Features

- **Decision Tree Classifier** with GridSearchCV hyperparameter tuning (80/20 split)
- **Rule-Based Expert System** with 20 clinical rules using Experta forward-chaining inference
- **Full Preprocessing Pipeline**: duplicate removal, missing value imputation, outlier capping (IQR), MinMaxScaler normalization
- **4-page Streamlit UI**: Prediction, Expert System, Data Explorer, Model Comparison
- **Model Comparison**: Radar chart, ROC Curve, metrics table (DT vs Expert System)

## Folder Structure

```
Heart_Disease_Detection/
├── data/
│   ├── raw_data.csv
│   └── cleaned_data.csv
├── notebooks/
│   └── data_analysis.py
├── rule_based_system/
│   ├── rules.py          (20 clinical rules)
│   └── expert_system.py
├── ml_model/
│   ├── train_model.py
│   ├── predict.py
│   ├── decision_tree_model.pkl
│   └── scaler.pkl
├── utils/
│   └── data_processing.py
├── reports/
│   └── accuracy_comparison.md
├── ui/
│   └── app.py
├── README.md
└── requirements.txt
```

## Dataset

- **Source**: UCI Heart Disease Dataset
- **Raw records**: 1,025 rows
- **After deduplication**: 305 unique rows
- **Features**: 13 clinical attributes (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
- **Target**: 1 = Heart Disease, 0 = No Disease
