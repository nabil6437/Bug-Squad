<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Heart%20Disease%20Detection&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=38&desc=AI-Powered%20Clinical%20Risk%20Assessment%20System&descAlignY=60&descSize=16" width="100%"/>

<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&pause=1000&color=14B8A6&center=true&vCenter=true&width=700&lines=Rule-Based+Expert+System;Decision+Tree+Classifier;Interactive+Streamlit+Dashboard;Full+Preprocessing+Pipeline" alt="Typing SVG" />
</a>

<br/>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.15%2B-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

## Table of Contents

<details open>
<summary>Click to expand</summary>

- [About the Project](#about-the-project)
- [Features](#features)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)

</details>

---

## About the Project

<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=6,11,20&height=3" width="100%"/>

A **Heart Disease Risk Prediction System** that combines a **Rule-Based Expert System** and a **Machine Learning Decision Tree** to analyze patient health indicators and predict cardiovascular risk, wrapped in an interactive Streamlit dashboard.

```
Clinical Data  →  Preprocessing Pipeline  →  Expert System + Decision Tree  →  Risk Assessment
```

---

## Features

<table>
<tr>
<td width="50%">

**Machine Learning**
- Decision Tree Classifier (Scikit-Learn)
- GridSearchCV Hyperparameter Tuning
- 80/20 Stratified Train/Test Split
- Accuracy, Precision, Recall, F1-Score

</td>
<td width="50%">

**Expert System**
- 20 Clinical Rules using Experta
- Forward-Chaining Inference Engine
- HIGH / MEDIUM / LOW Risk Levels
- Full Decision Trace and Explanation

</td>
</tr>
<tr>
<td>

**Preprocessing Pipeline**
- Duplicate Removal (720 rows removed)
- Missing Value Imputation (median/mode)
- IQR Outlier Capping (Winsorization)
- One-Hot Encoding (5 categorical cols)
- MinMaxScaler (continuous cols only)

</td>
<td>

**Streamlit Dashboard**
- Patient Risk Assessment Page
- Expert System Inference Page
- Interactive Data Explorer
- Model Comparison (Radar + ROC Curve)

</td>
</tr>
</table>

---

## Model Performance

<div align="center">

| Metric | Decision Tree | Expert System |
|:------:|:------------:|:-------------:|
| **Accuracy** | `77.1%` | `~64.0%` |
| **Precision** | `75.0%` | `~57.0%` |
| **Recall** | `75.0%` | `~93.0%` |
| **F1-Score** | `75.0%` | `~70.0%` |

</div>

> The Expert System has higher recall — catches more disease cases — making it better as a screening tool. The Decision Tree offers more balanced overall performance.

<details>
<summary>Best Hyperparameters (GridSearchCV)</summary>

```python
{
  "criterion":         "entropy",
  "max_depth":         5,
  "min_samples_leaf":  4,
  "min_samples_split": 2
}
# CV Folds: 5  |  Scoring: F1  |  Train: 244 samples  |  Test: 61 samples
```

</details>

---

## Project Structure

```
Heart_Disease_Detection/
│
├── data/
│   ├── raw_data.csv                  UCI Heart Disease Dataset (1,025 rows)
│   └── cleaned_data.csv              After full preprocessing (305 rows, 28 cols)
│
├── notebooks/
│   ├── data_analysis.ipynb           Visualization and EDA
│   └── model_training.ipynb          Full training walkthrough
│
├── rule_based_system/
│   ├── rules.py                      20 Clinical Rules (Experta)
│   └── expert_system.py              CLI interface
│
├── ml_model/
│   ├── train_model.py                Training script with GridSearchCV
│   ├── predict.py                    Inference with OHE + scaling
│   ├── decision_tree_model.pkl       Trained model
│   ├── scaler.pkl                    MinMaxScaler
│   ├── ohe_encoder.pkl               OneHotEncoder
│   └── model_meta.json               Feature names and config
│
├── utils/
│   └── data_processing.py            9-step preprocessing pipeline
│
├── reports/
│   ├── accuracy_comparison.md        DT vs Expert System comparison
│   ├── ml_metrics.json               Model evaluation results
│   └── *.png                         Visualization outputs
│
├── ui/
│   └── app.py                        4-page Streamlit dashboard
│
├── README.md
└── requirements.txt
```

---

## Setup and Installation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![pip](https://img.shields.io/badge/pip-latest-orange?style=flat-square)

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/Heart_Disease_Detection.git
cd Heart_Disease_Detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the Decision Tree model
python ml_model/train_model.py

# 4. Launch the Streamlit dashboard
streamlit run ui/app.py
```

---

## Usage

<details>
<summary>Run the Streamlit Dashboard</summary>

```bash
streamlit run ui/app.py
```

Open `http://localhost:8501` in your browser. The app has 4 pages:

| Page | Description |
|------|-------------|
| Prediction | Enter patient data and get a Decision Tree prediction with probability gauge |
| Expert System | Get a rule-based diagnosis with full inference trace |
| Data Explorer | Explore the dataset with interactive charts and correlation heatmap |
| Model Comparison | Side-by-side Radar chart, ROC Curve, and metrics table |

</details>

<details>
<summary>Run the Expert System via CLI</summary>

```bash
python rule_based_system/expert_system.py
```

</details>

<details>
<summary>Run Data Analysis Notebook</summary>

```bash
jupyter notebook notebooks/data_analysis.ipynb
```

</details>

---

## How It Works

**Preprocessing Pipeline — 9 Steps**

```
Step 1   Load raw CSV (1,025 rows)
Step 2   Remove duplicates          → 305 unique rows remain
Step 3   Fix target encoding        → 1 = Disease, 0 = No Disease
Step 4   Impute missing values      → median for continuous, mode for categorical
Step 5   Validate categorical cols  → remove rows with invalid values
Step 6   IQR Outlier Capping        → Winsorization on continuous features
Step 7   One-Hot Encoding           → cp, restecg, slope, ca, thal → 19 new cols
Step 8   MinMaxScaler               → age, trestbps, chol, thalach, oldpeak
Step 9   Correlation report         → feature importance vs target
```

**Expert System — Risk Scoring**

```
HIGH   rule fired  →  +3 points
MEDIUM rule fired  →  +1 point
LOW    rule fired  →  -2 points

Final level: any HIGH fired = HIGH risk
             any MEDIUM (no HIGH) = MEDIUM risk
             otherwise = LOW risk
```

---

## Dataset

| Property | Value |
|----------|-------|
| Source | UCI Heart Disease Dataset |
| Raw records | 1,025 rows |
| After deduplication | 305 unique rows |
| Features | 13 clinical attributes |
| Target | 1 = Heart Disease, 0 = No Disease |
| Class balance | 45.9% Disease / 54.1% Healthy |

**Features used:**

```
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
```

---

## Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer" width="100%"/>

**Heart Disease Detection System — Expert Systems Project**

</div>
