import pandas as pd
import numpy as np
import os ,sys ,json
from sklearn .metrics import accuracy_score ,precision_score ,recall_score ,f1_score

sys .path .insert (0 ,os .path .join (os .path .dirname (__file__ ),'..'))
from rule_based_system .rules import run_expert_system
import joblib

BASE =os .path .join (os .path .dirname (__file__ ),'..')
DATA_PATH =os .path .join (BASE ,'data','raw_data.csv')
MODEL_PATH =os .path .join (BASE ,'ml_model','decision_tree_model.pkl')
REPORT_PATH =os .path .join (os .path .dirname (__file__ ),'accuracy_comparison.md')

df =pd .read_csv (DATA_PATH )
df =df .drop_duplicates ().reset_index (drop =True )
df ['target']=1 -df ['target']
X =df .drop ('target',axis =1 )
y =df ['target']

from sklearn .model_selection import train_test_split
_ ,X_test ,_ ,y_test =train_test_split (X ,y ,test_size =0.2 ,random_state =42 ,stratify =y )

model =joblib .load (MODEL_PATH )
ml_preds =model .predict (X_test )

expert_preds =[]
for _ ,row in X_test .iterrows ():
    patient =row .to_dict ()
    risk ,_ =run_expert_system (patient )
    if risk =="high":
        expert_preds .append (1 )
    elif risk =="medium":
        expert_preds .append (1 )
    else :
        expert_preds .append (0 )
expert_preds =np .array (expert_preds )

def calc_metrics (y_true ,y_pred ):
    return {
    'Accuracy':round (accuracy_score (y_true ,y_pred ),4 ),
    'Precision':round (precision_score (y_true ,y_pred ,zero_division =0 ),4 ),
    'Recall':round (recall_score (y_true ,y_pred ,zero_division =0 ),4 ),
    'F1-Score':round (f1_score (y_true ,y_pred ,zero_division =0 ),4 )
    }

ml_metrics =calc_metrics (y_test ,ml_preds )
expert_metrics =calc_metrics (y_test ,expert_preds )

report =f"""# Accuracy Comparison Report

## Decision Tree Model vs Rule-Based Expert System

| Metric | Decision Tree | Expert System |
|--------|--------------|---------------|
| Accuracy | {ml_metrics ['Accuracy']} | {expert_metrics ['Accuracy']} |
| Precision | {ml_metrics ['Precision']} | {expert_metrics ['Precision']} |
| Recall | {ml_metrics ['Recall']} | {expert_metrics ['Recall']} |
| F1-Score | {ml_metrics ['F1-Score']} | {expert_metrics ['F1-Score']} |

## Analysis

### Decision Tree Model
- Learns patterns directly from data
- Handles complex feature interactions automatically
- Performance depends on training data quality and hyperparameter tuning
- Less interpretable as tree depth increases

### Rule-Based Expert System
- Uses human-defined medical rules
- Fully transparent and explainable
- Limited to predefined rule combinations
- Does not adapt to new data patterns

### Conclusion
The Decision Tree model achieves **{ml_metrics ['Accuracy']*100 :.1f}%** accuracy while the Expert System achieves **{expert_metrics ['Accuracy']*100 :.1f}%** accuracy on the test set.
"""

with open (REPORT_PATH ,'w')as f :
    f .write (report )

print (report )
