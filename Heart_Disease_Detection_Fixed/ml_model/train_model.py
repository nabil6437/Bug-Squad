import pandas as pd
import numpy as np
import os ,sys ,json
from sklearn .model_selection import train_test_split ,GridSearchCV
from sklearn .tree import DecisionTreeClassifier
from sklearn .metrics import (accuracy_score ,precision_score ,
recall_score ,f1_score ,classification_report )
import joblib

BASE =os .path .join (os .path .dirname (__file__ ),'..')
sys .path .insert (0 ,BASE )

from utils .data_processing import run_pipeline ,FEATURE_COLS

DATA_PATH =os .path .join (BASE ,'data','raw_data.csv')
CLEANED_PATH =os .path .join (BASE ,'data','cleaned_data.csv')
MODEL_PATH =os .path .join (os .path .dirname (__file__ ),'decision_tree_model.pkl')
SCALER_PATH =os .path .join (os .path .dirname (__file__ ),'scaler.pkl')
ENCODER_PATH =os .path .join (os .path .dirname (__file__ ),'ohe_encoder.pkl')
META_PATH =os .path .join (os .path .dirname (__file__ ),'model_meta.json')
METRICS_PATH =os .path .join (BASE ,'reports','ml_metrics.json')

df_clean ,df_scaled ,scaler ,encoder =run_pipeline (DATA_PATH ,CLEANED_PATH ,scale =True )

FEATURE_COLS_OHE =[c for c in df_scaled .columns if c !='target']
X =df_scaled [FEATURE_COLS_OHE ]
y =df_scaled ['target']

X_train ,X_test ,y_train ,y_test =train_test_split (
X ,y ,test_size =0.2 ,random_state =42 ,stratify =y
)
print (f"\nTrain: {len (X_train )} | Test: {len (X_test )}")

param_grid ={
'max_depth':[3 ,5 ,7 ,10 ,None ],
'min_samples_split':[2 ,5 ,10 ],
'min_samples_leaf':[1 ,2 ,4 ],
'criterion':['gini','entropy'],
}

print ("\nجاري Hyperparameter Tuning ...")
grid_search =GridSearchCV (
DecisionTreeClassifier (random_state =42 ),
param_grid ,cv =5 ,scoring ='f1',n_jobs =-1 ,verbose =0
)
grid_search .fit (X_train ,y_train )

best_model =grid_search .best_estimator_
print (f"أفضل parameters : {grid_search .best_params_ }")
print (f"أفضل CV F1      : {grid_search .best_score_ :.4f}")

y_pred =best_model .predict (X_test )

metrics ={
'accuracy':round (accuracy_score (y_test ,y_pred ),4 ),
'precision':round (precision_score (y_test ,y_pred ),4 ),
'recall':round (recall_score (y_test ,y_pred ),4 ),
'f1_score':round (f1_score (y_test ,y_pred ),4 ),
'best_params':grid_search .best_params_ ,
'train_size':len (X_train ),
'test_size':len (X_test ),
}

print ("\n=== تقييم Decision Tree ===")
print (classification_report (y_test ,y_pred ,target_names =["No Disease","Disease"]))

joblib .dump (best_model ,MODEL_PATH )
joblib .dump (scaler ,SCALER_PATH )
joblib .dump (encoder ,ENCODER_PATH )
print (f"Model   saved → {MODEL_PATH }")
print (f"Scaler  saved → {SCALER_PATH }")
print (f"Encoder saved → {ENCODER_PATH }")

meta ={
'feature_cols':FEATURE_COLS_OHE ,
'original_feature_cols':FEATURE_COLS ,
'target_map':{'0':'No Disease','1':'Disease'},
'continuous_cols':['age','trestbps','chol','thalach','oldpeak'],
'ohe_cols':['cp','restecg','slope','ca','thal'],
'binary_cols':['sex','fbs','exang'],
'sklearn_version':__import__ ('sklearn').__version__ ,
}
with open (META_PATH ,'w')as f :
    json .dump (meta ,f ,indent =2 )

os .makedirs (os .path .dirname (METRICS_PATH ),exist_ok =True )
with open (METRICS_PATH ,'w')as f :
    json .dump (metrics ,f ,indent =2 )

print (f"Meta + Metrics saved ✓")
print (f"\n✅ التدريب اكتمل | Accuracy: {metrics ['accuracy']*100 :.1f}% | F1: {metrics ['f1_score']*100 :.1f}%")
